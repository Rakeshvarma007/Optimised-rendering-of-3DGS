"""Diagnostic: LOD merged splat activation vs leaf (writes debug-c361f6.log)."""
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Build octree without pulling gsplat CUDA (lod_octree optionally imports gsplat.exporter).
import importlib.util

_lod_spec = importlib.util.spec_from_file_location(
    "lod_octree_local", os.path.join(ROOT, "lod_octree.py")
)
_lod_mod = importlib.util.module_from_spec(_lod_spec)
# Prevent gsplat package init when exporter import runs.
sys.modules["gsplat"] = type(sys)("gsplat")
sys.modules["gsplat.exporter"] = type(sys)("gsplat.exporter")
_lod_spec.loader.exec_module(_lod_mod)
LODOctree = _lod_mod.LODOctree


def log_entry(run_id: str, hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "c361f6",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "scripts/diag_lod_activation.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    path = os.path.join(ROOT, "debug-c361f6.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    n = 64
    points = torch.randn(n, 3, device=device) * 2.0
    scales_log = torch.randn(n, 3, device=device) * 0.5 - 2.0
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities_logit = torch.randn(n, device=device)
    sh0 = torch.rand(n, 1, 3, device=device)

    tree = LODOctree(
        points=points,
        scales=scales_log,
        quats=quats,
        opacities=opacities_logit,
        sh0=sh0,
        max_depth=4,
        min_gaussians=1000,
        raw_inputs=True,
    )
    tree.build()

    scales_act = torch.exp(scales_log)
    opacities_act = torch.sigmoid(opacities_logit)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.eye(3, device=device).unsqueeze(0)

    # Log expected bug signature from first merged node
    merged_raw_scale = None
    merged_raw_opacity = None

    def walk(node):
        nonlocal merged_raw_scale, merged_raw_opacity
        if node.merged_splat_data is not None and merged_raw_scale is None:
            merged_raw_scale = node.merged_splat_data["scales"].detach().float().mean().item()
            merged_raw_opacity = (
                node.merged_splat_data["opacities"].detach().float().mean().item()
            )
        for c in node.children:
            walk(c)

    walk(tree.root)

    log_entry(
        "pre-fix-sim",
        "A",
        "Merged node stores log/logit; leaf inputs are activated",
        {
            "merged_scale_raw_mean": merged_raw_scale,
            "merged_scale_exp_mean": float(torch.exp(scales_log).mean().item())
            if merged_raw_scale is not None
            else None,
            "leaf_scale_activated_mean": float(scales_act.mean().item()),
            "merged_opacity_raw_mean": merged_raw_opacity,
            "leaf_opacity_activated_mean": float(opacities_act.mean().item()),
            "bug_scale_ratio_if_unactivated": (
                merged_raw_scale / float(scales_act.mean().item())
                if merged_raw_scale is not None
                else None
            ),
        },
    )

    # Mirror rasterization_lod merged-branch activation (post-fix path)
    data = tree.root.merged_splat_data
    merged_out_scale_mean = None
    merged_out_opacity_mean = None
    if data is not None:
        ms = torch.exp(data["scales"]) if tree.raw_inputs else data["scales"]
        mo = torch.sigmoid(data["opacities"].squeeze(-1)) if tree.raw_inputs else data["opacities"]
        merged_out_scale_mean = float(ms.float().mean().item())
        merged_out_opacity_mean = float(mo.float().mean().item())

    log_entry(
        "post-fix",
        "A",
        "Merged tensors after exp/sigmoid align with leaf activated stats",
        {
            "merged_scale_out_mean": merged_out_scale_mean,
            "leaf_scale_activated_mean": float(scales_act.mean().item()),
            "scale_out_vs_leaf_ratio": (
                merged_out_scale_mean / float(scales_act.mean().item())
                if merged_out_scale_mean is not None
                else None
            ),
            "merged_opacity_out_mean": merged_out_opacity_mean,
            "leaf_opacity_activated_mean": float(opacities_act.mean().item()),
        },
    )


if __name__ == "__main__":
    main()
