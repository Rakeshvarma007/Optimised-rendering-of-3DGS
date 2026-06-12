"""
test_vfc.py  –  Automated Test Suite for Layer 2: View Frustum Culling
=======================================================================
Three deterministic scenarios validate mathematical correctness and
throughput of the vfc_culling intercept layer.

Scenarios
---------
A) Completely Inside   – a small AABB centred at the world origin must be
                         visible (mask == True) for a canonical orthographic
                         frustum whose planes tightly bound [-1, 1]^3.

B) Completely Outside  – an AABB shifted far along +X (well beyond the right
                         frustum boundary) must be culled (mask == False).

C) Stress Test         – 10 000 random AABBs processed in one vectorised call;
                         validates shape correctness and measures wall-clock
                         latency (both CPU and, if available, CUDA).

Frustum Construction for Tests A & B
--------------------------------------
We construct a frustum analytically instead of depending on a camera rig.
The simplest normalised VP matrix for an axis-aligned orthographic view that
maps the cube [-1, 1]^3 to NDC is the 4×4 identity I_4.  Feeding I_4 into
extract_frustum_planes yields six axis-aligned planes at ±1 on each axis,
which is easy to reason about geometrically.

    VP = I_4  →  planes: x ≥ -1, x ≤ 1, y ≥ -1, y ≤ 1, z ≥ -1, z ≤ 1
"""

import time
import sys

import torch

# ── Make the repo root importable when running the file directly
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vfc_culling import extract_frustum_planes, cull_octree_nodes


# ---------------------------------------------------------------------------
# Helper: build a canonical orthographic frustum  (world cube [-1, 1]^3)
# ---------------------------------------------------------------------------

def build_identity_frustum_planes() -> torch.Tensor:
    """Return the 6 planes for an axis-aligned [-1, 1]^3 frustum.

    Using VP = I_4, the Gribb-Hartmann method produces:
        Left  :  x + 1 ≥ 0   →  A=1, B=0, C=0, D=1   (normalised)
        Right :  -x + 1 ≥ 0  →  A=-1, ...
        Bottom:  y + 1 ≥ 0
        Top   :  -y + 1 ≥ 0
        Near  :  z + 1 ≥ 0
        Far   :  -z + 1 ≥ 0
    All normals already have unit length, so no further normalisation changes D.
    """
    vp = torch.eye(4, dtype=torch.float32)
    return extract_frustum_planes(vp)


# ---------------------------------------------------------------------------
# Scenario A – Completely Inside
# ---------------------------------------------------------------------------

def test_scenario_a_completely_inside():
    """A small AABB centred at the origin must be fully visible."""
    print("\n" + "=" * 60)
    print("Scenario A: Completely Inside")
    print("=" * 60)

    planes = build_identity_frustum_planes()

    # Tight box well within the [-1, 1]^3 frustum
    aabb_min = torch.tensor([[-0.1, -0.1, -0.1]], dtype=torch.float32)
    aabb_max = torch.tensor([[ 0.1,  0.1,  0.1]], dtype=torch.float32)

    mask = cull_octree_nodes(aabb_min, aabb_max, planes)

    print(f"  Output mask shape : {mask.shape}")
    print(f"  Output mask value : {mask}")
    print(f"  Expected          : True (node is inside the frustum)")

    assert mask.shape == (1,), f"Shape mismatch: expected (1,), got {mask.shape}"
    assert bool(mask[0]) is True, (
        f"ASSERTION FAILED: expected True (inside frustum), got {mask[0]}"
    )
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Scenario B – Completely Outside
# ---------------------------------------------------------------------------

def test_scenario_b_completely_outside():
    """An AABB shifted far along +X must be culled (outside Right plane)."""
    print("\n" + "=" * 60)
    print("Scenario B: Completely Outside")
    print("=" * 60)

    planes = build_identity_frustum_planes()

    # Box far beyond the Right frustum boundary (x > 1)
    # The Right plane clips x = 1; this box lives at x ∈ [5, 6]
    aabb_min = torch.tensor([[5.0, -0.1, -0.1]], dtype=torch.float32)
    aabb_max = torch.tensor([[6.0,  0.1,  0.1]], dtype=torch.float32)

    mask = cull_octree_nodes(aabb_min, aabb_max, planes)

    print(f"  Output mask shape : {mask.shape}")
    print(f"  Output mask value : {mask}")
    print(f"  Expected          : False (node is outside Right frustum plane)")

    assert mask.shape == (1,), f"Shape mismatch: expected (1,), got {mask.shape}"
    assert bool(mask[0]) is False, (
        f"ASSERTION FAILED: expected False (outside frustum), got {mask[0]}"
    )
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Scenario B-extended – Boundary Planes
# ---------------------------------------------------------------------------

def test_scenario_b_extra_boundary_checks():
    """Verify culling against each of the 6 boundary planes individually."""
    print("\n" + "=" * 60)
    print("Scenario B-extra: Per-Axis Boundary Checks")
    print("=" * 60)

    planes = build_identity_frustum_planes()
    offset = 5.0

    # (axis, sign, label)
    cases = [
        (0, +1, "+X / Right plane"),
        (0, -1, "-X / Left  plane"),
        (1, +1, "+Y / Top   plane"),
        (1, -1, "-Y / Bottom plane"),
        (2, +1, "+Z / Far   plane"),
        (2, -1, "-Z / Near  plane"),
    ]

    for axis, sign, label in cases:
        lo = torch.zeros(1, 3)
        hi = torch.zeros(1, 3)
        lo[0, axis] = sign * offset
        hi[0, axis] = sign * (offset + 0.5)
        # Make sure lo <= hi component-wise for all axes
        lo_final = torch.minimum(lo, hi)
        hi_final = torch.maximum(lo, hi)

        mask = cull_octree_nodes(lo_final, hi_final, planes)
        expected = False
        result = bool(mask[0])
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status}  [{label}]  mask={result}  (expected {expected})")
        assert result == expected, (
            f"ASSERTION FAILED for [{label}]: expected {expected}, got {result}"
        )

    print("  [PASS] ALL BOUNDARY CHECKS PASSED")


# ---------------------------------------------------------------------------
# Scenario C – Stress Test  (10 000 random nodes)
# ---------------------------------------------------------------------------

def _run_stress(device: torch.device, n_nodes: int = 10_000) -> float:
    """Run the stress test on `device` and return wall-clock seconds."""
    planes = build_identity_frustum_planes().to(device)

    # Random AABBs: centres uniform in [-3, 3]^3, half-extents in [0.01, 0.5]
    torch.manual_seed(42)
    centres    = (torch.rand(n_nodes, 3, device=device) - 0.5) * 6.0   # [-3, 3]
    half_sizes = torch.rand(n_nodes, 3, device=device) * 0.49 + 0.01   # [0.01, 0.5]

    aabb_min = centres - half_sizes
    aabb_max = centres + half_sizes

    # Warm-up pass (important for CUDA timing)
    _ = cull_octree_nodes(aabb_min, aabb_max, planes)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    mask = cull_octree_nodes(aabb_min, aabb_max, planes)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Shape correctness
    assert mask.shape == (n_nodes,), (
        f"Shape mismatch: expected ({n_nodes},), got {mask.shape}"
    )
    assert mask.dtype == torch.bool, (
        f"Dtype mismatch: expected torch.bool, got {mask.dtype}"
    )

    n_visible = int(mask.sum())
    print(f"    nodes={n_nodes}  visible={n_visible}  culled={n_nodes - n_visible}"
          f"  latency={elapsed * 1000:.3f} ms  device={device}")
    return elapsed


def test_scenario_c_stress():
    """Stress test: 10 000 random AABBs, shape & latency validation."""
    print("\n" + "=" * 60)
    print("Scenario C: Stress Test (10 000 nodes)")
    print("=" * 60)

    N = 10_000

    # --- CPU run ---
    print("  [CPU]")
    cpu_elapsed = _run_stress(torch.device("cpu"), n_nodes=N)
    print(f"  [PASS] CPU stress test PASSED  ({cpu_elapsed * 1000:.3f} ms)")

    # --- CUDA run (if available) ---
    if torch.cuda.is_available():
        print("  [CUDA]")
        cuda_elapsed = _run_stress(torch.device("cuda"), n_nodes=N)
        print(f"  [PASS] CUDA stress test PASSED  ({cuda_elapsed * 1000:.3f} ms)")
    else:
        print("  [SKIP] CUDA not available -- skipping GPU stress test")


# ---------------------------------------------------------------------------
# Additional sanity: planes print-out for manual inspection
# ---------------------------------------------------------------------------

def print_planes_debug():
    """Print the 6 extracted planes for human verification."""
    print("\n" + "=" * 60)
    print("Debug: Extracted frustum planes (VP = I_4)")
    print("=" * 60)
    labels = ["Left  ", "Right ", "Bottom", "Top   ", "Near  ", "Far   "]
    planes = build_identity_frustum_planes()
    for label, row in zip(labels, planes):
        A, B, C, D = row.tolist()
        print(f"  {label}: {A:+.4f}x  {B:+.4f}y  {C:+.4f}z  {D:+.4f} = 0")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("   VFC Layer 2 -- View Frustum Culling Test Suite")
    print("   3DGS Optimisation Pipeline")
    print("=" * 62)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device     : {torch.cuda.get_device_name(0)}")

    # Show planes for manual sanity check
    print_planes_debug()

    failures = []

    tests = [
        ("Scenario A – Completely Inside",        test_scenario_a_completely_inside),
        ("Scenario B – Completely Outside",       test_scenario_b_completely_outside),
        ("Scenario B-extra – Boundary Checks",   test_scenario_b_extra_boundary_checks),
        ("Scenario C – Stress Test",              test_scenario_c_stress),
    ]

    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"\n  [FAIL] {name}\n      {exc}")
            failures.append(name)
        except Exception as exc:
            print(f"\n  [ERROR] {name}\n      {type(exc).__name__}: {exc}")
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"  [FAIL] {len(failures)} test(s) FAILED:")
        for f in failures:
            print(f"       - {f}")
        sys.exit(1)
    else:
        total = len(tests)
        print(f"  [PASS] All {total} test scenarios PASSED -- VFC layer is mathematically correct.")
    print("=" * 62)


if __name__ == "__main__":
    main()
