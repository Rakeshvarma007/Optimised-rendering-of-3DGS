import torch
import math
import os
import argparse
import sys
from typing import Dict, List, Tuple, Optional

# Try to import from gsplat, otherwise define fallbacks
try:
    from gsplat.exporter import load_ply_to_splats, export_splats
    from gsplat.utils import normalized_quat_to_rotmat
except ImportError:
    load_ply_to_splats = None
    export_splats = None
    normalized_quat_to_rotmat = None


def custom_quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized quaternion in wxyz convention to rotation matrix.
    Supports both batched and unbatched input.
    """
    assert quat.shape[-1] == 4, quat.shape
    # Normalize
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix R [..., 3, 3] to a unit quaternion [..., 4] in wxyz convention.
    Handles arbitrary batch shapes and is numerically robust.
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(N, 4, dtype=R.dtype, device=R.device)

    # Case 1: Trace > 0
    mask1 = tr > 0
    if mask1.any():
        S = torch.sqrt(tr[mask1] + 1.0) * 2.0
        q[mask1, 0] = 0.25 * S
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S

    # Case 2: R00 is max diagonal element
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        S = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S
        q[mask2, 1] = 0.25 * S
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S

    # Case 3: R11 is max diagonal element
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        S = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S
        q[mask3, 2] = 0.25 * S
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S

    # Case 4: R22 is max diagonal element
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        S = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S
        q[mask4, 3] = 0.25 * S

    # Normalize quaternion to prevent drifts
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    return q.reshape(*shape, 4)


def safe_logit(x: torch.Tensor) -> torch.Tensor:
    """Compute division-by-zero safe logit representation."""
    x = torch.clamp(x, min=1e-7, max=1.0 - 1e-7)
    return torch.log(x / (1.0 - x))


class OctreeNode:
    def __init__(self, aabb: Tuple[torch.Tensor, torch.Tensor], depth: int = 0, parent: Optional['OctreeNode'] = None):
        """
        aabb: Tuple of (min_bounds, max_bounds), both torch.Tensors of shape (3,)
        """
        self.aabb = aabb
        self.depth = depth
        self.parent = parent
        self.children: List[OctreeNode] = []
        self.original_splat_indices: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.merged_splat_data: Optional[Dict[str, torch.Tensor]] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def aabb_volume(self) -> float:
        size = self.aabb[1] - self.aabb[0]
        return float(torch.prod(size).item())


class LODOctree:
    def __init__(
        self,
        points: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: Optional[torch.Tensor] = None,
        max_depth: int = 4,
        min_gaussians: int = 8,
        tau: float = 1e-4,
        epsilon_prune: float = 1e-3,
        gamma: float = 1.0,
        alpha_max: float = 0.99,
        raw_inputs: bool = True,
    ):
        """
        points: (N, 3) xyz positions
        scales: (N, 3) scale parameters (log-scale if raw_inputs=True)
        quats: (N, 4) quaternions (unnormalized if raw_inputs=True)
        opacities: (N, 1) or (N,) opacities (logits if raw_inputs=True)
        sh0: (N, 1, 3) or (N, 3) Degree 0 Spherical Harmonics
        shN: (N, K, 3) or (N, K*3) higher-order SH (optional)
        """
        self.points = points.detach().float()
        self.raw_inputs = raw_inputs

        # Keep everything in memory. Exponentiate/sigmoid if raw_inputs is True
        if raw_inputs:
            self.scales = torch.exp(scales.detach().float())
            self.opacities = torch.sigmoid(opacities.detach().float())
        else:
            self.scales = scales.detach().float()
            self.opacities = opacities.detach().float()

        # Ensure opacities is 1D or (N,) for uniform blending
        if self.opacities.ndim == 2 and self.opacities.shape[-1] == 1:
            self.opacities = self.opacities.squeeze(-1)

        # Normalize quats
        self.quats = quats.detach().float()
        self.quats = self.quats / (torch.norm(self.quats, dim=-1, keepdim=True) + 1e-8)

        # SH representations
        self.sh0 = sh0.detach().float()
        if self.sh0.ndim == 3:
            self.sh0 = self.sh0.squeeze(1)  # (N, 3)

        self.shN = shN.detach().float() if shN is not None else None
        if self.shN is not None and self.shN.ndim == 3:
            self.shN = self.shN.permute(0, 2, 1).reshape(len(points), -1)  # (N, K*3)

        self.max_depth = max_depth
        self.min_gaussians = min_gaussians
        self.tau = tau
        self.epsilon_prune = epsilon_prune
        self.gamma = gamma
        self.alpha_max = alpha_max

        # Scene bounding box
        min_bounds = torch.min(self.points, dim=0).values
        max_bounds = torch.max(self.points, dim=0).values
        # Add small buffer to make boundaries robust
        size = max_bounds - min_bounds
        min_bounds = min_bounds - 0.01 * size
        max_bounds = max_bounds + 0.01 * size

        self.root = OctreeNode((min_bounds, max_bounds), depth=0)
        self.root.original_splat_indices = torch.arange(len(points), device=points.device)

    def build(self):
        """Construct the Octree hierarchy and perform bottom-up merging."""
        print(f"Building Octree spatial hierarchy (max_depth={self.max_depth}, min_gaussians={self.min_gaussians})...")
        self._split_node(self.root)
        print("Executing bottom-up parameter merging and visual importance pruning...")
        self._merge_node(self.root)
        print("LOD Spatial Hierarchy construction completed successfully.")

    def _split_node(self, node: OctreeNode):
        """Recursively partitions points into 8 sub-octants until limits are reached."""
        indices = node.original_splat_indices

        # Base conditions: reached max depth or not enough points to justify splitting
        if node.depth >= self.max_depth or len(indices) <= self.min_gaussians:
            return

        min_b, max_b = node.aabb
        center = (min_b + max_b) / 2.0
        
        # Get coordinates of points belonging to this node
        node_points = self.points[indices]
        
        # Fast vectorized sorting into 8 octants based on center
        x_mask = node_points[:, 0] >= center[0]
        y_mask = node_points[:, 1] >= center[1]
        z_mask = node_points[:, 2] >= center[2]

        for i in range(8):
            # Compute bitmask for current octant (0-7)
            b_x = (i & 1) > 0
            b_y = (i & 2) > 0
            b_z = (i & 4) > 0

            # Find points matching this octant's spatial condition
            octant_mask = (x_mask == b_x) & (y_mask == b_y) & (z_mask == b_z)
            octant_indices = indices[octant_mask]

            # Only create child nodes if they contain points
            if len(octant_indices) > 0:
                # Calculate child AABB boundaries
                c_min = min_b.clone()
                c_max = max_b.clone()
                
                if b_x: c_min[0] = center[0]
                else:   c_max[0] = center[0]
                
                if b_y: c_min[1] = center[1]
                else:   c_max[1] = center[1]
                
                if b_z: c_min[2] = center[2]
                else:   c_max[2] = center[2]

                # Instantiate and attach child
                child = OctreeNode(aabb=(c_min, c_max), depth=node.depth + 1, parent=node)
                child.original_splat_indices = octant_indices
                node.children.append(child)
                
                # Recurse
                self._split_node(child)

    def _merge_node(self, node: OctreeNode):
        # 1. Recursive bottom-up tree traversal
        for child in node.children:
            self._merge_node(child)

        # 2. Gather and condition child or leaf tensors
        if node.is_leaf:
            indices = node.original_splat_indices
            if len(indices) == 0:
                return
                
            means = self.points[indices].view(-1, 3)
            
            # CRITICAL FIX: Normalize quaternions explicitly
            raw_quats = self.quats[indices].view(-1, 4)
            quats = raw_quats / (torch.norm(raw_quats, dim=-1, keepdim=True) + 1e-8)
            
            # FORCE ACTIVATION: Extract out of PLY optimization spaces
            scales = torch.exp(self.scales[indices].view(-1, 3))
            opacities = torch.sigmoid(self.opacities[indices].view(-1))
            
            # Handle Spherical Harmonics
            sh0_raw = self.sh0[indices]
            if sh0_raw.ndim == 3 and sh0_raw.shape[1] == 1:
                sh0_raw = sh0_raw.squeeze(1)
            sh0 = sh0_raw.view(-1, 3)
            
            shN = self.shN[indices] if self.shN is not None else None
            if shN is not None:
                shN = shN.view(means.shape[0], -1, 3)
        else:
            # Internal Node Processing
            valid_children = [child for child in node.children if child.merged_splat_data is not None]
            if len(valid_children) == 0: 
                return

            child_means, child_scales, child_quats, child_opacities, child_sh0 = [], [], [], [], []
            child_shN = []

            max_shN_dim = 0
            has_shN = False
            for child in valid_children:
                shN_data = child.merged_splat_data.get("shN")
                if shN_data is not None:
                    max_shN_dim = max(max_shN_dim, shN_data.shape[1])
                    has_shN = True

            for child in valid_children:
                data = child.merged_splat_data
                
                child_means.append(data["means"].view(-1, 3))
                child_quats.append(data["quats"].view(-1, 4))
                child_sh0.append(data["sh0"].view(-1, 3))
                
                # FORCE ACTIVATION: Child node values are stored re-encoded; lift back to linear space
                child_scales.append(torch.exp(data["scales"].view(-1, 3)))
                child_opacities.append(torch.sigmoid(data["opacities"].view(-1)))
                
                if has_shN:
                    curr_sh = data.get("shN")
                    if curr_sh is None:
                        curr_sh = torch.zeros((1, max_shN_dim, 3), device=data["means"].device, dtype=data["means"].dtype)
                    else:
                        curr_sh = curr_sh.view(1, -1, 3)
                        if curr_sh.shape[1] < max_shN_dim:
                            pad_size = max_shN_dim - curr_sh.shape[1]
                            padding = torch.zeros((1, pad_size, 3), device=curr_sh.device, dtype=curr_sh.dtype)
                            curr_sh = torch.cat([curr_sh, padding], dim=1)
                    child_shN.append(curr_sh)

            means = torch.cat(child_means, dim=0)
            scales = torch.cat(child_scales, dim=0)
            quats = torch.cat(child_quats, dim=0)
            opacities = torch.cat(child_opacities, dim=0)
            sh0 = torch.cat(child_sh0, dim=0)
            shN = torch.cat(child_shN, dim=0) if has_shN else None

        # 3. Structural Importance-Based Pruning Protocol
        prod_scales = torch.prod(scales, dim=-1)
        importance_scores = opacities * torch.clamp(prod_scales / self.tau, max=1.0)
        keep_mask = importance_scores >= self.epsilon_prune

        if not keep_mask.any():
            best_idx = torch.argmax(importance_scores)
            keep_mask[best_idx] = True

        means = means[keep_mask]
        scales = scales[keep_mask]
        quats = quats[keep_mask]
        opacities = opacities[keep_mask]
        sh0 = sh0[keep_mask]
        if shN is not None:
            shN = shN[keep_mask]

        # 4. Mathematical Merging Operations (Linear Physical Space)
        sum_alpha = torch.sum(opacities)
        if sum_alpha < 1e-8:
            weights = torch.ones_like(opacities) / len(opacities)
        else:
            weights = opacities / sum_alpha

        parent_mean = torch.sum(weights.unsqueeze(-1) * means, dim=0)

        # Covariances Merge
        rotmats = custom_quat_to_rotmat(quats)
        S_sq = torch.diag_embed(scales**2)
        covars = torch.matmul(rotmats, torch.matmul(S_sq, rotmats.transpose(-1, -2)))

        mean_diff = means - parent_mean
        covar_outer = torch.matmul(mean_diff.unsqueeze(-1), mean_diff.unsqueeze(-2))
        combined_covars = covars + covar_outer
        parent_covar = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * combined_covars, dim=0)

        # Re-extract optimized components from merged Covariance
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(parent_covar)
            eigenvalues = torch.clamp(eigenvalues, min=1e-8)
            parent_scale = torch.sqrt(eigenvalues)

            R_p = eigenvectors
            if torch.linalg.det(R_p) < 0:
                R_p = R_p.clone()
                R_p[:, 0] = -R_p[:, 0]

            parent_quat = rotmat_to_quat(R_p)
        except RuntimeError:
            parent_scale = torch.sum(weights.unsqueeze(-1) * scales, dim=0)
            parent_quat = quats[torch.argmax(opacities)]

        parent_sh0 = torch.sum(weights.unsqueeze(-1) * sh0, dim=0)
        parent_shN = None
        if shN is not None:
            parent_shN = torch.zeros((shN.shape[1], 3), device=shN.device, dtype=shN.dtype)

        # Volumetric Opacity Blending
        vol_sigmas = (4.0 / 3.0) * math.pi * torch.prod(scales, dim=-1)
        vol_aabb = node.aabb_volume + 1e-7
        vol_ratio_sum = torch.sum(vol_sigmas / vol_aabb)
        combined_alpha_transmission = 1.0 - torch.prod(1.0 - opacities)
        parent_opacity = torch.clamp(
            self.gamma * vol_ratio_sum * combined_alpha_transmission,
            max=self.alpha_max
        )

        # 5. FORCE DE-ACTIVATION: Convert back to optimization spaces before renderer ingest
        parent_scale_raw = torch.log(parent_scale + 1e-8)
        
        clamped_opacity = torch.clamp(parent_opacity, 1e-6, 1.0 - 1e-6)
        parent_opacity_raw = torch.log(clamped_opacity / (1.0 - clamped_opacity))

        node.merged_splat_data = {
            "means": parent_mean.unsqueeze(0),
            "scales": parent_scale_raw.unsqueeze(0),
            "quats": parent_quat.unsqueeze(0),
            "opacities": parent_opacity_raw.unsqueeze(0),
            "sh0": parent_sh0.unsqueeze(0).unsqueeze(0),
            "shN": parent_shN.unsqueeze(0) if parent_shN is not None else None
        }

    def get_splats_at_level(self, level: int) -> Dict[str, torch.Tensor]:
        """Collect all representative merged Gaussians at a given depth level.
        If a node is a leaf and lies above 'level', we collect its representation as well
        to preserve full coverage of the scene bounds.
        """
        means_list = []
        scales_list = []
        quats_list = []
        opacities_list = []
        sh0_list = []
        shN_list = []

        def collect(node: OctreeNode):
            if node.merged_splat_data is None:
                return

            # Collect when:
            # 1. We reached the target level
            # 2. Or the node is a leaf (cannot go deeper)
            if node.depth == level or node.is_leaf:
                data = node.merged_splat_data
                means_list.append(data["means"])
                scales_list.append(data["scales"])
                quats_list.append(data["quats"])
                opacities_list.append(data["opacities"])
                sh0_list.append(data["sh0"])
                if data["shN"] is not None:
                    shN_list.append(data["shN"])
            else:
                for child in node.children:
                    collect(child)

        collect(self.root)

        # Concatenate standard attributes
        res = {
            "means": torch.cat(means_list, dim=0),
            "scales": torch.cat(scales_list, dim=0),
            "quats": torch.cat(quats_list, dim=0),
            "opacities": torch.cat(opacities_list, dim=0),
            "sh0": torch.cat(sh0_list, dim=0)
        }

        # Normalize and concatenate SH coefficients
        if len(shN_list) > 0:
            # 1. Identify maximum SH coefficient dimension (K)
            max_shN_dim = max(s.shape[1] for s in shN_list)
            
            # 2. Normalize list elements
            normalized_shN_list = []
            for s in shN_list:
                if s.shape[1] < max_shN_dim:
                    # Pad missing coefficients with zeros
                    pad_size = max_shN_dim - s.shape[1]
                    padding = torch.zeros((1, pad_size, 3), device=s.device, dtype=s.dtype)
                    normalized_shN_list.append(torch.cat([s, padding], dim=1))
                else:
                    normalized_shN_list.append(s)
            
            res["shN"] = torch.cat(normalized_shN_list, dim=0)
        else:
            # Fallback for splats without SH coefficients
            N = len(res["means"])
            # Assuming 24 is your standard max coefficient count for SH degree 3
            res["shN"] = torch.zeros((N, 24, 3), dtype=res["means"].dtype, device=res["means"].device)

        return res


# Public alias for backward compatibility and cleaner API naming
GaussianLODBuilder = LODOctree


def main():
    parser = argparse.ArgumentParser(description="LOD Spatial Hierarchy Octree partition and bottom-up parameter merging utility.")
    parser.add_argument("--input_ply", type=str, required=True, help="Path to the trained high-fidelity .ply point cloud.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the multi-resolution LOD levels.")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum depth of the Octree.")
    parser.add_argument("--min_gaussians", type=int, default=8, help="Minimum Gaussians inside a node before splitting.")
    parser.add_argument("--tau", type=float, default=1e-4, help="Scale parameter factor for visual importance score.")
    parser.add_argument("--epsilon_prune", type=float, default=1e-3, help="Pruning threshold for visual importance score.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Volumetric scale factor for merged opacity calculations.")
    parser.add_argument("--alpha_max", type=float, default=0.99, help="Maximum allowed opacity for merged Gaussians.")
    args = parser.parse_args()

    if load_ply_to_splats is None or export_splats is None:
        print("Error: Could not import 'gsplat.exporter' components. Make sure gsplat is properly installed.")
        sys.exit(1)

    print(f"Loading trained Gaussian splats from: {args.input_ply}")
    if not os.path.exists(args.input_ply):
        print(f"Error: Point cloud file not found: {args.input_ply}")
        sys.exit(1)

    splats = load_ply_to_splats(args.input_ply)
    print(f"Loaded {len(splats['means'])} Gaussians. Initiating Octree partitioning...")

    # Load on GPU if available for maximum speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing on device: {device}")

    # Transfer parameters to target device
    device_splats = {k: v.to(device) for k, v in splats.items()}

    # Initialize LODOctree builder
    lod_builder = LODOctree(
        points=device_splats["means"],
        scales=device_splats["scales"],
        quats=device_splats["quats"],
        opacities=device_splats["opacities"],
        sh0=device_splats["sh0"],
        shN=device_splats["shN"],
        max_depth=args.max_depth,
        min_gaussians=args.min_gaussians,
        tau=args.tau,
        epsilon_prune=args.epsilon_prune,
        gamma=args.gamma,
        alpha_max=args.alpha_max,
        raw_inputs=True
    )

    # Build the hierarchy
    lod_builder.build()

    # Save each LOD level to a separate PLY file
    os.makedirs(args.output_dir, exist_ok=True)
    for depth in range(args.max_depth + 1):
        print(f"Exporting Level of Detail (LOD) level {depth}...")
        lod_splats = lod_builder.get_splats_at_level(depth)

        # Move back to CPU for standard saving
        lod_splats_cpu = {k: v.cpu() for k, v in lod_splats.items()}

        out_path = os.path.join(args.output_dir, f"point_cloud_lod_{depth}.ply")
        export_splats(
            means=lod_splats_cpu["means"],
            scales=lod_splats_cpu["scales"],
            quats=lod_splats_cpu["quats"],
            opacities=lod_splats_cpu["opacities"],
            sh0=lod_splats_cpu["sh0"],
            shN=lod_splats_cpu["shN"],
            format="ply",
            save_to=out_path
        )
        print(f"  Level {depth} saved with {len(lod_splats_cpu['means'])} Gaussians to {out_path}")

    # Export a serialized PyTorch checkpoint mapping for easy custom rendering integrations
    checkpoint_path = os.path.join(args.output_dir, "lod_octree_hierarchy.pt")
    print(f"Saving serialized PyTorch Octree model map to: {checkpoint_path}")

    # Helper to recursively serialize tree
    def serialize_node(node: OctreeNode) -> Dict:
        serialized = {
            "aabb": (node.aabb[0].cpu(), node.aabb[1].cpu()),
            "depth": node.depth,
            "original_splat_indices": node.original_splat_indices.cpu(),
            "merged_splat_data": {k: v.cpu() for k, v in node.merged_splat_data.items()} if node.merged_splat_data is not None else None,
            "children": [serialize_node(c) for c in node.children]
        }
        return serialized

    serialized_tree = serialize_node(lod_builder.root)
    torch.save(serialized_tree, checkpoint_path)
    print("Export pipeline executed successfully. Done!")


if __name__ == "__main__":
    main()