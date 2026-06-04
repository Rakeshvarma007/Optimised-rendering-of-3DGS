import pytest
import torch
import math
import sys
import os

# Add parent directory to path to import lod_octree
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lod_octree import (
    rotmat_to_quat,
    custom_quat_to_rotmat,
    safe_logit,
    LODOctree,
    OctreeNode
)


def test_quaternion_roundtrip():
    """Verify that rotation matrix and quaternion translations are mathematically consistent and roundtrip correctly."""
    torch.manual_seed(42)

    # 1. Create a set of random quaternions
    N = 100
    quats = torch.randn(N, 4)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    # 2. Convert to rotation matrix
    R = custom_quat_to_rotmat(quats)
    assert R.shape == (N, 3, 3)

    # Verify that rotation matrices are orthogonal (R * R^T = I)
    identity = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
    RT = R.transpose(-1, -2)
    assert torch.allclose(torch.matmul(R, RT), identity, atol=1e-5)

    # 3. Convert back to quaternions
    quats_back = rotmat_to_quat(R)
    assert quats_back.shape == (N, 4)

    # 4. Check that they represent the same orientation (q and -q represent the same rotation)
    dot_products = torch.abs(torch.sum(quats * quats_back, dim=-1))
    assert torch.allclose(dot_products, torch.ones_like(dot_products), atol=1e-5)


def test_octree_spatial_partitioning():
    """Verify that the Octree correctly splits and assigns points to the appropriate child octants."""
    torch.manual_seed(42)

    # Create 64 points in a 3D grid to guarantee points fall in every octant
    coords = torch.linspace(-1.0, 1.0, 4)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    points = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=1)

    N = len(points)
    scales = torch.zeros(N, 3) # log scale 0 => scale 1
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(N, 1)
    opacities = torch.zeros(N) # raw logit 0 => opacity 0.5
    sh0 = torch.randn(N, 3)

    # Instantiate LODOctree with a depth of 2
    lod_builder = LODOctree(
        points=points,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        max_depth=2,
        min_gaussians=4,
        raw_inputs=True
    )

    # Split root
    lod_builder._split_node(lod_builder.root)

    # The root must have 8 children since points cover all octants
    assert len(lod_builder.root.children) == 8

    # Verify that child nodes track their AABBs correctly and point indices are disjoint
    all_indices = []
    for child in lod_builder.root.children:
        assert child.depth == 1
        assert child.parent == lod_builder.root
        indices = child.original_splat_indices
        assert len(indices) > 0
        all_indices.extend(indices.tolist())

        # Check AABB
        min_b, max_b = child.aabb
        child_points = lod_builder.points[indices]
        assert torch.all(child_points >= min_b - 1e-6)
        assert torch.all(child_points <= max_b + 1e-6)

    # Verify all points are assigned and no duplicates
    assert len(all_indices) == N
    assert len(set(all_indices)) == N


def test_parameter_merging_math():
    """Verify that bottom-up parameter merging logic matches the mathematical formulations exactly."""
    torch.manual_seed(42)

    # Create 3 points in space
    points = torch.tensor([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0]
    ])
    scales = torch.tensor([
        [0.0, 0.0, 0.0],  # scale = 1, Vol = 4/3 * pi
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    quats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])
    # raw logit opacities: logit(0.5) = 0.0
    opacities = torch.tensor([0.0, 0.0, 0.0]) # physical opacities = [0.5, 0.5, 0.5]
    sh0 = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0]
    ])

    lod_builder = LODOctree(
        points=points,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        max_depth=1,
        min_gaussians=1,
        raw_inputs=True,
        gamma=1.0,
        alpha_max=0.99
    )

    # Force direct execution of merge on the root node containing all 3 Gaussians
    lod_builder._merge_node(lod_builder.root)
    merged = lod_builder.root.merged_splat_data

    assert merged is not None
    assert merged["means"].shape == (1, 3)
    assert merged["scales"].shape == (1, 3)
    assert merged["quats"].shape == (1, 4)
    assert merged["opacities"].shape == (1,)
    assert merged["sh0"].shape == (1, 1, 3)

    # 1. Means weighted average: since opacities are equal, it should be the direct mean of coordinates
    expected_mean = torch.mean(points, dim=0)
    assert torch.allclose(merged["means"].squeeze(0), expected_mean, atol=1e-5)

    # 2. SH Weighted average: color components should be averaged directly
    expected_sh0 = torch.mean(sh0, dim=0).unsqueeze(0).unsqueeze(0)
    assert torch.allclose(merged["sh0"], expected_sh0, atol=1e-5)

    # 3. Volumetric transmission-based combined opacity
    # combined_trans = 1 - (1 - 0.5)^3 = 1 - 0.125 = 0.875
    # Vol(Sigma) = 4/3 * pi * 1 = 4.18879
    # AABB bounds enclosing points: x in [-1.01, 1.01], y in [-0.02, 2.02], z in [-0.01, 0.01]
    # Vol(AABB) is calculated and multiplied by gamma
    # Let's ensure physical opacity is between 0 and alpha_max
    opacity_val = torch.sigmoid(merged["opacities"])
    assert opacity_val.item() <= 0.99 + 1e-5
    assert opacity_val.item() > 0.0


def test_importance_based_pruning():
    """Verify that unimportant Gaussians are pruned and only high impact ones are retained."""
    torch.manual_seed(42)

    # Create 3 points
    points = torch.randn(3, 3)
    # Scales: 1st very large, 2nd medium, 3rd tiny
    scales = torch.tensor([
        [2.0, 2.0, 2.0],  # product scale = e^6 = 403.4
        [0.0, 0.0, 0.0],  # product scale = e^0 = 1.0
        [-5.0, -5.0, -5.0] # product scale = e^-15 = 3e-7
    ])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    # Opacities: 1st fully opaque, 2nd average, 3rd almost transparent
    # logits: 5.0 => 0.993, 0.0 => 0.5, -5.0 => 0.0067
    opacities = torch.tensor([5.0, 0.0, -5.0])
    sh0 = torch.randn(3, 3)

    # Importance scoring threshold epsilon_prune = 0.1
    # tau = 1.0
    # Expected visual impact scores:
    # 1. 0.993 * min(1.0, 403.4/1.0) = 0.993 (keep)
    # 2. 0.5 * min(1.0, 1.0/1.0) = 0.5 (keep)
    # 3. 0.0067 * min(1.0, 3e-7/1.0) = 2e-9 (prune!)
    lod_builder = LODOctree(
        points=points,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        max_depth=1,
        min_gaussians=4,
        tau=1.0,
        epsilon_prune=0.1,
        raw_inputs=True
    )

    # Force merge
    lod_builder._merge_node(lod_builder.root)
    # To check what was merged, let's examine the node merge logic manually:
    # We prune the tiny 3rd point, and only merge points 1 & 2!
    # Weighted mean of points 1 & 2:
    w1 = 0.9933 / (0.9933 + 0.5)
    w2 = 0.5 / (0.9933 + 0.5)
    expected_mean = w1 * points[0] + w2 * points[1]
    
    assert torch.allclose(lod_builder.root.merged_splat_data["means"].squeeze(0), expected_mean, atol=1e-4)


def test_pruning_fallback():
    """Verify that if all Gaussians fall below the pruning threshold, the node keeps at least the best Gaussian."""
    torch.manual_seed(42)

    # All points are extremely tiny and transparent
    points = torch.randn(2, 3)
    scales = torch.tensor([[-10.0, -10.0, -10.0], [-20.0, -20.0, -20.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    opacities = torch.tensor([-10.0, -10.0]) # tiny opacity
    sh0 = torch.randn(2, 3)

    # High prune threshold
    lod_builder = LODOctree(
        points=points,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        max_depth=1,
        min_gaussians=4,
        tau=1.0,
        epsilon_prune=0.5,
        raw_inputs=True
    )

    # Execute merge
    lod_builder._merge_node(lod_builder.root)

    # The node's merged data must not be empty or None, it should fallback and keep the 1st (better) one
    assert lod_builder.root.merged_splat_data is not None
    assert lod_builder.root.merged_splat_data["means"].shape == (1, 3)
