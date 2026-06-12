"""
vfc_culling.py  –  Layer 2: View Frustum Culling (VFC)
======================================================
Implements a pure-PyTorch, vectorized frustum culling intercept layer for the
3DGS optimisation pipeline.  No CUDA compilation is required; all operations run
on ordinary PyTorch tensor primitives and benefit from auto-dispatched kernel
fusion on both CPU and CUDA devices.

Public API
----------
extract_frustum_planes(view_projection_matrix) -> planes  (6, 4)
cull_octree_nodes(aabb_min, aabb_max, planes)  -> bool mask (N,)

Mathematical Background
-----------------------
Gribb-Hartmann (2001) shows that each of the six frustum half-spaces can be
read off directly from the rows of the combined View-Projection (VP) matrix M:

    Left   : row3 + row0
    Right  : row3 - row0
    Bottom : row3 + row1
    Top    : row3 - row1
    Near   : row3 + row2   (OpenGL / NDC: [-1,1] range on Z)
    Far    : row3 - row2

Each raw plane vector is (A, B, C, D) such that Ax + By + Cz + D = 0.
After L2-normalising (A, B, C) the scalar D becomes the signed distance from
the origin to the plane, making all six distances geometrically comparable.

AABB vs Plane Culling (p-vertex test)
--------------------------------------
For a plane with outward-pointing normal n = (A, B, C), the AABB vertex
furthest along n (the "positive vertex" p) is:

    p_i = aabb_max_i  if  n_i >= 0  else  aabb_min_i

If  dot(n, p) + D < 0  the entire box is on the negative (outside) side of
that plane.  Performing this test across all 6 planes and logical-ANDing the
results gives a conservative visibility mask: a node marked False is entirely
outside the frustum and both it and its children can be discarded.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Plane extraction
# ---------------------------------------------------------------------------

def extract_frustum_planes(view_projection_matrix: torch.Tensor) -> torch.Tensor:
    """Extract and normalise the 6 frustum clip planes via Gribb-Hartmann.

    Parameters
    ----------
    view_projection_matrix : torch.Tensor, shape (4, 4)
        Column-major 4×4 view-projection matrix (i.e. the matrix M such that
        clip_pos = M @ world_pos for homogeneous world coordinates stored as
        column vectors).  The matrix must be on the same device as any tensors
        passed to :func:`cull_octree_nodes`.

    Returns
    -------
    planes : torch.Tensor, shape (6, 4)
        Row layout: [Left, Right, Bottom, Top, Near, Far].
        Each row is (A, B, C, D) in normalised form where (A, B, C) is the
        unit outward-pointing normal and D encodes the plane offset so that
        the half-space "inside" the frustum satisfies  Ax + By + Cz + D >= 0.
    """
    if view_projection_matrix.shape != (4, 4):
        raise ValueError(
            f"view_projection_matrix must be shape (4, 4), got {view_projection_matrix.shape}"
        )

    M = view_projection_matrix.float()

    # Gribb-Hartmann row combinations.
    # Row indices are 0-based; the matrix is stored in row-major PyTorch layout.
    row0 = M[0]  # (4,)
    row1 = M[1]
    row2 = M[2]
    row3 = M[3]

    raw_planes = torch.stack([
        row3 + row0,   # Left
        row3 - row0,   # Right
        row3 + row1,   # Bottom
        row3 - row1,   # Top
        row3 + row2,   # Near
        row3 - row2,   # Far
    ], dim=0)  # (6, 4)

    # Normalise each plane by the magnitude of its normal (A, B, C).
    # This turns D into an actual signed distance from the world origin.
    normals = raw_planes[:, :3]                             # (6, 3)
    norm_mag = torch.linalg.norm(normals, dim=1, keepdim=True)  # (6, 1)
    # Guard against degenerate planes (should never occur with a valid VP matrix)
    norm_mag = torch.clamp(norm_mag, min=1e-8)
    planes = raw_planes / norm_mag                          # (6, 4)  — broadcast over columns

    return planes


# ---------------------------------------------------------------------------
# Vectorised AABB culling
# ---------------------------------------------------------------------------

def cull_octree_nodes(
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    planes: torch.Tensor,
) -> torch.Tensor:
    """Vectorised frustum culling of N Octree AABB nodes against 6 clip planes.

    All operations are pure PyTorch tensor arithmetic; no Python loops over N.

    Parameters
    ----------
    aabb_min : torch.Tensor, shape (N, 3)
        World-space minimum corners of the N axis-aligned bounding boxes.
    aabb_max : torch.Tensor, shape (N, 3)
        World-space maximum corners of the N axis-aligned bounding boxes.
    planes : torch.Tensor, shape (6, 4)
        Normalised frustum planes as returned by :func:`extract_frustum_planes`.
        Each row is (A, B, C, D) with the outward-pointing convention so that
        inside points satisfy  Ax + By + Cz + D >= 0.

    Returns
    -------
    visible_mask : torch.Tensor, shape (N,), dtype=torch.bool
        ``True``  → node intersects or is fully inside the frustum (keep).
        ``False`` → node is completely outside at least one half-space (cull).

    Shape contract
    --------------
    Broadcasting layout used internally::

        normals    : (1, 6, 3)   planes[:, :3] expanded
        d_offsets  : (1, 6)     planes[:, 3]  expanded
        aabb_min   : (N, 1, 3)
        aabb_max   : (N, 1, 3)
        p_vertex   : (N, 6, 3)  — the "positive vertex" per plane
        signed_dist: (N, 6)     — dot(normal, p_vertex) + D
    """
    if aabb_min.shape != aabb_max.shape:
        raise ValueError(
            f"aabb_min and aabb_max must have the same shape, "
            f"got {aabb_min.shape} vs {aabb_max.shape}"
        )
    if aabb_min.dim() != 2 or aabb_min.shape[1] != 3:
        raise ValueError(
            f"aabb_min / aabb_max must be shape (N, 3), got {aabb_min.shape}"
        )
    if planes.shape != (6, 4):
        raise ValueError(
            f"planes must be shape (6, 4), got {planes.shape}"
        )

    # Ensure type & device consistency
    device = planes.device
    dtype  = planes.dtype
    aabb_min = aabb_min.to(device=device, dtype=dtype)
    aabb_max = aabb_max.to(device=device, dtype=dtype)

    # --- Expand for broadcasting ---
    # normals  : (1, 6, 3)
    normals   = planes[:, :3].unsqueeze(0)     # (1, 6, 3)
    # d_offsets: (1, 6)
    d_offsets = planes[:, 3].unsqueeze(0)      # (1, 6)

    # aabb corners expanded: (N, 1, 3)
    lo = aabb_min.unsqueeze(1)   # (N, 1, 3)
    hi = aabb_max.unsqueeze(1)   # (N, 1, 3)

    # --- p-vertex selection ---
    # For each plane's normal component n_i:
    #   if n_i >= 0 → choose aabb_max (furthest in +normal direction)
    #   else        → choose aabb_min
    # Shape: (N, 6, 3)
    p_vertex = torch.where(normals >= 0.0, hi, lo)   # broadcasts (N, 6, 3)

    # --- Signed distance of p-vertex to each plane ---
    # dot(normal, p_vertex) + D, shape: (N, 6)
    signed_dist = (normals * p_vertex).sum(dim=-1) + d_offsets  # (N, 6)

    # A node is outside iff its p-vertex is on the negative side of ANY plane.
    # i.e. signed_dist < 0 for that plane  →  entire box is outside.
    # We keep the node only if it passes ALL 6 tests.
    visible_mask = (signed_dist >= 0.0).all(dim=-1)   # (N,)

    return visible_mask
