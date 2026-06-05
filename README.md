# 3D Gaussian Splatting Level of Detail (LOD) Spatial Octree

This repository provides an optimized real-time rendering pipeline structure for 3D Gaussian Splatting (3DGS) featuring an active spatial hierarchy (LOD Octree). The builder reads a trained high-fidelity point cloud and generates a multi-resolution tree structure to balance performance and quality.

## Setup and Installation

Run the CLI utility using the custom conda or virtual environment (e.g. `gsplat_env`):

```bash
# Activate your environment
# then execute the pipeline:
python lod_octree.py --input_ply <path_to_ply> --output_dir <output_directory>
```

---

## Command Line Arguments

The `lod_octree.py` script accepts the following CLI arguments to customize the partition, parameter-merging, and pruning behavior:

| CLI Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input_ply` | `str` | *Required* | Path to the trained high-fidelity `.ply` point cloud file. |
| `--output_dir` | `str` | *Required* | Directory to save the multi-resolution LOD levels (PLY files and serialized `.pt` tree structure). |
| `--max_depth` | `int` | `4` | Maximum depth of the Octree spatial hierarchy. |
| `--min_gaussians` | `int` | `8` | Minimum number of Gaussians inside an Octree node before splitting it into child nodes. |
| `--tau` | `float` | `1e-4` | Scale parameter factor used in computing the visual importance score of a Gaussian. |
| `--epsilon_prune` | `float` | `1e-3` | Pruning threshold for visual importance score. Nodes with scores below this are pruned. |
| `--gamma` | `float` | `1.0` | Volumetric scale factor for merged parent opacity calculations. |
| `--alpha_max` | `float` | `0.99` | Maximum allowed opacity (clamping threshold) for merged parent Gaussians to prevent numerical singularities. |
| `--beta` | `float` | `0.5` | Sub-linear count exponent for Beer-Lambert density (typically `0.3` to `0.7`). Controls how opacity scales with child count. |
| `--kappa` | `float` | `None` | SH AC damping roll-off rate. Defaults to `max_depth` if not specified. Larger values preserve more high-frequency specular highlights. |

---

## Output Files

The utility exports the following outputs to the specified `--output_dir`:
1. **Multi-resolution PLY files**: `point_cloud_lod_{depth}.ply` containing Gaussians at each Level of Detail (LOD) level.
2. **Serialized PyTorch Octree mapping**: `lod_octree_hierarchy.pt` which represents the serialized tree architecture containing the hierarchical structure, bounding boxes, and merged splat parameters for custom renderer integration.
