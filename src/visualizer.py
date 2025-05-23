import numpy as np
import trimesh


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def visualize_rays(rays_o, rays_d, size=0.1):
    """
    Visualize rays using trimesh.
    Args:
        rays_o: [B, N, 3] numpy array, ray origins
        rays_d: [B, N, 3] numpy array, ray directions
        size: float, length of each ray segment to show
    """
    assert isinstance(rays_o, np.ndarray) and isinstance(rays_d, np.ndarray), "Input must be numpy arrays"
    assert rays_o.shape == rays_d.shape and rays_o.ndim == 3, "Shape must be [B, N, 3]"

    B, N, _ = rays_o.shape
    rays = []

    for b in range(B):
        origins = rays_o[b]  # [N, 3]
        directions = rays_d[b]  # [N, 3]
        endpoints = origins + size * directions

        # Stack line segments as [N, 2, 3]
        segments = np.stack([origins, endpoints], axis=1)
        rays.append(segments)

    # Concatenate all segments: [total_rays, 2, 3]
    rays = np.concatenate(rays, axis=0)

    # Convert to a Path3D
    ray_paths = trimesh.load_path(rays)

    # Create a box and axis for context
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))

    # Combine and visualize
    scene = trimesh.Scene([axes, box, ray_paths])
    scene.show()
