import numpy as np
import trimesh


def visualize_poses_opengl_style(poses, size=0.1, func=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    origins = []
    directions = []
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
        if func:
            origins.append(o)
            directions.append(dir)
    if func:
        origins = np.array(origins)
        directions = np.array(directions)
        c_opt, radius, pts = func(origins, directions)
        sphere = trimesh.creation.uv_sphere(radius=radius)
        sphere.apply_translation(c_opt)
        objects.append(sphere)

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


def visualize_density_grid(density_grid, grid_size, poses, size=0.1):
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if poses is not None:
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

    import torch
    from .cuda_extensions import raymarching

    device = density_grid.device

    # [H, H, H] 中所有坐标
    xs, ys, zs = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing='ij'
    )
    coords = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)  # [H³, 3]

    # 得到 morton 编码
    indices = raymarching.morton3D(coords).long()  # [H³]

    values = density_grid[indices]  # [H³]
    mask = values != -1  # -1 表示未激活的体素
    active_coords = coords[mask].cpu().numpy()  # [N, 3]

    normalized = (active_coords / (grid_size - 1)) * 2 - 1  # [N, 3] ∈ [-1, 1]

    cloud = trimesh.points.PointCloud(normalized)
    objects.append(cloud)
    trimesh.Scene(objects).show()


