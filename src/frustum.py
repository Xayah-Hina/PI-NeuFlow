import torch


def assemble_points(batch_rays_o, batch_rays_d, num_samples: int, near: float, far: float, randomize: bool):
    """
    :param batch_rays_o: [N, 3]
    :param batch_rays_d: [N, 3]
    :param num_samples: int
    :param near: float
    :param far: float
    :param randomize: bool
    :return:
        points: [N, num_samples, 3]
        deltas: [N, num_samples]
        rays_d: [N, 3]
    """
    assert batch_rays_o.shape[0] == batch_rays_d.shape[0], "batch_rays_o and batch_rays_d must have the same number of rays"
    N = batch_rays_d.shape[0]
    device = batch_rays_d.device
    dtype = batch_rays_d.dtype

    t_vals = torch.linspace(0., 1., steps=num_samples, device=device, dtype=batch_rays_d.dtype).unsqueeze(0)  # [1, num_samples]
    z_vals = (near * (1. - t_vals) + far * t_vals).expand(N, num_samples)  # [N, num_samples]

    if randomize:
        mid_vals = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [N, num_samples-1]
        upper_vals = torch.cat([mid_vals, z_vals[..., -1:]], -1)  # [N, num_samples]
        lower_vals = torch.cat([z_vals[..., :1], mid_vals], -1)  # [N, num_samples]
        t_rand = torch.rand(z_vals.shape, device=device, dtype=dtype)  # [N, num_samples]
        z_vals = lower_vals + (upper_vals - lower_vals) * t_rand  # [N, num_samples]
    batch_dist_vals = z_vals[..., 1:] - z_vals[..., :-1]  # [N, num_samples-1]

    batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * z_vals[..., :, None]  # [N, num_samples, 3]

    return batch_points, batch_dist_vals, z_vals


def assemble_points_with_time(batch_rays_o, batch_rays_d, time, num_samples: int, near: float, far: float, randomize: bool):
    """
    :param batch_rays_o: [N, 3]
    :param batch_rays_d: [N, 3]
    :param time: [1]
    :param num_samples: int
    :param near: float
    :param far: float
    :param randomize: bool
    :return:
        points: [N, num_samples, 4]
        deltas: [N, num_samples]
        rays_d: [N, 3]
    """
    batch_points, batch_dist_vals, z_vals = assemble_points(batch_rays_o, batch_rays_d, num_samples, near, far, randomize)  # [N, num_samples, 3]
    batch_points_with_time = torch.cat([batch_points, time.expand(batch_points.shape[0], batch_points.shape[1], 1)], dim=-1)  # [N, num_samples, 4]
    return batch_points_with_time, batch_dist_vals, z_vals


def pos_world2smoke(inputs_pts, s_w2s, s_scale):
    pts_world_homo = torch.cat([inputs_pts, torch.ones_like(inputs_pts[..., :1])], dim=-1)
    pts_sim_ = torch.matmul(s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
    pts_sim = pts_sim_ / s_scale
    return pts_sim


def is_inside(inputs_pts, s_w2s, s_scale, s_min, s_max):
    target_pts = pos_world2smoke(inputs_pts, s_w2s, s_scale)
    above = torch.logical_and(target_pts[..., 0] >= s_min[0], target_pts[..., 1] >= s_min[1])
    above = torch.logical_and(above, target_pts[..., 2] >= s_min[2])
    below = torch.logical_and(target_pts[..., 0] <= s_max[0], target_pts[..., 1] <= s_max[1])
    below = torch.logical_and(below, target_pts[..., 2] <= s_max[2])
    outputs = torch.logical_and(below, above)
    return outputs


def inside_mask(inputs_pts, s_w2s, s_scale, s_min, s_max, to_float=False):
    mask = is_inside(inputs_pts, s_w2s, s_scale, s_min, s_max)
    return mask.to(torch.float) if to_float else mask
