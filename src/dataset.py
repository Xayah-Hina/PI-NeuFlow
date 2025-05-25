from .cuda_extensions import raymarching
from scipy.optimize import minimize
import torch
import torchvision.io as io
import numpy as np
import tqdm
import yaml
import os
import random
import typing
import types


class PINeuFlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 dataset_type: typing.Literal['train', 'val', 'test'],
                 downscale: int,
                 use_preload: bool,
                 use_fp16: bool,
                 device: torch.device,
                 ):
        # self.images
        self.images = PINeuFlowDataset._load_images(dataset_path, dataset_type, downscale, device, use_preload, use_fp16)  # [T, V, H, W, C]

        # self.poses
        self.poses, self.focals, self.widths, self.heights, self.extra_params = PINeuFlowDataset._load_camera_calibrations(dataset_path, dataset_type, downscale, device, use_preload, use_fp16)

        # self.times
        self.times = torch.linspace(0, 1, steps=self.images.shape[0], dtype=torch.float16 if use_fp16 else torch.float32).view(-1, 1)
        if use_preload:
            self.times = self.times.to(device=device)

        # self.states
        self.states = types.SimpleNamespace()
        self.states.dataset_type = dataset_type
        self.states.use_fp16 = use_fp16
        self.states.T = self.images.shape[0]
        self.states.V = self.images.shape[1]
        self.states.H = self.images.shape[2]
        self.states.W = self.images.shape[3]
        self.states.C = self.images.shape[4]

        # visualize
        # from .visualizer import visualize_poses_opengl_style
        # visualize_poses_opengl_style(self.poses.detach().cpu(), size=0.1, func=None)

    def __getitem__(self, index):
        if self.states.dataset_type == 'train':
            time_shift = random.uniform(-0.5, 0.5)
        else:
            time_shift = 0
        video_index = random.randint(0, self.poses.shape[0] - 1)

        if index == 0 and time_shift <= 0:
            target_image = self.images[index, video_index]
            target_time = self.times[index]
        elif index == self.images.shape[0] - 1 and time_shift >= 0:
            target_image = self.images[index, video_index]
            target_time = self.times[index]
        else:
            if time_shift >= 0:
                target_image = (1 - time_shift) * self.images[index, video_index] + time_shift * self.images[index + 1, video_index]
                target_time = (1 - time_shift) * self.times[index] + time_shift * self.times[index + 1]
            else:
                target_image = (1 + time_shift) * self.images[index, video_index] + (-time_shift) * self.images[index - 1, video_index]
                target_time = (1 + time_shift) * self.times[index] + (-time_shift) * self.times[index - 1]

        return {
            'image': target_image,
            'pose': self.poses[video_index],
            'time': target_time,
            'video_index': video_index,
            'focal': self.focals[video_index],
        }

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_images(dataset_path, dataset_type, downscale, device: torch.device, use_preload: bool, use_fp16: bool):
        """
        Load images from the dataset path and return them as a tensor.
        Args:
            dataset_path (str): Path to the dataset.
            dataset_type (str): Type of the dataset ('train', 'val', 'test').
            downscale (int): Downscale factor for the images.
        Returns:
            torch.Tensor: Tensor containing the loaded images. [T, V, H, W, C]
        """
        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)
            videos_info = scene_info['training_videos'] if dataset_type == 'train' else scene_info['validation_videos']
            frames = []
            for path in tqdm.tqdm([os.path.normpath(os.path.join(dataset_path, video_path)) for video_path in videos_info], desc=f'[Loading Images ({dataset_type})...]'):
                try:
                    v_frames, a_frames, _info = io.read_video(path, pts_unit='sec')
                    frames.append(v_frames / 255.0)
                except Exception as e:
                    raise FileNotFoundError(f'Could not load video {path}: {e}')
            frames = torch.stack(frames)
            V, T, H, W, C = frames.shape
            H_downscale, W_downscale = int(H // downscale), int(W // downscale)
            frames_downscale = torch.nn.functional.interpolate(frames.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W), size=(H_downscale, W_downscale), mode='bilinear', align_corners=False).reshape(V, T, C, H_downscale, W_downscale).permute(1, 0, 3, 4, 2)

            if use_preload:
                frames_downscale = frames_downscale.to(torch.float16 if use_fp16 else torch.float32).to(device=device)

            return frames_downscale

    @staticmethod
    def _load_camera_calibrations(dataset_path, dataset_type, downscale, device: torch.device, use_preload: bool, use_fp16: bool):
        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)
            cameras_info = scene_info['training_camera_calibrations'] if dataset_type == 'train' else scene_info['validation_camera_calibrations']
            poses = []
            focals = []
            widths = []
            heights = []
            nears = []
            fars = []
            for path in tqdm.tqdm([os.path.normpath(os.path.join(dataset_path, camera_path)) for camera_path in cameras_info], desc=f'[Loading Camera ({dataset_type})...]'):
                try:
                    camera_info = np.load(path)
                    poses.append(torch.tensor(camera_info["cam_transform"]))
                    focals.append(float(camera_info["focal"]) * float(camera_info["width"]) / float(camera_info["aperture"]))
                    widths.append(int(camera_info["width"]))
                    heights.append(int(camera_info["height"]))
                    nears.append(float(camera_info["near"]))
                    fars.append(float(camera_info["far"]))
                except Exception as e:
                    raise FileNotFoundError(f'Could not load camera {path}: {e}')
            assert len(set(widths)) == 1, f"Error: Inconsistent widths found: {widths}. All cameras must have the same resolution."
            assert len(set(heights)) == 1, f"Error: Inconsistent heights found: {heights}. All cameras must have the same resolution."
            poses = torch.stack(poses)
            focals = torch.tensor(focals)
            widths = set(widths).pop()
            heights = set(heights).pop()
            nears = torch.tensor(nears)
            fars = torch.tensor(fars)

            focals = focals / downscale
            widths = widths // downscale
            heights = heights // downscale

            voxel_transform = torch.tensor(scene_info['voxel_transform'])
            new_poses, new_voxel_transform = PINeuFlowDataset._adjust_poses(poses.detach().cpu().numpy(), voxel_transform.detach().cpu().numpy())
            poses = torch.tensor(new_poses)
            voxel_transform = torch.tensor(new_voxel_transform)

            voxel_scale = torch.tensor(scene_info['voxel_scale'])
            s_min = torch.tensor(scene_info['s_min'])
            s_max = torch.tensor(scene_info['s_max'])
            s_w2s = torch.inverse(voxel_transform).expand([4, 4])
            s2w = torch.inverse(s_w2s)
            s_scale = voxel_scale.expand([3])

            if use_preload:
                poses = poses.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                focals = focals.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                nears = nears.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                fars = fars.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                voxel_transform = voxel_transform.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                voxel_scale = voxel_scale.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                s_min = s_min.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                s_max = s_max.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                s_w2s = s_w2s.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                s2w = s2w.to(torch.float16 if use_fp16 else torch.float32).to(device=device)
                s_scale = s_scale.to(torch.float16 if use_fp16 else torch.float32).to(device=device)

            extra_params = types.SimpleNamespace()
            extra_params.nears = nears
            extra_params.fars = fars
            extra_params.voxel_transform = voxel_transform
            extra_params.voxel_scale = voxel_scale
            extra_params.s_min = s_min
            extra_params.s_max = s_max
            extra_params.s_w2s = s_w2s
            extra_params.s2w = s2w
            extra_params.s_scale = s_scale
            return poses, focals, widths, heights, extra_params

    @staticmethod
    def _adjust_poses(poses, others):
        origins = []
        directions = []
        size = 0.1
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
            origins.append(o)
            directions.append(dir)
        origins = np.array(origins)
        directions = np.array(directions)
        c_opt, radius, pts = PINeuFlowDataset.find_min_enclosing_sphere_on_rays(origins, directions)

        for pose in poses:
            pose[:3, 3] -= c_opt
        others[:3, 3] -= c_opt
        return poses, others

    @staticmethod
    def find_min_enclosing_sphere_on_rays(origins, directions):
        """
        origins: [N, 3]
        directions: [N, 3]
        """
        N = origins.shape[0]

        # 初始化：t 全为 0（即只考虑相机原点）
        t_init = np.zeros(N)
        center_init = origins.mean(axis=0)
        x0 = np.concatenate([center_init, t_init])

        def objective(x):
            c = x[:3]
            t = x[3:]
            pts = origins + directions * t[:, None]
            dists = np.linalg.norm(pts - c[None, :], axis=1)
            return np.max(dists)

        res = minimize(objective, x0, method='L-BFGS-B')
        x_opt = res.x
        c_opt = x_opt[:3]
        t_opt = x_opt[3:]
        pts = origins + directions * t_opt[:, None]
        radius = np.max(np.linalg.norm(pts - c_opt[None, :], axis=1))
        return c_opt, radius, pts


class FrustumsSampler:
    def __init__(self, dataset: PINeuFlowDataset, num_rays, randomize: bool):
        dtype = dataset.images.dtype
        device = dataset.images.device

        # self.dataset
        self.dataset = dataset

        # self.rays_per_iter
        self.num_rays = num_rays if dataset.states.dataset_type == 'train' else -1

        # self.randomize
        self.randomize = randomize

        # occupancy grid
        self.cascade = 1
        self.grid_size = 128  # don't change this, it is hardcoded in the CUDA kernel
        self.time_size = dataset.states.T
        self.density_grid = torch.zeros(self.time_size, self.cascade, self.grid_size ** 3, dtype=dtype, device=device)  # [T, CAS, H * H * H]
        self.density_bitfield = torch.zeros(self.time_size, self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8, device=device)  # [T, CAS * H * H * H // 8]
        self.mean_density = 0
        self.iter_density = 0
        self.step_counter = torch.zeros(16, 2, dtype=torch.int32, device=device)  # 16 is hardcoded for averaging...
        self.mean_count = 0
        self.local_step = 0
        self.bound = 1.0
        self.density_scale = 1
        self.density_thresh = 0.01

        # temp
        self.aabb_train = torch.tensor([-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound], dtype=dtype, device=device)  # [xmin, ymin, zmin, xmax, ymax, zmax]
        self.min_near = 0.1

    @torch.no_grad()
    def mark_untrained_grid(self, poses, fx, fy, cx, cy, S=64):
        # poses: [B, 4, 4]

        B = poses.shape[0]

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid[0])
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    world_xyzs = (2 * coords.to(poses.dtype) / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3]  # [S, N, 3]

                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] < 0  # [S, N] NOTE: OPENGL STYLE, so z is negative in front of the camera
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * (-cam_xyzs[:, :, 2]) + half_grid_size * 2  # NOTE: OPENGL STYLE, so z is negative in front of the camera
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * (-cam_xyzs[:, :, 2]) + half_grid_size * 2  # NOTE: OPENGL STYLE, so z is negative in front of the camera
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)  # [N]

                            # update count
                            count[cas, indices] += mask
                            head += S

        # mark untrained grid as -1
        self.density_grid[count.unsqueeze(0).expand_as(self.density_grid) == 0] = -1

        print(f"cound: max - {count.max()}, min - {count.min()} 0. {(count == 0).sum()}, 1. {(count == 1).sum()}, 2. {(count == 2).sum()}, 3. {(count == 3).sum()}, 4. {(count == 4).sum()}, 5. {(count == 5).sum()}, 6. {(count == 6).sum()}, 7. {(count == 7).sum()}, 8. {(count == 8).sum()}")
        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

        # from .visualizer import visualize_density_grid
        # visualize_density_grid(self.density_grid[0, 0], grid_size=self.grid_size, poses=self.dataset.poses.detach().cpu())

    @torch.no_grad()
    def update_extra_state(self, network, decay=0.95, S=128):
        # call before each epoch to update extra states.

        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)

        # full update.
        if self.iter_density < 16:
            # if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            # for t, time in enumerate(self.dataset.times):
            for t, time in enumerate(tqdm.tqdm(self.dataset.times, desc="update extra state times...")):
                time: torch.Tensor
                for xs in X:
                    for ys in Y:
                        for zs in Z:

                            # construct points
                            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long()  # [N]
                            xyzs = 2 * coords.to(time.dtype) / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                half_time_size = 0.5 / self.time_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in coord [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # add noise in time [-hts, hts]
                                time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                                # query density
                                cas_xyzts = torch.cat([cas_xyzs, time_perturb[None].expand(cas_xyzs.shape[0], 1)], dim=-1)  # [N, 4]
                                sigmas = network.sigma(cas_xyzts)['sigma'].reshape(-1).detach()
                                sigmas *= self.density_scale
                                # assign
                                tmp_grid[t, cas, indices] = sigmas

        # partial update (half the computation)
        # just update 100 times should be enough... too time consuming.
        elif self.iter_density < 100:
            N = self.grid_size ** 3 // 4  # T * C * H * H * H / 4
            for t, time in enumerate(self.dataset.times):
                for cas in range(self.cascade):
                    # random sample some positions
                    coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[t, cas] > 0).squeeze(-1)  # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                    occ_indices = occ_indices[rand_mask]  # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.to(time.dtype)/ (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    half_time_size = 0.5 / self.time_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # add noise in time [-hts, hts]
                    time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                    cas_xyzts = torch.cat([cas_xyzs, time_perturb[None].expand(cas_xyzs.shape[0], 1)], dim=-1)  # [N, 4]
                    # query density
                    sigmas = network.sigma(cas_xyzts)['sigma'].reshape(-1).detach()
                    sigmas *= self.density_scale
                    # assign
                    tmp_grid[t, cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()  # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        for t in range(self.time_size):
            raymarching.packbits(self.density_grid[t], density_thresh, self.density_bitfield[t])

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

    def compute_nears_fars(self, rays_o, rays_d):
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train, self.min_near)
        return nears, fars

    def collate(self, batch: list):
        images = torch.stack([single['image'] for single in batch])  # [B, H, W, C]
        poses = torch.stack([single['pose'] for single in batch])  # [B, 4, 4]
        focals = torch.stack([single['focal'] for single in batch])  # [B]
        times = torch.stack([single['time'] for single in batch])  # [B, 1]
        rays_o, rays_d, pixels = FrustumsSampler._sample_rays_pixels(images=images, poses=poses, focals=focals, width=self.dataset.widths, height=self.dataset.heights, num_rays=self.num_rays, randomize=self.randomize, device=images.device)  # [B, N, 3]

        return {
            'pixels': pixels,  # [B, N, C]
            'times': times,  # [B, 1]
            'rays_o': rays_o,  # [B, N, 3]
            'rays_d': rays_d,  # [B, N, 3]
        }

    def dataloader(self, batch_size):
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate,
            shuffle=self.dataset.states.dataset_type == 'train',
            num_workers=0,
        )

    @staticmethod
    def _sample_rays_pixels(
            images: torch.Tensor,  # [N, H, W, 3]
            poses: torch.Tensor,  # [N, 4, 4]
            focals: torch.Tensor,  # [N]
            width: int,
            height: int,
            num_rays: int,
            randomize: bool,
            device: torch.device,
    ):
        """
        Sample UV positions and directions, and interpolate corresponding pixel values.

        Returns:
        - dirs_normalized: [N, num_rays, 3] or [N, H, W, 3]
        - sampled_rgb: [N, num_rays, 3] or [N, H, W, 3]
        """
        N = focals.shape[0]

        if num_rays == -1:
            u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=device), torch.linspace(0, height - 1, height, device=device), indexing='xy')  # (H, W), (H, W)
            u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)
            dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)
            dirs_normalized = torch.nn.functional.normalize(dirs, dim=-1)  # (N, H, W, 3)
            rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs_normalized.to(poses.dtype))
            rays_o = poses[:, None, None, :3, 3].expand_as(rays_d)
            rays_d = rays_d.reshape(N, -1, 3)
            rays_o = rays_o.reshape(N, -1, 3)
            sampled_rgb = images.reshape(N, -1, 3)
        else:
            # 1. Sample UV (pixel) coordinates
            u = torch.randint(0, width, (num_rays,), device=device, dtype=images.dtype)
            v = torch.randint(0, height, (num_rays,), device=device, dtype=images.dtype)

            if randomize:
                u = u + torch.rand_like(u)
                v = v + torch.rand_like(v)

            # 2. Compute directions
            u_normalized = (u[None, :] - width * 0.5) / focals[:, None]
            v_normalized = (v[None, :] - height * 0.5) / focals[:, None]
            dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)
            dirs_normalized = torch.nn.functional.normalize(dirs, dim=-1)  # [N, num_rays, 3]

            rays_d = torch.einsum('bij,bnj->bni', poses[:, :3, :3], dirs_normalized)  # (B, N, 3)
            rays_o = poses[:, None, :3, 3].expand_as(rays_d)  # (B, N, 3)

            # 3. Interpolate image pixel values at (u, v)
            # grid_sample expects coords in [-1, 1] normalized format
            grid_u = (u / (width - 1)) * 2 - 1  # [num_rays]
            grid_v = (v / (height - 1)) * 2 - 1  # [num_rays]
            grid = torch.stack([grid_u, grid_v], dim=-1)  # [num_rays, 2]
            grid = grid[None].expand(N, -1, -1)  # [N, num_rays, 2]
            grid = grid.unsqueeze(2)  # [N, num_rays, 1, 2] for grid_sample

            # reshape images for grid_sample: [N, C, H, W]
            images_ = images.permute(0, 3, 1, 2)  # [N, 3, H, W]
            sampled_rgb = torch.nn.functional.grid_sample(images_, grid, align_corners=True)  # [N, 3, num_rays, 1]
            sampled_rgb = sampled_rgb.squeeze(-1).permute(0, 2, 1)  # [N, num_rays, 3]

        return rays_o, rays_d, sampled_rgb  # [N, num_rays, 3]
