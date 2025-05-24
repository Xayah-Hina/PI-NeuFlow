from .visualizer import visualize_poses
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

        # visualize
        # visualize_poses(self.poses.detach().cpu(), size=0.1, func=self.find_min_enclosing_sphere_on_rays)

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

            new_poses = PINeuFlowDataset._adjust_poses(poses.detach().cpu().numpy())
            poses = torch.tensor(new_poses).to(device=device)

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
        # self.dataset
        self.dataset = dataset

        # self.rays_per_iter
        self.num_rays = num_rays if dataset.states.dataset_type == 'train' else -1

        # self.randomize
        self.randomize = randomize

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
            rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs_normalized)
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
