from .visualizer import visualize_poses
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
        self.poses, self.focals, self.widths, self.heights, self.extra_params = PINeuFlowDataset._load_camera_calibrations(dataset_path, dataset_type, downscale, device, use_preload)

        # self.times
        self.times = torch.linspace(0, 1, steps=self.images.shape[0], dtype=torch.float32).view(-1, 1)
        if use_preload:
            self.times = self.times.to(device=device)

        # self.states
        self.states = types.SimpleNamespace()
        self.states.dataset_type = dataset_type

        # visualize
        # visualize_poses(self.poses, size=0.1)

    def __getitem__(self, index):
        time_shift = random.uniform(-0.5, 0.5)
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
    def _load_camera_calibrations(dataset_path, dataset_type, downscale, device: torch.device, use_preload: bool):
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
                    poses.append(torch.tensor(camera_info["cam_transform"], dtype=torch.float32))
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
            focals = torch.tensor(focals, dtype=torch.float32)
            widths = set(widths).pop()
            heights = set(heights).pop()
            nears = torch.tensor(nears, dtype=torch.float32)
            fars = torch.tensor(fars, dtype=torch.float32)

            focals = focals / downscale
            widths = widths // downscale
            heights = heights // downscale

            voxel_transform = torch.tensor(scene_info['voxel_transform'], dtype=torch.float32)
            voxel_scale = torch.tensor(scene_info['voxel_scale'])
            s_min = torch.tensor(scene_info['s_min'])
            s_max = torch.tensor(scene_info['s_max'])
            s_w2s = torch.inverse(voxel_transform).expand([4, 4])
            s2w = torch.inverse(s_w2s)
            s_scale = voxel_scale.expand([3])

            if use_preload:
                poses = poses.to(device=device)
                focals = focals.to(device=device)
                nears = nears.to(device=device)
                fars = fars.to(device=device)
                voxel_transform = voxel_transform.to(device=device)
                voxel_scale = voxel_scale.to(device=device)
                s_min = s_min.to(device=device)
                s_max = s_max.to(device=device)
                s_w2s = s_w2s.to(device=device)
                s2w = s2w.to(device=device)
                s_scale = s_scale.to(device=device)

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


class FrustumsSampler:
    def __init__(self, dataset: PINeuFlowDataset, num_rays, randomize: bool = True):
        # self.dataset
        self.dataset = dataset

        # self.rays_per_iter
        self.num_rays = num_rays

        # self.randomize
        self.randomize = randomize

    def collate(self, batch: list):
        images = torch.stack([single['image'] for single in batch])  # [B, H, W, C]
        poses = torch.stack([single['pose'] for single in batch])  # [B, 4, 4]
        focals = torch.stack([single['focal'] for single in batch])  # [B]
        times = torch.stack([single['time'] for single in batch])  # [B, 1]
        dirs, pixels = FrustumsSampler._sample_uv_dirs_images(images=images, focals=focals, width=self.dataset.widths, height=self.dataset.heights, num_rays=self.num_rays, randomize=self.randomize, device=images.device)  # [B, N, 3]

        rays_d = torch.einsum('bij,bnj->bni', poses[:, :3, :3], dirs)  # (B, N, 3)
        rays_o = poses[:, None, :3, 3].expand_as(rays_d)  # (B, N, 3)

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
            shuffle=True,
            num_workers=0,
        )

    @staticmethod
    def _sample_uv_dirs_images(
            images: torch.Tensor,  # [N, H, W, 3]
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
        - dirs_normalized: [N, num_rays, 3]
        - sampled_rgb: [N, num_rays, 3]
        """
        N = focals.shape[0]

        # 1. Sample UV (pixel) coordinates
        u = torch.randint(0, width, (num_rays,), device=device, dtype=torch.float32)
        v = torch.randint(0, height, (num_rays,), device=device, dtype=torch.float32)

        if randomize:
            u = u + torch.rand_like(u)
            v = v + torch.rand_like(v)

        # 2. Compute directions
        u_normalized = (u[None, :] - width * 0.5) / focals[:, None]
        v_normalized = (v[None, :] - height * 0.5) / focals[:, None]
        dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)
        dirs_normalized = torch.nn.functional.normalize(dirs, dim=-1)  # [N, num_rays, 3]

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

        return dirs_normalized, sampled_rgb  # [N, num_rays, 3]
