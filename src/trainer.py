from .dataset import PINeuFlowDataset, FrustumsSampler
from .network import NetworkPINeuFlow
from .renderer import VolumeRenderer
from .visualizer import visualize_rays
from .cuda_extensions import raymarching
import torch
import torch.utils.tensorboard
import tqdm
import typing
import types
import math
import os


class Trainer:
    def __init__(self,
                 name: str,
                 workspace: str,
                 model: typing.Literal['hyfluid', 'PI-NeuFlow'],
                 learning_rate_encoder: float,
                 learning_rate_network: float,
                 use_fp16: bool,
                 use_compile: bool,
                 device: torch.device,
                 ):
        # self.model
        if model == 'PI-NeuFlow':
            self.model = NetworkPINeuFlow(
                encoding_xyzt='hyfluid',
                encoding_dir='sphere_harmonics',
                num_layers_sigma=3,
                num_layers_color=3,
                hidden_dim_sigma=64,
                hidden_dim_color=64,
                geo_feat_dim=32,
            ).to(device)
        else:
            raise NotImplementedError(f"Model {model} is not implemented.")

        # self.optimizer
        self.optimizer = torch.optim.RAdam(self.model.get_params(learning_rate_encoder, learning_rate_network), betas=(0.9, 0.99), eps=1e-15)

        # self.scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

        # occupancy grid
        self.cascade = 1
        self.grid_size = 128
        self.density_grid = torch.zeros([self.cascade, self.grid_size ** 3]).to(device)
        self.density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8).to(device)
        self.mean_density = 0
        self.iter_density = 0
        self.step_counter = torch.zeros(16, 2, dtype=torch.int32).to(device)  # 16 is hardcoded for averaging...
        self.mean_count = 0
        self.local_step = 0
        self.bound = 1.0
        self.density_scale = 1
        self.density_thresh = 0.01

        # self.scheduler
        target_lr_ratio = 0.0001
        gamma = math.exp(math.log(target_lr_ratio) / 20000)
        warmup_d = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=2000)
        main_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_d, main_scheduler_d], milestones=[2000])

        # self.states
        self.states = types.SimpleNamespace()
        self.states.name = name
        self.states.workspace = workspace
        self.states.use_fp16 = use_fp16
        self.states.device = device
        self.states.epoch = 0
        self.states.iteration = 0

        # debug
        self.writer = torch.utils.tensorboard.SummaryWriter(os.path.join(workspace, "run", name))

        if use_compile:
            self.compiled_render = torch.compile(VolumeRenderer.render)
            self.compiled_render_no_grad = torch.compile(VolumeRenderer.render_no_grad)
        else:
            self.compiled_render = VolumeRenderer.render
            self.compiled_render_no_grad = VolumeRenderer.render_no_grad

        self.load_checkpoint(checkpoint=os.path.join(workspace, 'checkpoint.pth'))

    @torch.no_grad()
    def mark_untrained_grid(self, poses, fx, fy, cx, cy, S=64):
        # poses: [B, 4, 4]

        B = poses.shape[0]

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    world_xyzs = (2 * coords.float().to(poses.dtype) / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

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
                            mask_z = cam_xyzs[:, :, 2] > 0  # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)  # [N]

                            # update count
                            count[cas, indices] += mask
                            head += S

        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)

        # full update.
        if self.iter_density < 16:
            # if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:

                        # construct points
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long()  # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.model.sigma(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4  # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device)  # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long()  # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)  # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask]  # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.model.sigma(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()  # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

    def train(self, train_dataset: PINeuFlowDataset, valid_dataset: PINeuFlowDataset | None, max_epochs: int):
        self.model.train()

        # preprocessing
        poses = train_dataset.poses
        self.mark_untrained_grid(poses, train_dataset.focals[0].item(), train_dataset.focals[0].item(), train_dataset.widths / 2, train_dataset.heights / 2)
        # preprocessing

        sampler = FrustumsSampler(dataset=train_dataset, num_rays=1024, randomize=True)
        train_loader = sampler.dataloader(batch_size=1)
        for epoch in range(self.states.epoch, max_epochs):
            self.states.epoch += 1

            for i, data in enumerate(tqdm.tqdm(train_loader)):
                data: dict

                for _ in range(train_loader.batch_size):
                    self.states.iteration += 1

                    self.optimizer.zero_grad()
                    with torch.amp.autocast('cuda', enabled=self.states.use_fp16):
                        rgb_map = self.compiled_render(
                            network=self.model,
                            rays_o=data['rays_o'][_],  # [N, C]
                            rays_d=data['rays_d'][_],  # [1]
                            time=data['times'][_],  # [N, 3]
                            extra_params=train_dataset.extra_params,
                            randomize=True,
                        )
                        gt_pixels = data['pixels'][_]  # [N, 3]
                        img_loss = torch.nn.functional.mse_loss(rgb_map, gt_pixels)  # [N, 3]
                    self.scaler.scale(img_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    self.writer.add_scalar("train/img_loss", img_loss, self.states.iteration)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.states.iteration)

                    # visualize_rays(train_rays_o.cpu().numpy(), train_rays_d.cpu().numpy(), size=0.1)

        self.save_checkpoint()
        # self.test(valid_dataset)

    def test(self, test_dataset: PINeuFlowDataset):
        import imageio.v3 as imageio
        import numpy as np
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()
        self.model.eval()

        sampler = FrustumsSampler(dataset=test_dataset, num_rays=-1, randomize=False)
        train_loader = sampler.dataloader(batch_size=1)

        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(train_loader)):
                data: dict

                for _ in range(train_loader.batch_size):
                    self.states.iteration += 1

                    gt_pixels = data['pixels'][_].reshape(test_dataset.heights, test_dataset.widths, 3)  # [H, W, 3]
                    rgb_map_final = []
                    total_ray_size = data['rays_o'][_].shape[0]
                    batch_ray_size = 1024 * 8
                    for start in range(0, total_ray_size, batch_ray_size):
                        rgb_map = self.compiled_render_no_grad(
                            network=self.model,
                            rays_o=data['rays_o'][_][start:start + batch_ray_size],  # [N, 3]
                            rays_d=data['rays_d'][_][start:start + batch_ray_size],  # [N, 3]
                            time=data['times'][_],  # [1]
                            extra_params=test_dataset.extra_params,
                            randomize=False,
                        )  # [N, 3]
                        rgb_map_final.append(rgb_map.detach().cpu())
                    rgb_map_final = torch.cat(rgb_map_final, dim=0).reshape(test_dataset.heights, test_dataset.widths, 3)

                    rgb8 = (255 * np.clip(rgb_map_final.numpy(), 0, 1)).astype(np.uint8)
                    gt8 = (255 * np.clip(gt_pixels.cpu().numpy(), 0, 1)).astype(np.uint8)
                    imageio.imwrite(os.path.join(f'{self.states.workspace}', 'rgb_{:03d}_{:03d}.png'.format(i, _)), rgb8)
                    imageio.imwrite(os.path.join(f'{self.states.workspace}', 'gt_{:03d}_{:03d}.png'.format(i, _)), gt8)

    def save_checkpoint(self):
        state = {
            'model': self.model.state_dict(),
        }

        torch.save(state, os.path.join(self.states.workspace, 'checkpoint.pth'))

    def load_checkpoint(self, checkpoint):
        if not os.path.exists(checkpoint):
            pass
        checkpoint_dict = torch.load(checkpoint, map_location=self.states.device)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print(f'[load_checkpoint] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}')
