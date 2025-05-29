from .dataset import PINeuFlowDataset, FrustumsSampler
from .network import NetWorkPINeuFlow
from .network_pinf import NetworkPINF
import torch
import torch.utils.tensorboard
import tqdm
import typing
import types
import math
import os


def fade_in_weight(step, start, duration):
    return min(max((float(step) - start) / duration, 0.0), 1.0 - 1e-8)


class Trainer:
    def __init__(self,
                 name: str,
                 workspace: str,
                 model: typing.Literal['pinf', 'hyfluid', 'PI-NeuFlow'],
                 model_state_dict,
                 background_color: typing.Literal['white', 'black'],
                 learning_rate_encoder: float,
                 learning_rate_network: float,
                 use_fp16: bool,
                 use_compile: bool,
                 use_tcnn: bool,
                 device: torch.device,
                 ):
        # self.model
        if model == 'PI-NeuFlow':
            self.model = NetWorkPINeuFlow(
                encoding_xyzt='hyfluid',
                encoding_dir='sphere_harmonics',
                background_color=background_color,
                use_tcnn=use_tcnn,
                num_layers_sigma=3,
                num_layers_color=3,
                hidden_dim_sigma=64,
                hidden_dim_color=64,
                geo_feat_dim=32,
            ).to(device)
        elif model == 'pinf':
            self.model = NetworkPINF(
                netdepth=8,
                netwidth=256,
                input_ch=4,
                use_viewdirs=False,
                omega=6,
                use_first_omega=True,
                fading_layers=50000,
                background_color=background_color,
            ).to(device)
        else:
            raise NotImplementedError(f"Model {model} is not implemented.")
        if model_state_dict:
            self.model.load_state_dict(model_state_dict, strict=False)

        # self.optimizer
        self.optimizer = torch.optim.RAdam(self.model.get_params(learning_rate_encoder, learning_rate_network), betas=(0.9, 0.99), eps=1e-15)

        # self.scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

        # self.scheduler
        target_lr_ratio = 0.1
        gamma = math.exp(math.log(target_lr_ratio) / 100000)
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
            self.compiled_render = torch.compile(self.model.render)
            self.compiled_render_no_grad = torch.compile(self.model.render_no_grad)
        else:
            self.compiled_render = self.model.render
            self.compiled_render_no_grad = self.model.render_no_grad

    def train(self, train_dataset: PINeuFlowDataset, valid_dataset: PINeuFlowDataset | None, max_epochs: int, cfg):
        self.model.train()

        sampler = FrustumsSampler(dataset=train_dataset, num_rays=1024 * 2, randomize=True)
        train_loader = sampler.dataloader(batch_size=1)

        # with torch.amp.autocast('cuda', enabled=self.states.use_fp16):
        #     sampler.mark_untrained_grid(
        #         poses=train_dataset.poses,
        #         fx=train_dataset.focals[0].item(),
        #         fy=train_dataset.focals[0].item(),
        #         cx=train_dataset.widths / 2,
        #         cy=train_dataset.heights / 2,
        #     )
        #     sampler.update_extra_state(network=self.model)
        # from .visualizer import visualize_density_grid
        # visualize_density_grid(sampler.density_grid[0, 0], grid_size=sampler.grid_size, poses=train_dataset.poses.detach().cpu())
        # visualize_density_grid(sampler.density_grid[0, 0], grid_size=sampler.grid_size, poses=None)

        for epoch in tqdm.trange(self.states.epoch, max_epochs):
            self.states.epoch += 1
            if self.states.epoch % 10 == 0:
                import datetime
                state = {
                    'train_cfg': cfg,
                    'model': self.model.state_dict(),
                }
                torch.save(state, os.path.join(self.states.workspace, f'checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))

            for i, data in enumerate(train_loader):
                data: dict

                for _ in range(train_loader.batch_size):
                    self.states.iteration += 1
                    tempo_fading = fade_in_weight(self.states.iteration, 0, 10000)

                    self.optimizer.zero_grad()
                    with torch.amp.autocast('cuda', enabled=self.states.use_fp16):
                        rgb_map, depth_map, extras = self.compiled_render(
                            rays_o=data['rays_o'][_],  # [N, C]
                            rays_d=data['rays_d'][_],  # [1]
                            time=data['times'][_],  # [N, 3]
                            extra_params=train_dataset.extra_params,
                            randomize=True,
                        )
                        gt_pixels = data['pixels'][_]  # [N, 3]
                        img_loss_d = torch.nn.functional.mse_loss(rgb_map, gt_pixels)  # [N, 3]
                        img_loss_s = torch.nn.functional.mse_loss(extras['static'].rgb, gt_pixels)  # [N, 3]
                        loss = img_loss_d * tempo_fading + img_loss_s * (1.0 - tempo_fading)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    self.writer.add_scalar("train/loss", loss, self.states.iteration)
                    self.writer.add_scalar("train/img_loss_d", img_loss_d, self.states.iteration)
                    self.writer.add_scalar("train/img_loss_s", img_loss_s, self.states.iteration)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.states.iteration)

                    # from .visualizer import visualize_rays
                    # visualize_rays(train_rays_o.cpu().numpy(), train_rays_d.cpu().numpy(), size=0.1)

        # self.test(valid_dataset)

    def test(self, test_dataset: PINeuFlowDataset):
        import imageio.v2 as imageiov2
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
            frames = []
            for i, data in enumerate(tqdm.tqdm(train_loader)):
                data: dict

                for _ in range(train_loader.batch_size):
                    self.states.iteration += 1

                    gt_pixels = data['pixels'][_].reshape(test_dataset.heights, test_dataset.widths, 3)  # [H, W, 3]
                    rgb_map_final = []
                    depth_map_final = []
                    rgb_static_map_final = []
                    rgb_dynamic_map_final = []
                    total_ray_size = data['rays_o'][_].shape[0]
                    batch_ray_size = 1024 * 8
                    for start in range(0, total_ray_size, batch_ray_size):
                        with torch.amp.autocast('cuda', enabled=self.states.use_fp16):
                            rgb_map, depth_map, extras = self.compiled_render_no_grad(
                                rays_o=data['rays_o'][_][start:start + batch_ray_size],  # [N, 3]
                                rays_d=data['rays_d'][_][start:start + batch_ray_size],  # [N, 3]
                                time=data['times'][_],  # [1]
                                extra_params=test_dataset.extra_params,
                                randomize=False,
                            )  # [N, 3]
                        rgb_map_final.append(rgb_map.detach().cpu())
                        rgb_static_map_final.append(extras['static'].rgb.detach().cpu())
                        rgb_dynamic_map_final.append(extras['dynamic'].rgb.detach().cpu())
                        # depth_map_final.append(depth_map.detach().cpu())
                    rgb_map_final = torch.cat(rgb_map_final, dim=0).reshape(test_dataset.heights, test_dataset.widths, 3)
                    # depth_map_final = torch.cat(depth_map_final, dim=0).reshape(test_dataset.heights, test_dataset.widths, 1)
                    rgb_static_map_final = torch.cat(rgb_static_map_final, dim=0).reshape(test_dataset.heights, test_dataset.widths, 3)
                    rgb_dynamic_map_final = torch.cat(rgb_dynamic_map_final, dim=0).reshape(test_dataset.heights, test_dataset.widths, 3)

                    rgb8 = (255 * np.clip(rgb_map_final.numpy(), 0, 1)).astype(np.uint8)
                    # depth8 = (255 * np.clip(depth_map_final.expand_as(rgb_map_final).numpy(), 0, 1)).astype(np.uint8)
                    gt8 = (255 * np.clip(gt_pixels.cpu().numpy(), 0, 1)).astype(np.uint8)
                    rgb_static8 = (255 * np.clip(rgb_static_map_final.numpy(), 0, 1)).astype(np.uint8)
                    rgb_dynamic8 = (255 * np.clip(rgb_dynamic_map_final.numpy(), 0, 1)).astype(np.uint8)

                    frame = np.concatenate([gt8, rgb8, rgb_static8, rgb_dynamic8], axis=1)
                    os.makedirs(os.path.join(self.states.workspace, 'images'), exist_ok=True)
                    imageio.imwrite(os.path.join(f'{self.states.workspace}', 'images', 'output_{:03d}_{:03d}.png'.format(i, _)), frame)
                    frames.append(frame)
            imageiov2.mimsave(os.path.join(f'{self.states.workspace}', 'images', 'output.mp4'), frames, fps=24)
