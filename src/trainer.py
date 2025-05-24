from .dataset import PINeuFlowDataset, FrustumsSampler
from .network import NetworkPINeuFlow
from .renderer import VolumeRenderer
from .visualizer import visualize_rays
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

        self.scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

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

        try:
            self.compiled_render = torch.compile(VolumeRenderer.render)
        except Exception:
            print("torch.compile is not available. Using the original render function.")
            self.compiled_render = VolumeRenderer.render

    def train(self, train_dataset: PINeuFlowDataset, valid_dataset: PINeuFlowDataset | None, max_epochs: int):
        self.model.train()

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

        self.test(valid_dataset)

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

        for i, data in enumerate(tqdm.tqdm(train_loader)):
            data: dict

            for _ in range(train_loader.batch_size):
                self.states.iteration += 1

                self.optimizer.zero_grad()
                gt_pixels = data['pixels'][_].reshape(test_dataset.heights, test_dataset.widths, 3)  # [H, W, 3]
                rgb_map_final = []
                total_ray_size = data['rays_o'][_].shape[0]
                batch_ray_size = 1024 * 8
                for start in range(0, total_ray_size, batch_ray_size):
                    rgb_map = self.compiled_render(
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
