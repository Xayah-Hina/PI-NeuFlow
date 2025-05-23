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
        self.states.device = device
        self.states.epoch = 0
        self.states.iteration = 0

        # debug
        self.writer = torch.utils.tensorboard.SummaryWriter(os.path.join(workspace, "run", name))

    def train(self, train_dataset: PINeuFlowDataset, valid_dataset: PINeuFlowDataset | None, max_epochs: int):
        self.model.train()

        sampler = FrustumsSampler(dataset=train_dataset, num_rays=1024)
        train_loader = sampler.dataloader(batch_size=1)
        for epoch in range(self.states.epoch, max_epochs):
            self.states.epoch += 1

            for i, data in enumerate(tqdm.tqdm(train_loader)):
                data: dict

                for _ in range(train_loader.batch_size):
                    self.states.iteration += 1

                    self.optimizer.zero_grad()
                    rgb_map = VolumeRenderer.render(
                        network=self.model,
                        rays_o=data['rays_o'][_],  # [N, C]
                        rays_d=data['rays_d'][_],  # [1]
                        time=data['times'][_],  # [N, 3]
                        extra_params=train_dataset.extra_params,
                    )
                    gt_pixels = data['pixels'][_]  # [N, 3]
                    img_loss = torch.nn.functional.mse_loss(rgb_map, gt_pixels)  # [N, 3]
                    img_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.writer.add_scalar("train/img_loss", img_loss, self.states.iteration)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.states.iteration)

                    # visualize_rays(train_rays_o.cpu().numpy(), train_rays_d.cpu().numpy(), size=0.1)

    def test(self, test_dataset: PINeuFlowDataset):
        self.model.eval()
