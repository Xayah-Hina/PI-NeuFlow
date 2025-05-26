from config import DatasetConfig, TrainConfig
from src.trainer import Trainer
from src.dataset import PINeuFlowDataset
import torch
import tyro
import os
import dataclasses
import datetime


@dataclasses.dataclass
class AppConfig:
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)


if __name__ == "__main__":
    cfg = tyro.cli(AppConfig)
    device = torch.device(cfg.train.device)

    # load checkpoint
    model_state_dict = None
    ckpt_path = os.path.join(cfg.train.workspace, cfg.train.ckpt)
    if os.path.exists(ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_ckpt = checkpoint_dict['train_cfg']
        cfg_ckpt.train.mode = cfg.train.mode
        cfg = cfg_ckpt
        model_state_dict = checkpoint_dict['model']

    trainer = Trainer(
        name="PI-NeuFlow",
        workspace=cfg.train.workspace,
        model=cfg.train.model,
        model_state_dict=model_state_dict,
        learning_rate_encoder=1e-3,
        learning_rate_network=1e-3,
        use_fp16=cfg.dataset.use_fp16,
        use_compile=cfg.train.compile,
        use_tcnn=cfg.train.use_tcnn,
        device=device
    )

    if cfg.train.mode == 'train':
        trainer.train(
            train_dataset=PINeuFlowDataset(
                dataset_path=os.path.join(cfg.dataset.data_dir, cfg.dataset.dataset_dir),
                dataset_type='train',
                downscale=cfg.dataset.downscale,
                use_preload=cfg.dataset.use_preload,
                use_fp16=cfg.dataset.use_fp16,
                device=device,
            ),
            valid_dataset=PINeuFlowDataset(
                dataset_path=os.path.join(cfg.dataset.data_dir, cfg.dataset.dataset_dir),
                dataset_type='val',
                downscale=cfg.dataset.downscale,
                use_preload=cfg.dataset.use_preload,
                use_fp16=cfg.dataset.use_fp16,
                device=device,
            ),
            max_epochs=10,
        )
        state = {
            'train_cfg': cfg,
            'model': trainer.model.state_dict(),
        }
        torch.save(state, os.path.join(trainer.states.workspace, f'checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
    elif cfg.train.mode == 'test':
        trainer.test(
            test_dataset=PINeuFlowDataset(
                dataset_path=os.path.join(cfg.dataset.data_dir, cfg.dataset.dataset_dir),
                dataset_type='test',
                downscale=cfg.dataset.downscale,
                use_preload=cfg.dataset.use_preload,
                use_fp16=cfg.dataset.use_fp16,
                device=device,
            ),
        )
    else:
        raise ValueError(f"Unknown mode {cfg.train.mode}.")
