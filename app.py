from config import DatasetConfig, TrainConfig
from src.trainer import Trainer
from src.dataset import PINeuFlowDataset
import torch
import tyro
import os
import dataclasses
import datetime


def open_file_dialog():
    from tkinter import filedialog, Tk
    while True:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(title="Select a checkpoint file", initialdir="workspace", filetypes=[("Checkpoint files", "*.pth")])
        root.destroy()

        if file_path and os.path.isfile(file_path):
            print("Successfully loaded checkpoint:", file_path)
            return file_path
        else:
            print("invalid file path, please select a valid checkpoint file.")


@dataclasses.dataclass
class AppConfig:
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)


if __name__ == "__main__":
    cfg = tyro.cli(AppConfig)
    device = torch.device(cfg.train.device)

    model_state_dict = None
    # load checkpoint
    if cfg.train.select_ckpt:
        checkpoint = open_file_dialog()
        ckpt_path = os.path.join(cfg.train.workspace, checkpoint)
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
        os.makedirs(os.path.join(trainer.states.workspace, "images"), exist_ok=True)
        torch.save(state, os.path.join(trainer.states.workspace, "images", f'checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
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
