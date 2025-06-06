import os
import dataclasses
import typing


@dataclasses.dataclass
class DatasetConfig:
    data_dir: str = dataclasses.field(default=os.path.abspath(os.path.join(os.getcwd(), 'data')), metadata={'help': 'data directory'})
    dataset_dir: str = dataclasses.field(default='', metadata={'help': 'base directory of dataset'})
    background_color: typing.Literal['white', 'black'] = dataclasses.field(default='black', metadata={'help': 'background color of the dataset'})
    downscale: int = dataclasses.field(default=1, metadata={'help': 'downscale factor for images'})

    use_preload: bool = dataclasses.field(default=True, metadata={'help': 'preload all data into GPU, accelerate training but use more GPU memory'})
    use_fp16: bool = dataclasses.field(default=True, metadata={'help': 'use amp mixed precision training'})

    def __post_init__(self):
        if not os.path.exists(os.path.join(self.data_dir, self.dataset_dir)):
            raise FileNotFoundError(f"Dataset directory {self.data_dir}/{self.dataset_dir} does not exist.")


@dataclasses.dataclass
class TrainConfig:
    # checkpoint options
    select_ckpt: bool = dataclasses.field(default=False, metadata={'help': 'select a checkpoint to continue training'})

    # optional options
    workspace: str = dataclasses.field(default='workspace', metadata={'help': 'workspace directory'})
    model: typing.Literal['pinf', 'hyfluid', 'PI-NeuFlow'] = dataclasses.field(default="PI-NeuFlow", metadata={'help': 'model name'})
    mode: typing.Literal["train", "test"] = dataclasses.field(default="train", metadata={"help": "mode of training"})
    compile: bool = dataclasses.field(default=False, metadata={'help': 'use torch.compile to compile the model for faster training'})
    device: str = dataclasses.field(default='cuda:0', metadata={'help': 'device to use, usually setting to None is OK. (auto choose device)'})
    use_tcnn: bool = dataclasses.field(default=True, metadata={'help': 'use tcnn for encoding and network'})
