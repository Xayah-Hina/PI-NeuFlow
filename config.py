import os
import dataclasses
import typing


@dataclasses.dataclass
class DatasetConfig:
    # required options
    dataset_dir: str = dataclasses.field(metadata={'help': 'base directory of dataset'})

    # optional options
    data_dir: str = dataclasses.field(default=os.path.abspath(os.path.join(os.getcwd(), 'data')), metadata={'help': 'data directory'})
    downscale: int = dataclasses.field(default=1, metadata={'help': 'downscale factor for images'})

    use_preload: bool = dataclasses.field(default=True, metadata={'help': 'preload all data into GPU, accelerate training but use more GPU memory'})
    use_fp16: bool = dataclasses.field(default=False, metadata={'help': 'use amp mixed precision training'})

    def __post_init__(self):
        if not os.path.exists(os.path.join(self.data_dir, self.dataset_dir)):
            raise FileNotFoundError(f"Dataset directory {self.data_dir}/{self.dataset_dir} does not exist.")


@dataclasses.dataclass
class TrainConfig:
    # optional options
    model: typing.Literal["PI-NeuFlow"] = dataclasses.field(default="PI-NeuFlow", metadata={'help': 'model name'})
    mode: typing.Literal["train", "test"] = dataclasses.field(default="train", metadata={"help": "mode of training"})
    device: str = dataclasses.field(default='cuda:0', metadata={'help': 'device to use, usually setting to None is OK. (auto choose device)'})
