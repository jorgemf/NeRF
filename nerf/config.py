from pathlib import Path
import torch
from enum import Enum

RUNS_DIR = Path("./runs")
DTYPE = torch.float
DATASET_PATH = Path("./data")


class DataFormat(str, Enum):
    LLFF = "llff"
    BLENDER = "blender"
    LINEMOD = "linemod"
    SCANNET = "scannet"
    DEEPVOXELS = "deepvoxels"


def setup():
    device = get_device()
    print(f"Using {device} device")
    if device == "cuda":
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__CUDA Name:', torch.cuda.get_device_name(0))
        GB = 1024*1024*1024
        print('__CUDA Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / GB)

    DATASET_PATH.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device
