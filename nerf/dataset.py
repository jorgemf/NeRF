from typing import Tuple
import os
from enum import Enum, auto
import requests
from zipfile import ZipFile
from pathlib import Path
import typer
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .config import DATASET_PATH, DataFormat
from .data_loaders.blender import load_blender_data
from .data_loaders.llff import load_llff_data
from .data_loaders.linemod import load_LINEMOD_data
from .data_loaders.scannet import load_scannet_data
from .data_loaders.deepvoxels import load_dv_data
from .ray_utils import get_rays

app = typer.Typer()


class DatasetType(Enum):
    """
    Enum for dataset types: train, validation and test
    """
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


@app.command()
def download() -> None:
    """
    Downloads example images from the foundational NeRF paper.
    """
    print("Downloading example images...")
    urls = [
        "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
    ]
    for url in urls:
        r = requests.get(url, allow_redirects=True)
        filename = Path(url).name
        file_path = DATASET_PATH.joinpath(filename)
        with open(file_path, 'wb') as f:
            f.write(r.content)
        if filename.endswith(".zip"):
            with ZipFile(file_path, 'r') as f:
                f.extractall(DATASET_PATH)
    print("Done.")


def load_data(dataset_path: str, format: DataFormat,  white_bkgd: bool = True, factor: int = 8,
              spherify: bool = True, llffhold: int = 8, no_ndc: bool = True, half_res: bool = True,
              testskip: int = 8, scannet_sceneID: str = "scene0000_00", shape: str = "armchair")\
        -> Tuple[np.ndarray, dict, Tuple[int, int], float, Tuple[float, float]]:
    """
    :param dataset_path: the path of the dataset
    :param format: the format of the dataset
    :param white_bkgd: set to render synthetic data on a white bkgd (always use for dvoxels)
    :param factor:  downsample factor for LLFF images
    :param spherify:  set for spherical 360 scenes
    :param llffhold: will take every 1/N images as LLFF test set, paper uses 8
    :param no_ndc: do not use normalized device coordinates (set for non-forward facing scenes)
    :param half_res: load blender synthetic data at 400x400 instead of 800x800
    :param testskip: will load 1/N images from test/val sets, useful for large datasets like deepvoxels
    :param scannet_sceneID:  sceneID to load from scannet
    :param shape: shape to load from deepvoxels. options : armchair / cube / greek / vase
    :return: a tuple of the bounding box, the data, the height and width of the images,
    the focal length and the near and far values
    """
    bounding_box = None
    K = None
    if format == DataFormat.LLFF:
        images, poses, bds, render_poses, i_test, bounding_box = \
            load_llff_data(dataset_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, dataset_path)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(images.shape[0])[::llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif format == DataFormat.BLENDER:
        images, poses, render_poses, hwf, i_split, bounding_box = \
            load_blender_data(dataset_path, half_res, testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, dataset_path)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif format == DataFormat.SCANNET:
        images, poses, render_poses, hwf, i_split, bounding_box = \
            load_scannet_data(dataset_path, scannet_sceneID, half_res)
        print('Loaded scannet', images.shape, render_poses.shape, hwf, dataset_path)
        i_train, i_val, i_test = i_split

        near = 0.1
        far = 10.0

    elif format == DataFormat.LINEMOD:
        images, poses, render_poses, hwf, K, i_split, near, far = \
            load_LINEMOD_data(dataset_path, half_res, testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif format == DataFormat.DEEPVOXELS:

        images, poses, render_poses, hwf, i_split = \
            load_dv_data(scene=shape, basedir=dataset_path, testskip=testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, dataset_path)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset format', format, 'exiting')
        return

    height, width, focal = hwf
    height, width = int(height), int(width)

    data = {
        DatasetType.TRAIN.name: {
            'images': images[i_train],
            'poses': poses[i_train],
        },
        DatasetType.VAL.name: {
            'images': images[i_val],
            'poses': poses[i_val],
        },
        DatasetType.TEST.name: {
            'images': images[i_test],
            'poses': poses[i_test],
        },
    }

    return bounding_box, data, (height, width), focal, (near, far)


class CustomRaysDataset(Dataset):
    """
    Custom dataset for images. It contains all the rays (source and direction) and the rgb values
    for each ray.
    """
    def __init__(self, images, poses, height, width, focal_length, near, far, full_images=False):
        super().__init__()
        self.images = images
        self.poses = poses
        self.height = height
        self.width = width
        self.focal_length = focal_length
        self.near = near
        self.far = far
        self.full_images = full_images

        print(f'get rays, images: {len(images)}')
        all_rays_o = []
        all_rays_d = []
        all_rgb = []
        for image, pose in zip(images, poses):
            pose = torch.from_numpy(pose)
            rgb = torch.from_numpy(image)
            rays_origen, rays_directions = get_rays(height, width, focal_length, pose)

            if full_images:
                rays_origen = rays_origen.view(1, *rays_origen.shape)
                rays_directions = rays_directions.view(1, *rays_directions.shape)
                rgb = rgb.view(1, *rgb.shape)
            else:
                rays_origen = rays_origen.view(-1, 3)
                rays_directions = rays_directions.view(-1, 3)
                rgb = rgb.view(-1, 3)

            all_rays_o.append(rays_origen)
            all_rays_d.append(rays_directions)
            all_rgb.append(rgb)
        self.rays_origen = torch.cat(all_rays_o, dim=0)
        self.rays_directions = torch.cat(all_rays_d, dim=0)
        self.rgb = torch.cat(all_rgb, dim=0)

    def __len__(self):
        return len(self.rays_origen)  # number of rays

    def __getitem__(self, idx):
        return self.rays_origen[idx], self.rays_directions[idx], self.rgb[idx]


def get_datasets(dataset_path: str, format: DataFormat, batch_size: int) \
        -> Tuple[DataLoader, torch.Tensor]:
    """
    Gets a dataset from the given path and format.
    :param dataset_path: path of the dataset
    :param format: the format of the dataset
    :param batch_size: the batch size you want the dataset to be
    :return: a tuple of the dataset and the bounding box as a tensor
    """
    bounding_box, data, (height, width), focal, (near, far) = load_data(dataset_path, format)
    data_loaders = {}
    for type in DatasetType:
        images = data[type.name]['images']
        poses = data[type.name]['poses']
        ds = CustomRaysDataset(images, poses, height, width, focal, near, far,
                               full_images=type != DatasetType.TRAIN)
        data_loader = DataLoader(ds,
                                 batch_size=batch_size if type == DatasetType.TRAIN else 1,
                                 shuffle=(type == DatasetType.TRAIN),
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=os.cpu_count() if type == DatasetType.TRAIN else 2)
        data_loaders[type] = data_loader

    return data_loaders, bounding_box
