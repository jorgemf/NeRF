import pdb
from math import exp, log, floor, log10
import torch
from .encoders import hash_encoder


def total_variation_loss(embeddings: torch.Tensor, min_resolution: torch.Tensor,
                         max_resolution: torch.Tensor, level: int, log2_hashmap_size: int,
                         n_levels: int = 16) -> torch.Tensor:
    """
    Compute total variation loss for a given level of the network.
    :param embeddings: the embeddings of the network
    :param min_resolution: minimum resolution of the network
    :param max_resolution: maximum resolution of the network
    :param level: current level of the network
    :param log2_hashmap_size: the log2 of the size of the hash map
    :param n_levels: number of levels of the network
    :return: the varational loss for the given level as a tensor
    """
    # Get resolution
    b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1))
    resolution = torch.tensor(floor(min_resolution * b ** level))

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50  # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(torch.clip(resolution / 10.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution - cube_size, (3,))
    idx = min_vertex + torch.stack([torch.arange(cube_size + 1) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2]), dim=-1)

    embedding_layer = embeddings[level]
    hashed_indices = hash_encoder(cube_indices, log2_hashmap_size).to(embedding_layer.weight.device)
    cube_embeddings = embedding_layer(hashed_indices)

    # Compute loss
    tv_x = torch.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
    tv_y = torch.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
    tv_z = torch.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()

    return (tv_x + tv_y + tv_z) / cube_size

def mse2psnr(mse: float) -> float:
    """
    Convert MSE to PSNR
    :param mse: mean square error
    :return: the signal to noise ratio
    """
    return -10 * log10(mse) / log(10.)
