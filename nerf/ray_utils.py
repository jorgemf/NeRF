from typing import Tuple

import torch


def get_rays(height: int, width: int, focal_length: float, c2w: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get rays from camera to each voxel.
    :param height: height of the target image
    :param width: width of the target image
    :param focal_length: focal length of the camera
    :param c2w: camera to view matrix
    :return: a tuple of rays origin and rays direction as height x width x 3 tensors
    """
    i, j = torch.meshgrid(torch.linspace(0, width - 1, width),
                          torch.linspace(0, height - 1, height))
    i = i.t()
    j = j.t()
    directions = torch.stack([(i - width / 2) / focal_length,
                              -(j - height / 2) / focal_length,
                              -torch.ones_like(i)], -1)
    rays_directions = directions @ c2w[:3, :3].T
    rays_directions = rays_directions / torch.norm(rays_directions, dim=-1, keepdim=True)
    rays_origin = c2w[:3, -1].expand(rays_directions.shape)
    return rays_origin, rays_directions  # [H, W, 3], [H, W, 3]


def get_ndc_rays(height: int, width: int, focal: float, near: float, rays_o: torch.Tensor,
                 rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    :param height: height of the target image
    :param width: width of the target image
    :param focal_length: focal length of the camera
    :param c2w: camera to view matrix
    :return: a tuple of rays origin and rays direction as height x width x 3 tensors
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (width / (2. * focal)) * ox_oz
    o1 = -1. / (height / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (width / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (height / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d
