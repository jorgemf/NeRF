import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for NeRF.
    """

    def __init__(self, dim: int, num_freqs: int):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.num_freqs = num_freqs
        self.freqs = 2.0 ** torch.arange(num_freqs)

    def forward(self, x):
        # Expand dimensions
        x = x.unsqueeze(-2)

        # Scale input by frequencies
        scaled_x = x * self.freqs.view(1, 1, -1)

        # Apply sin and cos to even and odd indices
        x = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)

        # Flatten last two dimensions
        x = x.view(*x.shape[:-2], -1)

        return x


def hash_encoder(coords, log2_hashmap_size) -> torch.Tensor:
    """
    Hashes the coordinates to a hashmap of size 2^log2_hashmap_size
    :param coords: coordinates of each point in space. This function can process up to 7 dim
    coordinates
    :param log2_hashmap_size: log2 of the size of the hashmap
    :return: the hashed coordinates as a tensor
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


class HashEncoder(nn.Module):
    """
    Hash encoder module for NeRF.
    """

    def __init__(self, bounding_box=(torch.tensor([-1., -1., -1.]), torch.tensor([1., 1., 1.])),
                 n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16,
                 finest_resolution=512):
        """
        :param bounding_box: the bounding box of the scene
        :param n_levels: the number of leves of the hash encoder
        :param n_features_per_level: number of features per level
        :param log2_hashmap_size: the log2 of the size of the hashmap
        :param base_resolution: the base (minimum) resolution of the hash encoder
        :param finest_resolution: the finest (maximum) resolution of the hash encoder
        """
        super(HashEncoder, self).__init__()
        assert n_levels > 0
        assert n_features_per_level > 0
        assert log2_hashmap_size > 0
        assert base_resolution > 0
        assert finest_resolution > 0
        assert finest_resolution >= base_resolution
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        # grow factor
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) /
                           (n_levels - 1))

        # initialize embeddings
        embed_list = []
        for _ in range(n_levels):
            embed = nn.Embedding(2 ** self.log2_hashmap_size, self.n_features_per_level)
            nn.init.uniform_(embed.weight, a=-0.0001, b=0.0001)
            embed_list.append(embed)
        self.embeddings = nn.ModuleList(embed_list)

        self.box_offsets = torch.tensor(
            [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])

        box_min, box_max = bounding_box
        resolutions = [math.floor(self.base_resolution * self.b ** i) for i in range(n_levels)]
        self.resolutions = torch.tensor(resolutions)
        self.grid_sizes = torch.cat([(box_max - box_min) / res for res in resolutions], -1)

    def get_voxel_vertices(self,
                           xyz: torch.Tensor,
                           level: int,
                           log2_hashmap_size: torch.Tensor) \
            -> tuple[: torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the vertices of the voxels for the given coordinates and level.
        :param xyz: the 3D coordinates of the points, usually bach size x 3
        :param level: the level of the hash encoder
        :param log2_hashmap_size: the log2 of the size of the hashmap
        :return: a tuple of the min and max vertices of the voxels and the hashed voxel indices
        """
        box_min, box_max = self.bounding_box

        # clip the points outside the bounding box
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

        grid_size = self.grid_sizes[level]

        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        ones = torch.tensor([1.0, 1.0, 1.0]).to(xyz.device)
        voxel_max_vertex = voxel_min_vertex + ones * grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offsets
        hashed_voxel_indices = hash_encoder(voxel_indices, log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def trilinear_interp(self,
                         xyz: torch.Tensor,
                         voxel_min_vertex: torch.Tensor,
                         voxel_max_vertex: torch.Tensor,
                         voxel_embedds: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation of the given coordinates.
        :param xyz: the 3D coordinates of the points, usually bach size x 3
        :param voxel_min_vertex: minimum vertex of the voxels
        :param voxel_max_vertex: maximum vertex of the voxels
        :param voxel_embedds: embeddings of the voxels
        :return:  the interpolated embeddings as a tensor
        """

        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (xyz - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + \
              voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + \
              voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + \
              voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + \
              voxel_embedds[:, 7] * weights[:, 0][:, None]

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the hash encoder.
        :param xyz: the 3D coordinates of the points, usually bach size x 3
        :return: a tuple of the embedded coordinates and a mask of the points that are on the
        surface
        """
        x_embedded_all = []
        if xyz.device != self.resolutions.device:
            self.bounding_box = (self.bounding_box[0].to(xyz.device),
                                 self.bounding_box[1].to(xyz.device))
            self.box_offsets = self.box_offsets.to(xyz.device)
            self.grid_sizes = self.grid_sizes.to(xyz.device)
            self.resolutions = self.resolutions.to(xyz.device)

        for level in range(self.n_levels):
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = \
                self.get_voxel_vertices(xyz, level, self.log2_hashmap_size)

            voxel_embedds = self.embeddings[level](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(xyz, voxel_min_vertex, voxel_max_vertex,
                                               voxel_embedds)
            x_embedded_all.append(x_embedded)

        with torch.no_grad():
            box_min, box_max = self.bounding_box
            keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
            keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask


class SphericalEncoder(nn.Module):
    """
    Spherical encoder module for NeRF. This is used to create the embeddings of the rays directions.
    """

    def __init__(self, input_dim: int = 3, degree: int = 4):
        """
        :param input_dim: the dimension of the input
        :param degree: the degree of the spherical harmonics
        """

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spherical encoder.
        :param input: usually a 3D direction
        :return: the spherical harmonics of the input
        """

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype,
                             device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                # result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
