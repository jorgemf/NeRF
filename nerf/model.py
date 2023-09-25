from typing import List
import pdb
import torch
from torch import nn
from torch.distributions import Categorical
from .ray_utils import get_rays
from .encoders import HashEncoder, SphericalEncoder


class NeRFAbstract(nn.Module):
    def __init__(self):
        super(NeRFAbstract, self).__init__()

class NeRF(NeRFAbstract):

    def __init__(self,
                 xyz_channels: int,
                 d_channels: int,
                 layers: int = 8,
                 channels: int = 256,
                 residual_layers: List[int] = [4],
                 output_rgb: int = 3,
                 output_sigma: int = 1):
        super(NeRF, self).__init__()
        assert layers > 2
        assert channels > 0
        self.layers = layers
        self.channels = channels
        self.residual_layers = residual_layers

        pts_linears = []
        for i in range(self.layers):
            if i == 0:
                # xyz input
                pts_linears += [nn.Linear(xyz_channels, channels),
                                nn.ReLU()]
            elif i in self.residual_layers:
                # hidden residual layers
                pts_linears += [nn.Linear(channels + xyz_channels, channels),
                                nn.ReLU()]
            else:
                # hidden layers
                pts_linears += [nn.Linear(channels, channels),
                                nn.ReLU()]
        self.pts_linears = nn.ModuleList(pts_linears)

        # features vector
        self.pts_linears_features = nn.Sequential(nn.Linear(channels, channels),
                                                  nn.ReLU())
        # sigma
        self.pts_linears_sigma = nn.Sequential(nn.Linear(channels, output_sigma),
                                               nn.ReLU())

        # features vector + direction -> rgb
        self.rgb_linear = nn.Sequential(nn.Linear(channels + d_channels, channels // 2),
                                        nn.ReLU(),
                                        nn.Linear(channels // 2, output_rgb),
                                        nn.Sigmoid())

    def forward(self, xyz_encoded, direction_encoded):
        h = xyz_encoded
        h_residual = xyz_encoded
        for i in range(self.layers):
            if i in self.residual_layers:
                # add residual
                h = torch.cat([h_residual, h], -1)
            h = self.pts_linears[i * 2](h)  # linear
            h = self.pts_linears[i * 2 + 1](h)  # relu

        features = self.pts_linears_features(h)
        sigma = torch.squeeze(self.pts_linears_sigma(h))

        h = torch.cat([direction_encoded, features], -1)
        rgb = self.rgb_linear(h)

        return rgb, sigma



class NeRFSmall(NeRFAbstract):
    def __init__(self,
                 input_ch: int, input_ch_views: int,
                 num_layers: int = 2,
                 hidden_dim: int = 64,
                 geo_feat_dim: int = 15,
                 num_layers_color: int = 4,
                 hidden_dim_color: int = 64,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, xyz_encoded, direction_encoded):
        input_pts = xyz_encoded
        input_views = direction_encoded

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        color = torch.nn.functional.sigmoid(h)
        sigma = torch.nn.functional.relu(sigma)

        return color, sigma.unsqueeze(dim=-1)


class Render(object):

    def __init__(self,
                 embedder_points: HashEncoder,
                 embedder_directions: SphericalEncoder,
                 nerf: NeRFAbstract):
        self.embedder_points = embedder_points
        self.embedder_directions = embedder_directions
        self.nerf = nerf

    def render(self, height: int, width: int, focal_length: float, near: float, far: float, c2w,
               number_samples: int, chunk_size: int, perturb: bool, device=None):
        rays_o, rays_d = get_rays(height, width, focal_length, c2w)
        if device is not None:
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)

        image_shape = rays_d.shape
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        rgb, depth, accumulated, entropy = \
            self.render_rays(rays_o, rays_d, near, far, number_samples, chunk_size, perturb)

        rgb = torch.reshape(rgb, image_shape)
        depth = torch.reshape(depth, image_shape[:-1] + [depth.shape[-1]])
        accumulated = torch.reshape(accumulated, image_shape[:-1] + [accumulated.shape[-1]])

        return rgb, depth, accumulated, entropy

    def render_rays(self, rays_origin, rays_directions, near: float, far: float,
                    num_samples: int, chunk_size: int, perturb: bool):
        assert chunk_size % num_samples == 0, "Chunk size must be a multiple of num_samples"

        # near = near * torch.ones_like(rays_d[..., :1])
        # far = far * torch.ones_like(rays_d[..., :1])

        number_rays = rays_origin.shape[0]
        t_vals = torch.linspace(0., 1., steps=num_samples)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        # z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.to(rays_origin.device)

        z_vals = z_vals.expand([number_rays, num_samples])

        if perturb:
            mids = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(z_vals.device)

            z_vals = lower + (upper - lower) * t_rand

        xyz = rays_origin[..., None, :] + \
              rays_directions[..., None, :] * z_vals[..., :, None]  # [rays, samples, 3]

        raw_rgb = []
        raw_sigma = []
        chunk = chunk_size // num_samples
        for i in range(0, number_rays, chunk):
            xyz_batch = xyz[i:i + chunk]
            directions_batch = rays_directions[i:i + chunk]
            xyz_batch = xyz_batch.view(-1, 3)

            xyz_encoded, mask = self.embedder_points(xyz_batch)

            directions_batch_encoded = self.embedder_directions(directions_batch)
            channles_d = self.embedder_directions.out_dim
            directions_batch_encoded = directions_batch_encoded.view(-1, 1, channles_d)
            directions_batch_encoded = torch.tile(directions_batch_encoded, [1, num_samples, 1])
            directions_batch_encoded = directions_batch_encoded.view(-1, channles_d)

            rgb, sigma = self.nerf(xyz_encoded, directions_batch_encoded)

            # set 0 for invalid points
            mask = mask.view(-1, 1)
            sigma = sigma * mask
            rgb = rgb * mask

            raw_rgb.append(rgb.view(-1, num_samples, 3))
            raw_sigma.append(sigma.view(-1, num_samples))

        raw_rgb = torch.cat(raw_rgb, 0)
        raw_sigma = torch.cat(raw_sigma, 0)

        rgb, depth, accumulated, entropy = self.raw2outputs(raw_rgb, raw_sigma,
                                                            z_vals, rays_directions)

        return rgb, depth, accumulated, entropy

    def raw2outputs(self, rgb, sigma, z_vals, rays_d):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        init_dists = torch.Tensor([1e10]).to(rgb.device).expand(dists[..., :1].shape)
        dists = torch.cat([dists, init_dists], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # sigma_loss = sigma_sparsity_loss(raw[...,3])
        alpha = 1. - torch.exp(-sigma * dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

        ones = torch.ones((alpha.shape[0], 1)).to(rgb.device)
        beta = torch.cat([ones, 1. - alpha + 1e-10], -1)
        weights = alpha * torch.cumprod(beta, -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
        acc_map = torch.sum(weights, -1)

        # white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

        # Calculate weights sparsity loss
        try:
            probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True) + 1e-6], dim=-1)
            entropy = Categorical(probs=probs).entropy()
        except:
            pdb.set_trace()

        return rgb_map, depth_map, acc_map, entropy