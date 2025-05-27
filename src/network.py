from .encoder import get_encoder
from .frustum import *
import torch
import typing
import types
import tinycudann as tcnn


class NetworkPINeuFlowDynamics(torch.nn.Module):
    def __init__(self,
                 encoding_xyzt: typing.Literal['hyfluid'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash', 'hyfluid'],
                 use_tcnn: bool,
                 num_layers_sigma,
                 num_layers_color,
                 hidden_dim_sigma,
                 hidden_dim_color,
                 geo_feat_dim,
                 ):
        super().__init__()
        # self.encoder_xyzt
        self.encoder_xyzt = get_encoder(encoding_xyzt)

        # self.encoder_dir
        self.encoder_dir = get_encoder(encoding_dir)

        # self.sigma_net
        if use_tcnn:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.encoder_xyzt.num_levels * self.encoder_xyzt.features_per_level,
                n_output_dims=1 + geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_sigma,
                    "n_hidden_layers": num_layers_sigma - 1,
                },
            )
        else:
            sigma_net = []
            for l in range(num_layers_sigma):
                if l == 0:
                    in_dim = self.encoder_xyzt.num_levels * self.encoder_xyzt.features_per_level
                else:
                    in_dim = hidden_dim_sigma
                if l == num_layers_sigma - 1:
                    out_dim = 1 + geo_feat_dim  # SB sigma + features for color
                else:
                    out_dim = hidden_dim_sigma
                sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
            self.sigma_net = torch.nn.ModuleList(sigma_net)

        # self.color_net
        if use_tcnn:
            self.color_net = tcnn.Network(
                n_input_dims=self.encoder_dir.output_dim + geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )
        else:
            color_net = []
            for l in range(num_layers_color):
                if l == 0:
                    in_dim = self.encoder_dir.output_dim + geo_feat_dim
                else:
                    in_dim = hidden_dim_color
                if l == num_layers_color - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_color
                color_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
            self.color_net = torch.nn.ModuleList(color_net)

        # self.states
        self.states = types.SimpleNamespace()
        self.states.use_tcnn = use_tcnn
        self.states.num_layers_sigma = num_layers_sigma
        self.states.num_layers_color = num_layers_color

    def forward(self, xyzt, dirs):
        # sigma
        enc_xyzt = self.encoder_xyzt(xyzt)
        if self.states.use_tcnn:
            h = self.sigma_net(enc_xyzt)
        else:
            h = enc_xyzt
            for l in range(self.states.num_layers_sigma):
                h = self.sigma_net[l](h)
                if l != self.states.num_layers_sigma - 1:
                    h = torch.nn.functional.relu(h, inplace=True)
        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        # color
        enc_dirs = self.encoder_dir(dirs)
        h = torch.cat([enc_dirs, geo_feat], dim=-1)
        if self.states.use_tcnn:
            h = self.color_net(h)
        else:
            for l in range(self.states.num_layers_color):
                h = self.color_net[l](h)
                if l != self.states.num_layers_color - 1:
                    h = torch.nn.functional.relu(h, inplace=True)
        # sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return sigma, rgb

    def sigma(self, xyzt):
        enc_xyzt = self.encoder_xyzt(xyzt)

        # sigma
        if self.states.use_tcnn:
            h = self.sigma_net(enc_xyzt)
        else:
            h = enc_xyzt
            for l in range(self.states.num_layers_sigma):
                h = self.sigma_net[l](h)
                if l != self.states.num_layers_sigma - 1:
                    h = torch.nn.functional.relu(h, inplace=True)
        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def get_params(self, learning_rate_encoder, learning_rate_network):

        params = [
            {'params': self.encoder_xyzt.parameters(), 'lr': learning_rate_encoder, 'eps': 1e-15},
            {'params': self.sigma_net.parameters(), 'lr': learning_rate_network, 'weight_decay': 1e-6},
            {'params': self.encoder_dir.parameters(), 'lr': learning_rate_encoder, 'eps': 1e-15},
            {'params': self.color_net.parameters(), 'lr': learning_rate_network, 'weight_decay': 1e-6},
        ]

        return params


class NetWorkPINeuFlow(torch.nn.Module):
    def __init__(self,
                 encoding_xyzt: typing.Literal['hyfluid'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash', 'hyfluid'],
                 background_color: typing.Literal['white', 'black'],
                 use_tcnn: bool,
                 num_layers_sigma,
                 num_layers_color,
                 hidden_dim_sigma,
                 hidden_dim_color,
                 geo_feat_dim,
                 ):
        super().__init__()
        self.dynamic_model = NetworkPINeuFlowDynamics(
            encoding_xyzt=encoding_xyzt,
            encoding_dir=encoding_dir,
            use_tcnn=use_tcnn,
            num_layers_sigma=num_layers_sigma,
            num_layers_color=num_layers_color,
            hidden_dim_sigma=hidden_dim_sigma,
            hidden_dim_color=hidden_dim_color,
            geo_feat_dim=geo_feat_dim,
        )
        self.background_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) if background_color == 'white' else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    def forward(self, xyzt, dirs):
        sigma_d, rgb_d = self.dynamic_model(xyzt, dirs)
        return {
            'sigma_d': sigma_d,
            'rgb_d': rgb_d,
        }

    def get_params(self, learning_rate_static, learning_rate_dynamic):
        params = self.dynamic_model.get_params(learning_rate_static, learning_rate_dynamic)
        return params

    def render(self, rays_o, rays_d, time, extra_params, randomize):
        """
        :param rays_o: [N, 3]
        :param rays_d: [N, 3]
        :param time: [1]
        :param extra_params: ExtraParams
        :param randomize: bool
        :return:
            rgb_map: [N, 3]
        """
        N = rays_d.shape[0]
        num_samples = 192
        device = rays_d.device

        points_with_time, dist_vals, z_vals = assemble_points_with_time(
            batch_rays_o=rays_o,
            batch_rays_d=rays_d,
            time=time,
            num_samples=num_samples,
            near=extra_params.nears[0],
            far=extra_params.fars[0],
            randomize=randomize,
        )
        points_with_time_flat = points_with_time.reshape(-1, 4)  # [N * num_samples, 4]
        bbox_mask = inside_mask(points_with_time_flat[..., :3], extra_params.s_w2s, extra_params.s_scale, extra_params.s_min, extra_params.s_max, to_float=False)  # [N * num_samples]
        if not torch.any(bbox_mask):
            return torch.zeros_like(rays_o)
        points_time_flat_filtered = points_with_time_flat[bbox_mask]  # [filtered / N * num_samples, 4]
        rays_d_flat = rays_d[:, None, :].expand(N, num_samples, 3).reshape(-1, 3)  # [N * num_samples, 3]
        rays_d_flat_filtered = rays_d_flat[bbox_mask]  # [filtered / N * num_samples, 3]

        raw_output = self(points_time_flat_filtered, rays_d_flat_filtered)  # [filtered / N * num_samples,], [filtered / N * num_samples, 3]
        sigma_filtered = raw_output['sigma_d']  # [filtered / N * num_samples,]
        rgb_filtered = raw_output['rgb_d']  # [filtered / N * num_samples, 3]
        sigma = torch.zeros([N * num_samples], device=device, dtype=sigma_filtered.dtype).masked_scatter(bbox_mask, sigma_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        rgb = torch.zeros([N * num_samples, 3], device=device, dtype=rgb_filtered.dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]

        dists_cat = torch.cat([dist_vals, dist_vals[..., -1:]], -1)  # [N, N_depths]
        dists_final = dists_cat * torch.norm(rays_d[..., None, :], dim=-1)  # [N, N_depths]

        noise = 0.
        alpha = 1. - torch.exp(-torch.nn.functional.relu(sigma[..., -1] + noise) * dists_final)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]

        acc_map = torch.sum(weights, dim=-1)  # 每条光线的总权重（可能 < 1）
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) + (1. - acc_map)[..., None] * self.background_color

        depth_map = torch.sum(weights * z_vals, dim=-1)  # 每条光线的深度
        return rgb_map, depth_map

    @torch.no_grad()
    def render_no_grad(self, rays_o, rays_d, time, extra_params, randomize):
        """
        :param rays_o: [N, 3]
        :param rays_d: [N, 3]
        :param time: [1]
        :param extra_params: ExtraParams
        :param randomize: bool
        :return:
            rgb_map: [N, 3]
        """
        with torch.no_grad():
            rgb_map, depth_map = self.render(rays_o, rays_d, time, extra_params, randomize)
        return rgb_map, depth_map
