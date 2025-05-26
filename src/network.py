from .encoder import get_encoder
from .cuda_extensions.ffmlp import FFMLP
import torch
import typing
import types


class NetworkPINeuFlow(torch.nn.Module):
    def __init__(self,
                 encoding_xyzt: typing.Literal['hyfluid'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash', 'hyfluid'],
                 use_ffmlp: bool,
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
        if use_ffmlp:
            self.sigma_net = FFMLP(
                input_dim=self.encoder_xyzt.num_levels * self.encoder_xyzt.features_per_level,
                output_dim=1 + geo_feat_dim,
                hidden_dim=hidden_dim_sigma,
                num_layers=num_layers_sigma,
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
        if use_ffmlp:
            self.color_net = FFMLP(
                input_dim=self.encoder_dir.output_dim + geo_feat_dim,
                output_dim=3,
                hidden_dim=hidden_dim_color,
                num_layers=num_layers_color,
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
        self.states.num_layers_sigma = num_layers_sigma
        self.states.num_layers_color = num_layers_color

    def forward(self, xyzt, dirs):
        # sigma
        enc_xyzt = self.encoder_xyzt(xyzt)
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
        for l in range(self.states.num_layers_color):
            h = self.color_net[l](h)
            if l != self.states.num_layers_color - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs

    def sigma(self, xyzt):
        enc_xyzt = self.encoder_xyzt(xyzt)

        # sigma
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
