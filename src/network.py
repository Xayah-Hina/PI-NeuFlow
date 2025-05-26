from .encoder import get_encoder
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
        rgbs = torch.sigmoid(h)

        return sigma, rgbs

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
