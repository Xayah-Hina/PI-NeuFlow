from .encoder import get_encoder
import torch
import typing
import types


class NeRFSmall(torch.nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 dtype=torch.float32,
                 ):
        super(NeRFSmall, self).__init__()
        self.encoder_xyzt = get_encoder('hyfluid')

        self.input_ch = self.encoder_xyzt.num_levels * self.encoder_xyzt.features_per_level
        self.rgb = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))

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
                out_dim = 1  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False, dtype=dtype))

        self.sigma_net = torch.nn.ModuleList(sigma_net)

        self.color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = 1
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 1
            else:
                out_dim = hidden_dim_color

            self.color_net.append(torch.nn.Linear(in_dim, out_dim, bias=True, dtype=dtype))

    def forward(self, x):
        h = x[..., :self.input_ch]
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, inplace=True)

        sigma = h
        return sigma

    def get_params(self, learning_rate_encoder, learning_rate_network):

        params = [
            {'params': self.encoder_xyzt.parameters(), 'lr': learning_rate_encoder, 'eps': 1e-15},
            {'params': self.sigma_net.parameters(), 'lr': learning_rate_network, 'weight_decay': 1e-6},
            {'params': self.rgb, 'lr': learning_rate_network, 'eps': 1e-15},
        ]

        return params


class _trunc_exp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


class NetworkPINeuFlow(torch.nn.Module):
    def __init__(self,
                 encoding_xyzt: typing.Literal['hyfluid'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash', 'hyfluid'],
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
        enc_xyzt = self.encoder_xyzt(xyzt)
        enc_dirs = self.encoder_dir(dirs)

        # sigma
        h = enc_xyzt
        for l in range(self.states.num_layers_sigma):
            h = self.sigma_net[l](h)
            if l != self.states.num_layers_sigma - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        h = torch.cat([enc_dirs, geo_feat], dim=-1)
        for l in range(self.states.num_layers_color):
            h = self.color_net[l](h)
            if l != self.states.num_layers_color - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs

    def get_params(self, learning_rate_encoder, learning_rate_network):

        params = [
            {'params': self.encoder_xyzt.parameters(), 'lr': learning_rate_encoder, 'eps': 1e-15},
            {'params': self.sigma_net.parameters(), 'lr': learning_rate_network, 'weight_decay': 1e-6},
            {'params': self.encoder_dir.parameters(), 'lr': learning_rate_encoder, 'eps': 1e-15},
            {'params': self.color_net.parameters(), 'lr': learning_rate_network, 'weight_decay': 1e-6},
        ]

        return params
