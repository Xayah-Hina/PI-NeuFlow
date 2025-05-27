import torch
import numpy as np


class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                width = 1 / in_features
            else:
                width = np.sqrt(6 / in_features) / self.omega_0
            self.linear.weight.uniform_(-width, width)

    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


class SIREN_NeRFt(torch.nn.Module):  # Alias for SIREN_NeRF_t, used in original NeRF paper
    def __init__(self, D=8, W=256, input_ch=4, skips=(4,), use_viewdirs=False, first_omega_0=30.0, unique_first=False, fading_fin_step=0):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = 3 if use_viewdirs else 0
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0

        hidden_omega_0 = 1.0

        self.pts_linears = torch.nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0, is_first=unique_first)] +
            [SineLayer(W, W, omega_0=hidden_omega_0)
             if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D - 1)]
        )

        self.sigma_linear = torch.nn.Linear(W, 1)

        if use_viewdirs:
            self.views_linear = SineLayer(3, W // 2, omega_0=first_omega_0)
            self.feature_linear = SineLayer(W, W // 2, omega_0=hidden_omega_0)
            self.feature_view_linears = torch.nn.ModuleList([SineLayer(W, W, omega_0=hidden_omega_0)])

        self.rgb_linear = torch.nn.Linear(W, 3)

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - radiance_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def query_density_and_feature(self, input_pts: torch.Tensor):
        h = input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        sigma = self.sigma_linear(h)
        return torch.nn.functional.relu(sigma), h

    def query_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_density_and_feature(x)[0]

    def forward(self, x, dirs):
        sigma, h = self.query_density_and_feature(x)

        if self.use_viewdirs:
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(dirs)

            h = torch.cat([input_pts_feature, input_views_feature], -1)

            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        # outputs = torch.cat([rgb, sigma], -1)

        return torch.sigmoid(rgb), sigma, {}


class NetworkPINF(torch.nn.Module):
    def __init__(self, netdepth, netwidth, input_ch, use_viewdirs, omega, use_first_omega, fading_layers):
        super().__init__()
        self.static_model = SIREN_NeRFt(
            D=netdepth,
            W=netwidth,
            input_ch=input_ch - 1,
            use_viewdirs=use_viewdirs,
            first_omega_0=omega,
            unique_first=use_first_omega,
            fading_fin_step=fading_layers
        )
        self.dynamic_model = SIREN_NeRFt(
            D=netdepth,
            W=netwidth,
            input_ch=input_ch,
            use_viewdirs=use_viewdirs,
            first_omega_0=omega,
            unique_first=use_first_omega,
            fading_fin_step=fading_layers
        )

    def forward(self, x, dirs):
        rgb_s, sigma_s, extra_s = self.static_model.forward(x[..., :3], dirs)
        rgb_d, sigma_d, extra_d = self.dynamic_model.forward(x, dirs)
        return {
            'rgb_s': rgb_s,
            'sigma_s': sigma_s,
            'extra_s': extra_s,
            'rgb_d': rgb_d,
            'sigma_d': sigma_d,
            'extra_d': extra_d
        }

    def get_params(self, learning_rate_static, learning_rate_dynamic):
        params = [
            {'params': self.static_model.parameters(), 'lr': learning_rate_static, 'eps': 1e-15},
            {'params': self.dynamic_model.parameters(), 'lr': learning_rate_dynamic, 'eps': 1e-15},
        ]
        return params
