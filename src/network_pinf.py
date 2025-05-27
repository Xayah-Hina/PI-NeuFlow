from .frustum import *
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
            step_ratio = torch.clamp(
                torch.tensor(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), device=h.device),
                0.0, 1.0
            )
            ma = 1 + (self.D - 2) * step_ratio
            m = torch.arange(self.D, dtype=h.float32, device=h.device)
            weights = torch.clamp(1 + ma - m, 0, 1) * torch.clamp(1 + m - ma, 0, 1)

            h = 0
            for w, y in zip(weights, h_layers):
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

        self.fading_step += 1

        return torch.sigmoid(rgb), sigma, {}


def weighted_sum_of_samples(wei_list: list[torch.Tensor], content: list[torch.Tensor] | torch.Tensor | None):
    if isinstance(content, list):  # list of [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * ct, dim=-2) for weights, ct in zip(wei_list, content)]

    elif content is not None:  # [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * content, dim=-2) for weights in wei_list]

    return [torch.sum(weights, dim=-1) for weights in wei_list]


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
        dtype = rays_d.dtype

        points_with_time, dist_vals, z_vals = assemble_points_with_time(
            batch_rays_o=rays_o,
            batch_rays_d=rays_d,
            time=time,
            num_samples=num_samples,
            near=extra_params.nears[0],
            far=extra_params.fars[0],
            randomize=randomize,
        )
        dists_cat = torch.cat([dist_vals, dist_vals[..., -1:]], -1)  # [N, N_depths]
        dists_final = dists_cat * torch.norm(rays_d[..., None, :], dim=-1)  # [N, N_depths]

        points_with_time_flat = points_with_time.reshape(-1, 4)  # [N * num_samples, 4]
        bbox_mask = inside_mask(points_with_time_flat[..., :3], extra_params.s_w2s, extra_params.s_scale, extra_params.s_min, extra_params.s_max, to_float=False)  # [N * num_samples]
        if not torch.any(bbox_mask):
            return torch.zeros_like(rays_o)
        points_time_flat_filtered = points_with_time_flat[bbox_mask]  # [filtered / N * num_samples, 4]
        rays_d_flat = rays_d[:, None, :].expand(N, num_samples, 3).reshape(-1, 3)  # [N * num_samples, 3]
        rays_d_flat_filtered = rays_d_flat[bbox_mask]  # [filtered / N * num_samples, 3]

        raw_output = self(points_time_flat_filtered, rays_d_flat_filtered)  # [filtered / N * num_samples,], [filtered / N * num_samples, 3]
        sigma_d_filtered = raw_output['sigma_d']  # [filtered / N * num_samples,]
        sigma_s_filtered = raw_output['sigma_s']  # [filtered / N * num_samples,]
        rgb_d_filtered = raw_output['rgb_d']  # [filtered / N * num_samples, 3]
        rgb_s_filtered = raw_output['rgb_s']  # [filtered / N * num_samples, 3]
        sigma_d = torch.zeros([N * num_samples], device=device, dtype=dtype).masked_scatter(bbox_mask, sigma_d_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        sigma_s = torch.zeros([N * num_samples], device=device, dtype=dtype).masked_scatter(bbox_mask, sigma_s_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        rgb_d = torch.zeros([N * num_samples, 3], device=device, dtype=dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_d_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]
        rgb_s = torch.zeros([N * num_samples, 3], device=device, dtype=dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_s_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]

        noise = 0.
        alpha_d = 1. - torch.exp(-torch.nn.functional.relu(sigma_d[..., -1] + noise) * dists_final)  # [N, num_samples]
        alpha_s = 1. - torch.exp(-torch.nn.functional.relu(sigma_s[..., -1] + noise) * dists_final)  # [N, num_samples]

        alpha_list = [alpha_d, alpha_s]
        color_list = [rgb_d, rgb_s]

        dens = 1.0 - torch.stack(alpha_list, dim=-1)  # [n_rays, n_samples, 2]
        dens = torch.cat([dens, torch.prod(dens, dim=-1, keepdim=True)], dim=-1) + 1e-9  # [n_rays, n_samples, 3]
        Ti_all = torch.cumprod(dens, dim=-2) / dens  # [n_rays, n_samples, 3], accu along samples, exclusive
        weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # a list of [n_rays, n_samples]

        rgb_map = sum(weighted_sum_of_samples(weights_list, color_list))  # [n_rays, 3]
        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = sum(weighted_sum_of_samples(weights_list, None))  # [n_rays]

        depth_map = sum(weighted_sum_of_samples(weights_list, z_vals[..., None]))  # [n_rays]

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
