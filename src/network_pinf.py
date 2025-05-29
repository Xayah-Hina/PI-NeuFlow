from .frustum import *
import torch
import typing
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
                width = (6.0 / in_features) ** 0.5 / self.omega_0
            self.linear.weight.uniform_(-width, width)

    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


class SIREN_NeRFt(torch.nn.Module):
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

    def query_density_and_feature(self, input_pts: torch.Tensor, cond: torch.Tensor):
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

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        return self.query_density_and_feature(x, cond)[0]

    def forward(self, x, dirs, cond: torch.Tensor = None):
        sigma, h = self.query_density_and_feature(x, cond)

        if self.use_viewdirs:
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(dirs)

            h = torch.cat([input_pts_feature, input_views_feature], -1)

            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        # outputs = torch.cat([rgb, sigma], -1)

        return torch.sigmoid(rgb), sigma, {}


def weighted_sum_of_samples(wei_list: list[torch.Tensor], content: list[torch.Tensor] | torch.Tensor | None):
    if isinstance(content, list):  # list of [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * ct, dim=-2) for weights, ct in zip(wei_list, content)]

    elif content is not None:  # [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * content, dim=-2) for weights in wei_list]

    return [torch.sum(weights, dim=-1) for weights in wei_list]


class NeRFOutputs:
    def __init__(self, rgb_map: torch.Tensor, depth_map: torch.Tensor | None, acc_map: torch.Tensor, **kwargs):
        """
        Args:
            rgb_map: [n_rays, 3]. Estimated RGB color of a ray.
            depth_map: [n_rays]. Depth map. Optional.
            acc_map: [n_rays]. Sum of weights along each ray.
        """
        self.rgb = rgb_map
        self.depth = depth_map
        self.acc = acc_map
        self.extras = kwargs

    def as_tuple(self):
        return self.rgb, self.depth, self.acc, self.extras

    @staticmethod
    def merge(outputs: list["NeRFOutputs"], shape=None, skip_extras=False) -> "NeRFOutputs":
        """Merge list of outputs into one
        Args:
            outputs: Outputs from different batches.
            shape: If not none, reshape merged outputs' first dimension
            skip_extras: Ignore extras when merging, used for merging coarse outputs
        """
        if len(outputs) == 1:  # when training
            return outputs[0]
        extras = {}
        if not skip_extras:
            keys = outputs[0].extras.keys()  # all extras must have same keys
            extras = {k: [] for k in keys}
            for output in outputs:
                for k in keys:
                    extras[k].append(output.extras[k])
            for k in extras:
                assert isinstance(extras[k][0], (torch.Tensor, NeRFOutputs)), \
                    "All extras must be either torch.Tensor or NeRFOutputs when merging"
                if isinstance(extras[k][0], NeRFOutputs):
                    extras[k] = NeRFOutputs.merge(extras[k], shape)  # recursive merging
                elif extras[k][0].dim() == 0:
                    extras[k] = torch.tensor(extras[k]).mean()  # scalar value, reduce to avg
                else:
                    extras[k] = torch.cat(extras[k])

        ret = NeRFOutputs(
            torch.cat([out.rgb for out in outputs]),
            torch.cat([out.depth for out in outputs]) if outputs[0].depth is not None else None,
            torch.cat([out.acc for out in outputs]),
            **extras
        )
        if shape is not None:
            ret.rgb = ret.rgb.reshape(*shape, 3)
            ret.depth = ret.depth.reshape(shape) if ret.depth is not None else None
            ret.acc = ret.acc.reshape(shape)
            for k in ret.extras:
                if isinstance(ret.extras[k], torch.Tensor) and ret.extras[k].dim() > 0:
                    ret.extras[k] = torch.reshape(ret.extras[k], [*shape, *ret.extras[k].shape[1:]])
        return ret

    def add_background(self, background: torch.Tensor):
        """Add background to rgb output
        Args:
            background: scalar or image
        """
        self.rgb = self.rgb + background * (1.0 - self.acc[..., None])
        for v in self.extras.values():
            if isinstance(v, NeRFOutputs):
                v.add_background(background)


def raw2outputs(raw, z_vals, rays_d, mask=None, cos_anneal_ratio=1.0) -> tuple[NeRFOutputs, torch.Tensor]:
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: returned result of RadianceField: rgb, sigma, extra. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        mask: [num_rays, num_samples]. aabb masking
        cos_anneal_ratio: float.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [n_rays, n_samples]
    dists = torch.cat([dists, dists[..., -1:]], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    def sigma2alpha(sigma: torch.Tensor):  # [n_rays, n_samples, 1] -> [n_rays, n_samples, 1]
        if mask is not None:
            sigma = sigma * mask
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # [n_rays, n_samples]
        return alpha

    extra: dict = raw[2]
    gradients = None
    if 'sdf' in extra:
        raise NotImplementedError("SDF is not supported in raw2outputs")

    elif 'sigma_s' in extra:
        alpha_list = [sigma2alpha(extra['sigma_d']), sigma2alpha(extra['sigma_s'])]
        color_list = [extra['rgb_d'], extra['rgb_s']]

    else:
        # shortcut for single model
        alpha = sigma2alpha(raw[1])
        rgb = raw[0]
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]
        # depth_map = torch.sum(weights * z_vals, -1)
        depth_map = None  # unused
        acc_map = torch.sum(weights, -1)
        return NeRFOutputs(rgb_map, depth_map, acc_map), weights

    for key in 'rgb_s', 'rgb_d', 'dynamic':
        extra.pop(key, None)

    dens = 1.0 - torch.stack(alpha_list, dim=-1)  # [n_rays, n_samples, 2]
    dens = torch.cat([dens, torch.prod(dens, dim=-1, keepdim=True)], dim=-1) + 1e-9  # [n_rays, n_samples, 3]
    Ti_all = torch.cumprod(dens, dim=-2) / dens  # [n_rays, n_samples, 3], accu along samples, exclusive
    weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # a list of [n_rays, n_samples]

    rgb_map = sum(weighted_sum_of_samples(weights_list, color_list))  # [n_rays, 3]
    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = sum(weighted_sum_of_samples(weights_list, None))  # [n_rays]

    # Estimated depth map is expected distance.
    # Disparity map is inverse depth.
    # depth_map = sum(weighted_sum_of_samples(weights_list, z_vals[..., None]))  # [n_rays]
    depth_map = None
    # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    # alpha * Ti
    weights = weights_list[0]  # [n_rays, n_samples]

    if len(alpha_list) > 1:  # hybrid model
        self_weights_list = [alpha_list[alpha_i] * Ti_all[..., alpha_i] for alpha_i in
                             range(len(alpha_list))]  # a list of [n_rays, n_samples]
        rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
        acc_map_stack = weighted_sum_of_samples(self_weights_list, None)

        if gradients is not None:
            extra['grad_map'] = weighted_sum_of_samples(self_weights_list[1:], gradients)[0]

        # assume len(alpha_list) == 2 for hybrid model
        extra['dynamic'] = NeRFOutputs(rgb_map_stack[0], None, acc_map_stack[0])
        extra['static'] = NeRFOutputs(rgb_map_stack[1], None, acc_map_stack[1])

    return NeRFOutputs(rgb_map, depth_map, acc_map, **extra), weights


class NetworkPINF(torch.nn.Module):
    def __init__(self, netdepth, netwidth, input_ch, use_viewdirs, omega, use_first_omega, fading_layers, background_color: typing.Literal['white', 'black']):
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
        self.background_color = torch.tensor([1.0, 1.0, 1.0]) if background_color == 'white' else torch.tensor([0.0, 0.0, 0.0])

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
        bbox_mask = torch.ones_like(points_with_time_flat[..., 0], dtype=torch.bool)  # [N * num_samples, 1]
        # bbox_mask = inside_mask(points_with_time_flat[..., :3], extra_params.s_w2s, extra_params.s_scale, extra_params.s_min, extra_params.s_max, to_float=False)  # [N * num_samples]
        # if not torch.any(bbox_mask):
        #     return torch.zeros_like(rays_o)
        points_time_flat_filtered = points_with_time_flat[bbox_mask]  # [filtered / N * num_samples, 4]
        rays_d_flat = rays_d[:, None, :].expand(N, num_samples, 3).reshape(-1, 3)  # [N * num_samples, 3]
        rays_d_flat_filtered = rays_d_flat[bbox_mask]  # [filtered / N * num_samples, 3]

        raw_output = self(points_time_flat_filtered, rays_d_flat_filtered)  # [filtered / N * num_samples,], [filtered / N * num_samples, 3]

##########
        sigma_d_filtered = raw_output['sigma_d']  # [filtered / N * num_samples,]
        sigma_s_filtered = raw_output['sigma_s']  # [filtered / N * num_samples,]
        rgb_d_filtered = raw_output['rgb_d']  # [filtered / N * num_samples, 3]
        rgb_s_filtered = raw_output['rgb_s']  # [filtered / N * num_samples, 3]
        sigma_d = torch.zeros([N * num_samples], device=device, dtype=dtype).masked_scatter(bbox_mask, sigma_d_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        sigma_s = torch.zeros([N * num_samples], device=device, dtype=dtype).masked_scatter(bbox_mask, sigma_s_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        rgb_d = torch.zeros([N * num_samples, 3], device=device, dtype=dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_d_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]
        rgb_s = torch.zeros([N * num_samples, 3], device=device, dtype=dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_s_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]


        sigma = sigma_s + sigma_d
        rgb = (rgb_s * sigma_s + rgb_d * sigma_d) / (sigma + 1e-6)
        raw = [
            rgb.reshape(N, num_samples, 3),
            sigma.reshape(N, num_samples, 1),
            {
                'rgb_s': rgb_s.reshape(N, num_samples, 3),
                'rgb_d': rgb_d.reshape(N, num_samples, 3),
                'sigma_s': sigma_s.reshape(N, num_samples, 1),
                'sigma_d': sigma_d.reshape(N, num_samples, 1),
            }
        ]
        output, _1 = raw2outputs(raw, z_vals.reshape(N, num_samples), rays_d, bbox_mask.reshape(N, num_samples, 1))
##########
        # sigma_d_filtered = raw_output['sigma_d']  # [filtered / N * num_samples,]
        # sigma_s_filtered = raw_output['sigma_s']  # [filtered / N * num_samples,]
        # rgb_d_filtered = raw_output['rgb_d']  # [filtered / N * num_samples, 3]
        # rgb_s_filtered = raw_output['rgb_s']  # [filtered / N * num_samples, 3]
        # sigma_d = torch.zeros([N * num_samples], device=device, dtype=dtype).masked_scatter(bbox_mask, sigma_d_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        # sigma_s = torch.zeros([N * num_samples], device=device, dtype=dtype).masked_scatter(bbox_mask, sigma_s_filtered).reshape(N, num_samples, 1)  # [N, num_samples, 1]
        # rgb_d = torch.zeros([N * num_samples, 3], device=device, dtype=dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_d_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]
        # rgb_s = torch.zeros([N * num_samples, 3], device=device, dtype=dtype).masked_scatter(bbox_mask.unsqueeze(-1), rgb_s_filtered).reshape(N, num_samples, 3)  # [N, num_samples, 3]
        #
        # noise = 0.
        # alpha_d = 1. - torch.exp(-torch.nn.functional.relu(sigma_d[..., -1] + noise) * dists_final)  # [N, num_samples]
        # alpha_s = 1. - torch.exp(-torch.nn.functional.relu(sigma_s[..., -1] + noise) * dists_final)  # [N, num_samples]
        #
        # alpha_list = [alpha_d, alpha_s]
        # color_list = [rgb_d, rgb_s]

        # dens = 1.0 - torch.stack(alpha_list, dim=-1)  # [n_rays, n_samples, 2]
        # dens = torch.cat([dens, torch.prod(dens, dim=-1, keepdim=True)], dim=-1) + 1e-9  # [n_rays, n_samples, 3]
        # Ti_all = torch.cumprod(dens, dim=-2) / dens  # [n_rays, n_samples, 3], accu along samples, exclusive
        # weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # a list of [n_rays, n_samples]
        #
        # rgb_map = sum(weighted_sum_of_samples(weights_list, color_list))  # [n_rays, 3]
        # # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        # acc_map = sum(weighted_sum_of_samples(weights_list, None))  # [n_rays]
        # rgb_map = rgb_map + self.background_color.to(device).to(dtype) * (1.0 - acc_map[..., None])
        #
        # depth_map = sum(weighted_sum_of_samples(weights_list, z_vals[..., None]))  # [n_rays]
        #
        # self_weights_list = [alpha_list[alpha_i] * Ti_all[..., alpha_i] for alpha_i in range(len(alpha_list))]  # a list of [n_rays, n_samples]
        # rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
        # acc_map_stack = weighted_sum_of_samples(self_weights_list, None)
        # extras = {
        #     'acc_d': acc_map_stack[0],
        #     'acc_s': acc_map_stack[1],
        #     'rgb_d': rgb_map_stack[0] + self.background_color.to(device).to(dtype) * (1.0 - acc_map_stack[0][..., None]),
        #     'rgb_s': rgb_map_stack[1] + self.background_color.to(device).to(dtype) * (1.0 - acc_map_stack[1][..., None]),
        # }

        rgb_map = output.rgb
        depth_map = output.depth
        extras = output.extras
        return rgb_map, depth_map, extras

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
            rgb_map, depth_map, extras = self.render(rays_o, rays_d, time, extra_params, randomize)
        return rgb_map, depth_map, extras
