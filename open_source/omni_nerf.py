import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["load_nerf"]

omni_nerf_scene = "stanford_area_3"

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        if x.shape[-1] > self.input_ch + self.input_ch_views:
            input_pts, input_views, input_depth = torch.split(x, [self.input_ch, self.input_ch_views, 1], dim=-1)
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class NeRFGnt(NeRF):
    def __init__(self, **kwargs):
        """
        """
        super(NeRFGnt, self).__init__(**kwargs)
        if self.use_viewdirs:
            self.gradient_linear = nn.Linear(self.W // 2, 3)

    def forward(self, x):
        if x.shape[-1] > self.input_ch + self.input_ch_views:
            input_pts, input_views, input_depth = torch.split(x, [self.input_ch, self.input_ch_views, 1], dim=-1)
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            gradient = self.gradient_linear(h)
            outputs = torch.cat([rgb, alpha, gradient], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


def batchify(fn, chunk=1024 * 32):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def to_camera_frame(pts):
    """
    pts: tensor in shape of [points, 3]
    """
    global omni_nerf_scene
    if omni_nerf_scene == "stanford_area_1":
        w2c = torch.tensor([
            [0.9973723292350769, 0.07230910658836365, -0.004451838321983814, -0.18107201159000397],
            [-0.003514759475365281, -0.013081424869596958, -0.9999082684516907, 1.5389373302459717],
            [-0.07236070930957794, 0.9972964525222778, -0.012792902067303658, -10.814047813415527],
            [0, 0, 0, 1],
        ])
    elif omni_nerf_scene == "stanford_area_2":
        w2c = torch.tensor([
            [0, 0, 0, 1],
            [0.14232482016086578, -0.989805281162262, -0.005397543776780367, 18.769657135009766],
            [-0.01314542070031166, 0.0035624413285404444, -0.9999072551727295, 1.358992338180542],
            [0.9897326827049255, 0.14238257706165314, -0.01250438392162323, -4.544514179229736],
        ])
    elif omni_nerf_scene == "stanford_area_3":
        w2c = torch.tensor([
            [0.33813756704330444, -0.9410862326622009, 0.004434713162481785, -5.711871147155762],
            [0.006247888319194317, -0.0024673265870660543, -0.9999774098396301, 1.36056649684906],
            [0.9410759210586548, 0.33815765380859375, 0.005045505706220865, -2.4272472858428955],
            [0, 0, 0, 1],
        ])
    elif omni_nerf_scene == "stanford_area_4":
        w2c = torch.tensor([
            [0, 0, 0, 1],
            [0.9344097375869751, 0.3561612665653229, 0.005258799996227026, -8.624170303344727],
            [0.006873208098113537, -0.0032674663234502077, -0.9999710321426392, 1.2457789182662964],
            [-0.35613375902175903, 0.934418797492981, -0.005501122679561377, -3.1593074798583984],
        ])
    elif omni_nerf_scene == "stanford_area_5a":
        w2c = torch.tensor([
            [0, 0, 0, 1],
            [-0.29167312383651733, 0.9564557671546936, 0.01091797649860382, -17.1992130279541],
            [0.0021006062161177397, 0.012054765596985817, -0.9999251365661621, 1.1426421403884888],
            [-0.9565157294273376, -0.291628360748291, -0.0055251880548894405, 8.328186988830566],
        ])
    elif omni_nerf_scene == "stanford_area_5b":
        w2c = torch.tensor([
            [0, 0, 0, 1],
            [-0.7952287793159485, 0.6062952876091003, -0.004145996179431677, -6.498640060424805],
            [0.007829352281987667, 0.0034311353228986263, -0.9999634623527527, 1.5605558156967163],
            [-0.6062589287757874, -0.7952321767807007, -0.007475437130779028, -13.57103157043457],
        ])
    elif omni_nerf_scene == "stanford_area_6":
        w2c = torch.tensor([
            [0, 0, 0, 1],
            [0.178488627076149, -0.9839391708374023, -0.002350610913708806, 14.007564544677734],
            [0.0013128143036738038, 0.002627116860821843, -0.9999957084655762, 1.3671321868896484],
            [0.9839410781860352, 0.17848478257656097, 0.0017606399487704039, -1.0646774768829346],
        ])
    else:
        raise NotImplementedError()
    ori_shape = pts.shape
    pts = pts.reshape((-1, 3)).T
    return (w2c @ torch.cat([pts, torch.ones_like(pts)[:1, :]], dim=0))[:3, :].T.reshape(ori_shape)


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    Prepares inputs and applies network 'fn'.
    inputs: tensor in shape of [N_rays, N_sample_points, 3]
    viewdirs: tensor in shape of [N_rays, 3]
    """
    import torch.nn.functional as F
    assert torch.allclose(viewdirs, F.normalize(inputs[:, 1] - inputs[:, 0], p=2, dim=-1), atol=1e-3, rtol=1e-3)
    inputs = to_camera_frame(inputs)
    viewdirs = F.normalize(inputs[:, 1] - inputs[:, 0], p=2, dim=-1)

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def load_nerf(args, device):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if not args.use_gradient:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    else:
        model = NeRFGnt(D=args.netdepth, W=args.netwidth,
                        input_ch=input_ch, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    model_fine = None
    if args.N_importance > 0:
        if not args.use_gradient:
            model_fine = NeRF(D=args.netdepth, W=args.netwidth,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRFGnt(D=args.netdepth, W=args.netwidth,
                                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    ##########################

    # Load checkpoints
    ckpt_path = args.ckpt_path
    print('Loading from', ckpt_path)
    ckpt = torch.load(ckpt_path)

    # Load model
    model.load_state_dict(ckpt['network_fn_state_dict'])
    if model_fine is not None:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'pertub': args.pertub,
        'white_bkgd': args.white_bkgd,
        'network_query_fn': network_query_fn,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'raw_noise_std': args.raw_noise_std,
    }

    return render_kwargs_train
