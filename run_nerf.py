import os
import time

import imageio
import torch
from tqdm import tqdm, trange

from load_st3d import load_st3d_data
from run_nerf_helpers import *
import utils.backend as backend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(fn, chunk=1024 * 32):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
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


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        """render a chunk rays each time
        """
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def activate_density(dens):
    return F.softplus(dens + np.log(1 / (1 - 0.95) - 1))


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
        grad_map: [num_rays]. Predicted gradietn of a ray.
    """
    # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-activate_density(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10], -1), -1)[
                      :, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    if raw.shape[-1] == 7:
        grad = torch.tanh(raw[..., 4:])  # [N_rays, N_samples, 3]
        grad_map = torch.sum(weights[..., None] * grad, -2)  # [N_rays, 3]
    else:
        grad_map = torch.zeros(depth_map.shape)

    return rgb_map, depth_map, grad_map, weights, alpha


@torch.no_grad()
def render_path(rays, hw, render_kwargs, savedir=None, render_factor=0):
    H, W = hw
    rays_o, rays_d = rays
    if render_factor != 0:
        rays_o = rays_o.reshape(-1, H, W, 3)[:, ::render_factor, ::render_factor].reshape(-1, 3)
        rays_d = rays_d.reshape(-1, H, W, 3)[:, ::render_factor, ::render_factor].reshape(-1, 3)
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        assert rays_o.shape[0] % (H * W) == 0

    rgbs, deps = [], []
    t = time.time()
    batch = H * W

    print("render near, far = ", render_kwargs['near'], render_kwargs['far'])
    for i in tqdm(range(rays_o.shape[0] // batch)):
        print(i, time.time() - t)
        t = time.time()
        rgb, dep, _, _, _, _ = render(H, W, rays=[rays_o[i * batch:(i + 1) * batch], rays_d[i * batch:(i + 1) * batch]],
                                      pts_noise=0, **render_kwargs)
        rgb8 = rgb.reshape(H, W, 3).cpu().numpy()
        rgbs.append(rgb8)

    rgbs = np.stack(rgbs, 0)

    return rgbs, deps


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                lindisp=False,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                use_viewdirs=True,
                pts_noise=0.0001):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      use_viewdirs: whether to add viewdirs
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      depth_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    if use_viewdirs:
        viewdirs = rays_d
    else:
        viewdirs = None

    t_vals = torch.linspace(0., 1., steps=N_samples).to(near)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])

    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    # TODO: modify this
    # t_rand = torch.rand_like(z_vals)
    t_rand = torch.ones_like(z_vals) * 0.5
    z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, network_fn)

    rgb_map, depth_map, grad_map, weights, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    ret = {}
    if N_importance > 0:
        ret['rgb0'] = rgb_map
        ret['depth0'] = depth_map
        ret['grad0'] = grad_map
        ret['weights0'] = weights
        ret['z_vals0'] = z_vals

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=True, pytest=False)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        # raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, depth_map, grad_map, weights, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    ret.update({'rgb_map': rgb_map, 'depth_map': depth_map, 'grad_map': grad_map, "weights": weights, "z_vals": z_vals})

    return ret


def render(H, W, rays, near=0.0, far=2.0, far_replace=None, use_viewdirs=True, **kwargs):
    rays_o, rays_d = rays
    near = near * torch.ones_like(rays_d[..., :1])
    if far_replace is None:
        far = far * torch.ones_like(rays_d[..., :1])
    else:
        far = far_replace.reshape(rays_d[..., :1].shape).to(rays_d)

    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk=1024 * 32, use_viewdirs=use_viewdirs, **kwargs)
    # TODO: set a tag to control reset or not
    sh = rays_d.shape
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'grad_map', 'weights', 'z_vals']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args):
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

    grad_vars = list(model.parameters())

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
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'N_importance': args.N_importance,
        'network_fine': model_fine.cuda(),
        'N_samples': args.N_samples,
        'network_fn': model.cuda(),
        'use_viewdirs': args.use_viewdirs,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    # import configargparse
    # parser = configargparse.ArgumentParser()
    # parser.add_argument('--config', is_config_file=True,
    #                     help='config file path')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default='area3',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store ckpts and logs')
    parser.add_argument("--rgb_path", type=str, default='none',
                        help='input rgb path')
    parser.add_argument("--depth_path", type=str, default='none',
                        help='input depth path')

    # training options
    parser.add_argument("--iters", type=int, default=25000,
                        help='number of training iterations')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1400,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=20,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='st3d',
                        help='options: st3d / multi_mp3d ')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## st3d flags
    parser.add_argument("--use_depth", type=bool, default=True,
                        help='use depth to update')
    parser.add_argument("--use_dist", type=bool, default=True,
                        help='use distribution loss to update')
    parser.add_argument("--use_indoor", action='store_true',
                        help='penalty passing through rays to update')
    parser.add_argument("--use_gradient", type=bool, default=False,
                        help='use gradient to update')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=25000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=25000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=25000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    params = backend.load_parameters()
    args.__dict__.update(params)

    # Load data
    if args.dataset_type == 'st3d':
        rays_o, rays_d, rays_g, rays_rgb, rays_depth, hw, near, far = load_st3d_data(args.rgb_path, args.depth_path)
        rays_o, rays_o_test = rays_o
        rays_d, rays_d_test = rays_d
        rays_rgb, rays_rgb_test = rays_rgb
        rays_depth, rays_depth_test = rays_depth
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    H, W = hw

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # if args.config is not None:
    #     f = os.path.join(basedir, expname, 'config.txt')
    #     with open(f, 'w') as file:
    #         file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(0.1 ** (1 / (args.lrate_decay * 1000))))
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lrate, total_steps=args.iters + 5)

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_train_{}_{:06d}'.format('test' if args.render_test else 'path',
                                                                           start))

            os.makedirs(testsavedir, exist_ok=True)
            rays_o_test = torch.Tensor(rays_o_test).to(device)
            rays_d_test = torch.Tensor(rays_d_test).to(device)

            rgbs, _ = render_path([rays_o_test, rays_d_test], hw, render_kwargs_test, savedir=testsavedir,
                                  render_factor=args.render_factor)
            print('Done rendering', testsavedir)

            # calculate MSE and PSNR for last image(gt pose)
            gt_loss = img2mse(torch.tensor(rgbs[-1]).cuda(), torch.tensor(rays_rgb_test[-1]).cuda())
            gt_psnr = mse2psnr(gt_loss)
            print('ground truth loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
            with open(os.path.join(testsavedir, 'statistics.txt'), 'w') as f:
                f.write('loss: {}, psnr: {}'.format(gt_loss, gt_psnr))

            rgbs = np.concatenate([rgbs[:-1], rgbs[:-1][::-1]])
            imageio.mimwrite(os.path.join(testsavedir, 'video2.gif'), to8b(rgbs), fps=10)

            return

    # Move training data to GPU
    rays_o = torch.Tensor(rays_o).to(device)
    rays_d = torch.Tensor(rays_d).to(device)
    rays_rgb = torch.Tensor(rays_rgb).to(device)
    rays_depth = torch.Tensor(rays_depth).to(device)
    rays_o_test = torch.Tensor(rays_o_test).to(device)
    rays_d_test = torch.Tensor(rays_d_test).to(device)
    rays_rgb_test = torch.Tensor(rays_rgb_test).to(device)
    if args.use_gradient:
        rays_g = torch.Tensor(rays_g).to(device)

    print('shuffle rays')
    rand_idx = torch.randperm(rays_rgb.shape[0])
    rays_o = rays_o[rand_idx]
    rays_d = rays_d[rand_idx]
    rays_rgb = rays_rgb[rand_idx]
    rays_depth = rays_depth[rand_idx]
    if args.use_gradient:
        rays_g = rays_g[rand_idx]
    print('done')

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    i_batch = 0
    N_iters = args.iters + 1
    start = start + 1
    print('Begin, iter: %d' % start)
    for i in range(start, N_iters):
        batch_o = rays_o[i_batch:i_batch + N_rand]
        batch_d = rays_d[i_batch:i_batch + N_rand]

        target_rgb = rays_rgb[i_batch:i_batch + N_rand]
        target_depth = rays_depth[i_batch:i_batch + N_rand]
        target_g = rays_g[i_batch:i_batch + N_rand]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_o = rays_o[rand_idx]
            rays_d = rays_d[rand_idx]
            rays_rgb = rays_rgb[rand_idx]
            rays_depth = rays_depth[rand_idx]
            if args.use_gradient:
                rays_g = rays_g[rand_idx]

            i_batch = 0

        batch_o, batch_d, target_rgb, target_depth = batch_o.cuda(), batch_d.cuda(), target_rgb.cuda(), target_depth.cuda()
        if args.use_gradient:
            target_g = target_g.cuda()
        #####  Core optimization loop  #####

        rgb, dep, grad, weights, z_vals, extras = render(H, W, rays=[batch_o, batch_d], pts_noise=0,
                                                         **render_kwargs_train)

        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        img_loss = img2mse(rgb, target_rgb)

        # depth_loss
        if args.use_depth:
            depth_loss = img2mse(dep, target_depth)
        else:
            depth_loss = 0

        # distribution loss
        if args.use_dist:
            dist_loss = weights[z_vals < target_depth[:, None] * 0.99].mean()
        else:
            dist_loss = 0

        if args.use_indoor:
            indoor_loss = -torch.log(1 - weights.sum(dim=-1) + 1e-6).mean() * 0.01
        else:
            indoor_loss = 0

        if args.use_gradient:
            grad_loss = img2mse(grad, target_g)
        else:
            grad_loss = 0

        loss = img_loss + depth_loss + grad_loss + dist_loss + indoor_loss
        psnr = mse2psnr(img_loss)

        # loss for coarse network
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_rgb)
            loss = loss + img_loss0
            info = "img_loss0:" + str(img_loss0.detach())
            # depth_loss
            if args.use_depth:
                depth_loss0 = img2mse(extras['depth0'], target_depth)
                loss = loss + depth_loss0
                info += "depth_loss0: " + str(depth_loss0.detach())
            if args.use_dist:
                dist_loss0 = extras['weights0'][extras['z_vals0'] < target_depth[:, None] * 0.99].mean()
                loss = loss + dist_loss0
                info += "dist_loss0: " + str(dist_loss0.detach())
            if args.use_indoor:
                indoor_loss0 = -torch.log(1 - extras['weights0'].sum(dim=-1) + 1e-6).mean() * 0.01
                loss = loss + indoor_loss0
                info += "indoor_loss0: " + str(indoor_loss0.detach())
            if args.use_gradient:
                grad_loss0 = img2mse(extras['grad0'], target_g)
                loss = loss + grad_loss0

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        scheduler.step()
        ################################

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            backend.save_artifact(path, "checkpoint")

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            with torch.no_grad():
                rgbs, _ = render_path([rays_o_test, rays_d_test], hw, render_kwargs_test, savedir=testsavedir,
                                      render_factor=8)
            print('Done rendering', testsavedir)

            # calculate MSE and PSNR for last image(gt pose)
            # gt_loss = img2mse(torch.tensor(rgbs[-1]), torch.tensor(rays_rgb_test[-1]))
            # gt_psnr = mse2psnr(gt_loss)
            # print('ground truth loss: {}, psnr: {}'.format(gt_loss, gt_psnr))

            rgbs = np.concatenate([rgbs[:-1], rgbs[:-1][::-1]])
            imageio.mimwrite(os.path.join(testsavedir, 'video2.gif'), to8b(rgbs), fps=10)
            backend.save_artifact(os.path.join(testsavedir, 'video2.gif'), "video")

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    train()
