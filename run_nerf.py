import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.构造一个适用于较小批次的“fn”版本。
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.准备输入数据，并应用网络fn
    fn：nerf网络
    embed_fn：位置编码器
    embeddirs_fn：方向编码器
    netchunk：并行发送的点数
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)  # 对输入进行位置编码，得到编码后的结果，是一个 array 数组
    # 视图方向不为 None，即输入了视图方向，那么我们就应该考虑对视图方向作出处理，用以生成颜色
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # 对视图方向进行编码
        embedded = torch.cat([embedded, embedded_dirs], -1)  # 将编码后的位置和方向聚合到一起

    outputs_flat = batchify(fn, netchunk)(embedded)  # 将编码过的点以批处理的形式输入到 网络模型 中得到 输出（RGB,A）
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True, near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.图像的高度（以像素为单位）。
      W: int. Width of image in pixels.图像的宽度（以像素为单位）。
      focal: float. Focal length of pinhole camera.针孔相机的焦距。
      chunk: int. Maximum number of rays to process simultaneously. 并行处理的最大光线数。
        Used to control maximum memory usage. Does not affect final results.用于控制最大内存使用量。不影响最终结果。
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for each example in batch.批处理中每个示例的射线原点和方向。
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.相机到世界的变换矩阵。
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.表示射线原点在NDC坐标中的方向。
      near: float or array of shape [batch_size]. Nearest distance for a ray.射线的最近距离。
      far: float or array of shape [batch_size]. Farthest distance for a ray.射线的最远距离。
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.如果为True，则在模型中使用视点方向。
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.射线的预测RGB值。
      disp_map: [batch_size]. Disparity map. Inverse of depth.视差图。深度的倒数。
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.沿射线累积的不透明度（alpha）。
      extras: dict with everything returned by render_rays().保存render_rays（）返回的所有内容，保存在字典中。
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays
    # 如果使用视图方向，计算光线 ray_d 的单位方向作为 view_dirs
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d   # 视点方向
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)    # 将 viewdirs 中的每个向量归一化为单位长度向量
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float() # 貌似该行可以不要

    sh = rays_d.shape  # [B, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # 生成光线的远近端，用于确定边界框，并将其聚合到 rays 中: [1024, 8]
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    # 视图方向聚合到光线中: [1024, 8+3]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape 开始并行计算光线属性
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.实例化NeRF的MLP模型。
    """
    # 位置编码器
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)  # input_ch为3维坐标(x,y,z)编码(L=10)之后的维度：3x2x10
    # 方向编码器
    input_ch_views = 0  # 方向坐标(θ,Φ,?)编码(L=4)之后的维度：3x2x4。方向坐标给的是二维度的，但要转化成标准三自由度编码
    embeddirs_fn = None
    if args.use_viewdirs:  # 若考虑方向，则获取 方向坐标编码器 和 编码后坐标的维度
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4  # 输出的维度。若是细采样，维度为5，否则为4
    skips = [4]
    # 创建模型
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())  # 模型中的梯度变量
    # 若为细采样阶段
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
    # 定义一个查询点的颜色和密度的匿名函数，即给定点坐标、方向以及查询网络，我们就可以得到该点在该网络下的输出（[rgb,alpha]）
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.chunk)
    # 创建 optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # 加载 checkpoints
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

        # 加载 model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,   # 一个匿名函数，输入位置坐标、方向坐标以及神经网络，返回该点对应的颜色和密度
        'perturb': args.perturb,    # 扰动，对整体算法理解没有影响
        'N_samples': args.N_samples,  # 粗网络的采样点数
        'network_fn': model,    # 粗网络模型
        'network_fine': model_fine,  # 细网络模型
        'N_importance': args.N_importance,  # 每条光线上细采样点的数量
        'use_viewdirs': args.use_viewdirs,  # 是否使用视点方向，影响到神经网络是否输出颜色
        'white_bkgd': args.white_bkgd,  # 如果为 True 将输入的 png 图像的透明部分转换成白色
        'raw_noise_std': args.raw_noise_std,    # 归一化密度
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.将模型的预测转换为语义上有意义的值。
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.模型预测的原始结果
        z_vals: [num_rays, num_samples along ray]. 每天射线方向的N_samples个深度采样值，介于[near, far]之间
        rays_d: [num_rays, 3]. Direction of each ray.每条光线的方向。
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.光线的估计RGB颜色。
        disp_map: [num_rays]. Disparity map. Inverse of depth map.视差图。深度图的倒数。
        acc_map: [num_rays]. Sum of weights along each ray.每条射线的权重总和。
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.每个采样颜色对应的权重。
        depth_map: [num_rays]. Estimated distance to object.到物体的估计距离。
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # 计算相邻两点之间的深度差
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)     # 转换为？
    # 颜色
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3] 每个点的 RGB 值
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples] 即透明度
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) #
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.射线批
      network_fn: function. Model for predicting RGB and density at each point in space.用于预测空间中每个点的RGB和密度的模型。
      network_query_fn: function used for passing queries to network_fn.用于将查询传递到network_fn的函数。
      N_samples: int. Number of different times to sample along each ray.沿每条射线采样的次数。
      retraw: bool. If True, include model's raw, unprocessed predictions.如果为True，则包括模型的原始、未处理的预测结果。
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.如果为True，则在反向深度中进行线性采样。
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.如果非零，则在时间上的分层随机点对每条射线进行采样。
      N_importance: int. Number of additional times to sample along each ray.# 每条射线附加的“细”样本数
        These samples are only passed to network_fine.这些采样点仅传递给network_fine进行调优。
      network_fine: "fine" network with same spec as network_fn.与network_fn规格相同的“精细”网络。
      white_bkgd: bool. If True, assume a white background.如果为True，则假设为白色背景。
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.如果为True，则打印更多调试信息。
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.光线的估计RGB颜色。来自fine模型。
      disp_map: [num_rays]. Disparity map. 1 / depth.视差图。深度的倒数。
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.沿每条光线累积的不透明度。
      raw: [num_rays, num_samples, 4]. Raw predictions from model.来自模型的原始预测。
      rgb0: See rgb_map. Output for coarse model.          coarse模型的输出。
      disp0: See disp_map. Output for coarse model.        coarse模型的输出。
      acc0: See acc_map. Output for coarse model.          coarse模型的输出。
      z_std: [num_rays]. Standard deviation of distances along ray for each sample.每个样本沿射线距离的标准偏差。
    """
    # 从 ray 中分离出 rays_o, rays_d, viewdirs, near, far
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    # 确定采样点位置
    t_vals = torch.linspace(0., 1., steps=N_samples)  # 在 0-1 内生成 N_samples 个等差点
    # 根据参数确定不同的采样方式（正采样或者逆采样）,从而确定采样点在边界框内的的具体位置
    """对于 z_vals 的计算。举一个例子，我们想要生成 a-b 之间的 n 个均匀的数组，我们改如何计算呢？
    按照我们习惯的计算方法应该将将 a - b 之间的长度分成 n-1 等份，每份距离为 d ，然后依次将 d 加到 a 上即可，
    表述为数学形式即为：
    d = (b-a)/(n-1)
    ai = a + i*d 
       = a + i*(b-a)/(n-1)
       = a * (1-i/(n-1)) + b * i/(n-1)  (i = 0,1,2,....,n-1)
    参考：https://blog.csdn.net/qq_41071191/article/details/126052795#comments_26257558
    """
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    # N_samples个采样深度值，介于[near, far]之间
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    # 生成光线上每个采样点的位置。空间中的点的位置可以表述为：原点+方向与距离的乘积。
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # 将光线上的每个点投入到 MLP 网络 network_fn 中前向传播得到每个点对应的 （RGB，σ）
    raw = network_query_fn(pts, viewdirs, network_fn)
    # 渲染，即进行离散点的积分
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')  # 生成config.txt文件
    parser.add_argument("--expname", type=str,
                        help='experiment name')  # 指定实验名称
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')  # 指定输出目录
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')  # 指定数据目录

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    # 采样的射线个数，即采样的像素点个数，这个数字在 model_c 和 model_f 中是一样的，
    # 在 lego 数据集中默认 1024，则在model_c 和 model_f 中各自采样 1024 条射线。
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--c", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
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
    parser.add_argument("--raw_noise_std", type=float, default=0.,
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
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # 1、加载数据
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    elif args.dataset_type == 'blender':
        # 加载数据
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        # train,val, test对应数据集的id。以lego数据集为例，共有138=100+13+25张图片：[0...99], [100...112], [113, 137]
        i_train, i_val, i_test = i_split
        # 设定边界框的远近边界
        near = 2.
        far = 6.
        # 将 RGBA 转换成 RGB 图像
        if args.white_bkgd:
            # 如果使用白色背景
            alpha = images[..., -1:]
            b = (1. - alpha)
            # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            images = images[..., :3] * images[..., -1:] + b
        else:
            images = images[..., :3]
    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 构造内参矩阵
    if K is None:
        K = np.array([  # 这里偏移量cx,cy分别取W，H的一半
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
    # 判断：渲染测试集位姿 还是 渲染render_poses（生成的180度环形视角的位姿[40,4,4]）
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file日志目录和配置目录
    basedir = args.basedir  # 指定输出目录
    expname = args.expname  # 指定实验名称
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 2、初始化网络模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # 在网络模型训练好以后我们保存整个网络，在测试渲染时只需要将 render_only 参数置 True，不再对网络进行训练，直接得到渲染结果
    if args.render_only:    # 仅渲染（当网络已训练好时可设置为True，则不再对网络进行训练，直接得到渲染结果）
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return
    # 3、如果以批处理的形式对进行训练，首先生成所有图片的光线
    # Prepare raybatch tensor if batching random rays 如果批处理随机射线，则准备射线批处理张量
    N_rand = args.N_rand    # 采样的射线个数，即采样的像素点个数
    use_batching = not args.no_batching     # 是否以批处理的形式生成光线
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-？)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    # 4、开始训练
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            # 分批加载光线，大小为 N_rand
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3]
            batch = torch.transpose(batch, 0, 1)    # [2+1, B, 3]
            # batch_rays[2, B, 3]表示[ro+rd,B,3]; target_s[B, 3]表示每条射线的真实颜色
            batch_rays, target_s = batch[:2], batch[2]  # [2, B, 3]  [B, 3]  将光线和对应的像素点颜色分离
            i_batch += N_rand
            # 经过一定批次的处理后，所有的图片都经过了一次。这时候要对数据打乱，重新再挑选
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])    # 生成随机序号
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        # Random from one image
        else:
            # 从所有的图像中随机选择一张图像用于训练
            img_i = np.random.choice(i_train)
            target = images[img_i]  # rgb
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4] # 位姿

            if N_rand is not None:
                # 获取该图片中每个像素点对应的光线的 原点和方向
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                # 前precrop_iters轮 生成(H/4~3H/2) * (W/4~3W/4)范围内的像素坐标(当precrop_frac取0.5时)
                # 相当于从图像中心crop(裁剪)出一个(H/2, W/2)的图像
                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)    # dH = H/4
                    dW = int(W // 2 * args.precrop_frac)    # dW = W/4
                    # 生成H/2 * W/2范围内每个像素点的笛卡尔坐标 (H/2, W/2, 2)
                    coords = torch.stack(torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                    ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                # 后N_iters-precrop_iters轮 生成整个图像中每个像素的笛卡尔坐标
                else:
                    # 生成整个图像中每个像素的笛卡尔坐标# (H, W, 2)
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)

                coords = torch.reshape(coords, [-1, 2])  # (H*W, 2)
                # 选择N_rand个射线，而不是全部
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)  #随机选N_rand个
                select_coords = coords[select_inds].long()  # (N_rand, 2)   # 像素坐标
                # 选择对应的光线
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                # rgb真值 ground truth
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        """
         可以注意到：不论是 use_batching 的情况还是单张图像的情况，每个 epoch 选择的光线的数量是恒定的，即 N_rand 。
         这么做实际上是为了减少计算的工程量。虽然每次都只随机挑选了一部分像素对应的光线，但是经过多达 200 000 次的训练
         实际上已经足以把所有的像素对应的光线都挑选一遍了。我们得到生成的光线以后就可以对光线进行渲染操作了。
        """
        # 训练
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()