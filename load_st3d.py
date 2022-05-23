import numpy as np
from PIL import Image
import math
import cv2


def preprocess(rgb_path, depth_path):
    rgb = np.array(Image.open(rgb_path).convert('RGB')) / 255
    d = np.array(Image.open(depth_path))
    H, W = rgb.shape[:2]

    d[d == 65535] = 0
    d = d / 512.0  # accurate dis

    loops = 0
    while d.min() == 0:
        loops += 1
        if loops > 1000:
            assert False, "impossible to fill depth img"
        idx_0, idx_1 = np.where(d == 0)
        d_fill = np.zeros(d.shape)
        d_fill[idx_0, idx_1] = 1

        for i in range(H):
            y_idx = np.where(d_fill[i] > 0)[0]

            if len(y_idx) == 0: continue
            if len(y_idx) == 1:
                d_fill[i, y_idx[0]] = (d[i, y_idx[0] - 1] + d[i, (y_idx[0] + 1) % W]) / 2
                continue
            if len(y_idx) == W:
                d_fill[i] = 0
                if i != 0 and d[i - 1, 0] != 0:
                    d[i, 0] = d[i - 1, 0]
                else:
                    d[i, 0] = d[min(i + 1, H - 1), 0]
                continue

            gaps = [[s, e] for s, e in zip(y_idx, y_idx[1:]) if s + 1 < e]
            edges = np.concatenate([y_idx[:1], np.array(sum(gaps, [])), y_idx[-1:]])

            interval = [[int(s), int(e) + 1] for s, e in zip(edges[::2], edges[1:][::2])]
            if interval[0][0] == 0:
                interval[0][0] = interval[-1][0] - W
                interval = interval[:-1]

            for s, e in interval:
                if s < 0:
                    interp = np.linspace(d[i, s - 1], d[i, (e + 1) % W], e - s)
                    d_fill[i, s:] = interp[:-s]
                    d_fill[i, :e] = interp[-s:]
                else:
                    d_fill[i, s:e] = np.linspace(d[i, s - 1], d[i, (e + 1) % W], e - s)
        d = d + d_fill
    return rgb, d


def getRay_d(H, W):
    _y = np.repeat(np.array(range(W)).reshape(1, W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1, H), W, axis=0).T

    _theta = (1 - 2 * _x / H) * np.pi / 2  # latitude
    _phi = 2 * math.pi * (0.5 - _y / W)  # longitude

    axis0 = (np.cos(_theta) * np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1)
    axis2 = (-np.cos(_theta) * np.sin(_phi)).reshape(H, W, 1)
    original_coord = np.concatenate((axis0, axis1, axis2), axis=2)
    return original_coord


def load_st3d_data(rgb_path, depth_path):
    rgb, d = preprocess(rgb_path, depth_path)
    # normalize to [0, 1]
    d = d.reshape(rgb.shape[0], rgb.shape[1], 1)
    H, W = rgb.shape[:2]
    near, far = d.min() * 0.95, d.max() * 1.05

    gradient = cv2.Laplacian(rgb, cv2.CV_64F)
    gradient = 2 * (gradient - np.min(gradient)) / np.ptp(gradient) - 1

    original_coord = getRay_d(H, W)
    # *c2w here
    coord = original_coord * d

    # load training camera poses
    num_train = 1

    # load testing camera poses
    cam_origin = np.array([[0.0, 0.0, 0.0],
                           [-0.05, 0., -0.05],
                           [-0.03, 0., -0.05],
                           [-0.01, 0., -0.05],
                           [0.01, 0., -0.05],
                           [0.03, 0., -0.05],
                           [0.05, 0., -0.05],
                           [0.05, 0., -0.03],
                           [0.05, 0., -0.01],
                           [0.05, 0., 0.01],
                           [0.05, 0., 0.03],
                           [0.0, 0.0, 0.0]]) * 10

    rays_o, rays_d, rays_g, rays_rgb, rays_depth = [], [], [], [], []
    rays_o_test, rays_d_test, rays_rgb_test, rays_depth_test = [], [], [], []

    for i, p in enumerate(cam_origin):
        dep = np.linalg.norm(coord - p, axis=-1)

        if i < num_train:
            dir = coord - p  # direction = end point - start point
            dir = dir / np.linalg.norm(dir, axis=-1)[..., None]
            rays_o.append(np.repeat(p.reshape(1, -1), H * W, axis=0))
            rays_g.append(gradient.reshape(-1, 3))
            rays_d.append(dir.reshape(-1, 3))
            rays_rgb.append(rgb.reshape(-1, 3))
            rays_depth.append(dep.reshape(-1))

        elif i < num_train + 10:
            rays_o_test.append(np.repeat(p.reshape(1, -1), H * W, axis=0))
            rays_d_test.append(original_coord.reshape(-1, 3))
            rays_rgb_test.append(rgb.reshape(-1, 3))  # anyway fill it with rgb
            rays_depth_test.append(dep.reshape(-1))

        elif i == num_train + 10:
            rays_o_test.append(np.repeat(p.reshape(1, -1), H * W, axis=0))
            rays_d_test.append(coord.reshape(-1, 3))
            rays_rgb_test.append(rgb.reshape(-1, 3))
            rays_depth_test.append(dep.reshape(-1))

    rays_o, rays_o_test = np.concatenate(rays_o, axis=0), np.concatenate(rays_o_test, axis=0)
    rays_d, rays_d_test = np.concatenate(rays_d, axis=0), np.concatenate(rays_d_test, axis=0)
    rays_g = np.concatenate(rays_g, axis=0)
    rays_rgb, rays_rgb_test = np.concatenate(rays_rgb, axis=0), np.concatenate(rays_rgb_test, axis=0)
    rays_depth, rays_depth_test = np.concatenate(rays_depth, axis=0), np.concatenate(rays_depth_test, axis=0)

    # rays_o, rays_d, rays_g, rays_rgb, rays_depth, [H, W]
    # all in flatten format : [N(~H*W*num_train), 3 or 1]
    return [rays_o, rays_o_test], [rays_d, rays_d_test], rays_g, [rays_rgb, rays_rgb_test], \
           [rays_depth, rays_depth_test], [H, W], near, far
