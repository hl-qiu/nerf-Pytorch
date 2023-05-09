import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

# 构建平移矩阵，将球心沿着 z 轴移动 t 个单位
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()
# 构建绕 x 轴旋转 phi 度的旋转矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()
# 构建绕 y 轴旋转 th 度的旋转矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)   # 沿着 z 轴移动 radius 个单位
    c2w = rot_phi(phi/180.*np.pi) @ c2w # 绕 x 轴旋转 phi 度
    c2w = rot_theta(theta/180.*np.pi) @ c2w # 绕 y 轴旋转 theta 度
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    # 读取
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1    # 如果是 train 文件夹，连续读取图像数据
        else:
            skip = testskip # 否则，按照步长testskip读取图像数据
        # 以指定步长读取frame中的位姿矩阵
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)   # 包含了 train、test、val 的图像的列表
        all_poses.append(poses) # 包含了 train、test、val 的位姿的列表
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    # 把列表聚合为一个数组 [N, H, W, 4]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    # 计算焦距
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # 制作用于测试训练效果的 渲染pose[40,4,4]
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    # 缩放图像
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    """
        imgs : 根据 .json 文件加载到的所有图像数据:（N，H，W，4）其中 N 代表用于 train、test、val 的总数量
        poses : 转置矩阵。（N，4，4）
        render_poses : 用于测试的环绕位姿 pose： （40，4，4）
        i_split : [[0:train], [train:val], [val:test]]
    """
    return imgs, poses, render_poses, [H, W, focal], i_split


