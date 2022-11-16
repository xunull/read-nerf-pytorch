import os
import torch
import numpy as np
import imageio
import json
import cv2

# 平移
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# 绕x轴的旋转
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# 绕y轴的旋转
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    """
    theta: -180 -- +180，间隔为9
    phi: 固定值 -30
    radius: 固定值 4
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    """
    testskip: test和val数据集，只会读取其中的一部分数据，跳着读取
    """
    splits = ['train', 'val', 'test']
    # 存储了三个json文件的数据
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            # 测试集如果数量很多，可能会设置testskip
            skip = testskip
        # 读取所有的图片，以及所有对应的transform_matrix
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # 归一化
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)，4通道 rgba
        poses = np.array(poses).astype(np.float32)
        # 用于计算train val test的递增值
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    # train val test 三个list
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    # train test val 拼一起
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    # meta使用了上面的局部变量，train test val 这个变量值是相同的，文件中这三个值确实是相同的
    camera_angle_x = float(meta['camera_angle_x'])
    # 焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    #  np.linspace(-180, 180, 40 + 1) 9度一个间隔
    # (40,4,4), 渲染的结果就是40帧
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        # 焦距一半
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            # 调整成一半的大小
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split
