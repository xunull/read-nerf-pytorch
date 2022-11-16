import torch
import torch.nn as nn
import numpy as np

__all__ = ['img2mse', 'mse2psnr', 'to8b', 'get_embedder', 'get_rays', 'get_rays_np', 'ndc_rays', 'sample_pdf']

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
        d = self.kwargs['input_dims']  # 3
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # sin(x),sin(2x),sin(4x),sin(8x),sin(16x),sin(32x),sin(64x),sin(128x),sin(256x),sin(512x)
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns

        # 3D坐标是63，2D方向是27
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# 位置编码相关
def get_embedder(multires, i=0):
    """
    multires: 3D 坐标是10，2D方向是4
    """
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
    # 第一个返回值是lamda，给定x，返回其位置编码
    return embed, embedder_obj.out_dim


# ----------------------------------------------------------------------------------------------------------------------

# Ray helpers
def get_rays(H, W, K, c2w):
    """
    K：相机内参矩阵
    c2w: 相机到世界坐标系的转换
    """
    # j
    # [0,......]
    # [1,......]
    # [W-1,....]
    # i
    # [0,..,H-1]
    # [0,..,H-1]
    # [0,..,H-1]

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i = i.t()
    j = j.t()
    # [400,400,3]
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dirs [400,400,3] -> [400,400,1,3]
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d [400,400,3]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 前三行，最后一列，定义了相机的平移，因此可以得到射线的原点o
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    # 与上面的方法相似，这个是使用的numpy，上面是使用的torch
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    bins: z_vals_mid
    """

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    # 归一化 [bs, 62]
    # 概率密度函数
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 累积分布函数
    cdf = torch.cumsum(pdf, -1)
    # 在第一个位置补0
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])  # [bs,128]

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF

    u = u.contiguous()
    # u 是随机生成的
    # 找到对应的插入的位置
    inds = torch.searchsorted(cdf, u, right=True)
    # 前一个位置，为了防止inds中的0的前一个是-1，这里就还是0
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 最大的位置就是cdf的上限位置，防止过头，跟上面的意义相同
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # (batch, N_samples, 2)
    inds_g = torch.stack([below, above], -1)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # (batch, N_samples, 63)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # 如[1024,128,63] 提取 根据 inds_g[i][j][0] inds_g[i][j][1]
    # cdf_g [1024,128,2]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 如上, bins 是从2到6的采样点，是64个点的中间值
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 差值
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # 防止过小
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom

    # lower+线性插值
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# ----------------------------------------------------------------------------------------------------------------------

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
