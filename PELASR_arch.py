import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class PELA(nn.Module):
    '''
    Perception-enhanced Linear Attention（PeLA）
    '''
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        # 确保输入维度能被头数整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  # 输入特征维度
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度

        # QKV投影层（仅Q单独投影，KV联合投影）
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力 dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出 dropout

        self.sr_ratio = sr_ratio  # 空间缩减比例（降低计算量）
        if sr_ratio > 1:
            # 空间缩减卷积（类似下采样）
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)  # 空间缩减后的归一化

        self.focusing_factor = focusing_factor  # 聚焦因子（增强重要特征）
        # 深度可分离卷积（捕获局部特征）
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))  # 缩放参数

        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))


    def forward(self, x, H, W):
        # print(x.shape)
        b, c, h, w = x.shape
        x = to_3d(x)
        B, N, C = x.shape  # B:批次，N:序列长度（patch数），C:特征维度
        q = self.q(x)  # Q投影：(B, N, C)

        # 计算KV（根据sr_ratio决定是否空间缩减）
        if self.sr_ratio > 1:
            # 空间缩减：将序列维度从N=H*W缩减为(H/sr_ratio)*(W/sr_ratio)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
            h_k, w_k = x_.shape[2] // self.sr_ratio, x_.shape[3] // self.sr_ratio  # 动态获取下采样后的尺寸
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # (B, N', C)，N'=N/(sr_ratio^2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)  # 分离K和V
        else:
            h_k, w_k = h, w
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        k, v = kv[0], kv[1]  # K和V: (B, N', C)
        n = k.shape[1]  # 缩减后的序列长度

        # 动态生成位置编码（基于当前K的尺寸）
        if self.sr_ratio > 1:
            pos_encoding = nn.Parameter(torch.zeros(B, C, h_k, w_k, device=x.device))  # 初始化空编码
            pos_encoding = to_3d(pos_encoding)  # 转为3D [b, n_k, c]
        else:
            pos_encoding = nn.Parameter(torch.zeros(B, C, h, w, device=x.device))
            pos_encoding = to_3d(pos_encoding)

        # 位置编码注入K
        k = k + pos_encoding

        # 聚焦机制：增强重要特征的权重
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()  # 激活函数确保非负性
        scale = nn.Softplus()(self.scale)  # 缩放参数（确保为正）
        q = kernel_function(q) + 1e-6  # 避免零值
        k = kernel_function(k) + 1e-6
        q = q / scale  # 缩放
        k = k / scale
        # 归一化后通过聚焦因子增强
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor  # 逐元素幂运算增强差异
        k = k ** focusing_factor
        # 恢复原始模长比例
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        # 多头注意力拆分
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N', head_dim)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N', head_dim)

        # 线性注意力计算（低复杂度O(N)）
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # 归一化因子
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))  # K^T @ V（带缩放）
        x = q @ kv * z  # 注意力输出

        # 若使用空间缩减，需将V插值回原始序列长度（用于后续局部特征融合）
        if self.sr_ratio > 1:
            v = F.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n),
                              size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        # 融合局部特征（深度可分离卷积）
        x = x.transpose(1, 2).reshape(B, N, C)  # 多头合并：(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)  # 重塑为特征图
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)  # 局部特征残差连接

        # 输出投影和dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        x = to_4d(x, h, w)

        return x


class RFFN(nn.Module):
    """
    Refined Feature Feed-forward Network（RFFN）
    """
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        # 线性层1：升维并拆分双分支（输入BCHW需先展平为序列）
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),  # 输出=2×hidden_dim，拆分双分支
            act_layer()
        )
        # 深度可分离卷积：空间特征精炼
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        # 线性层2：降维恢复输入通道数
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

        self.dim = dim
        self.hidden_dim = hidden_dim
        # 通道拆分配置：1/4通道卷积精炼，3/4通道直接保留
        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        # 部分通道精炼卷积
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        """
        前向传播流程（输入输出均为BCHW）：
        部分通道精炼 → 展平序列 → 门控融合 → 重塑回BCHW
        """
        B, C, H, W = x.size()  # 解析输入维度：Batch, Channels, Height, Width
        HW = H * W  # 序列长度

        # 步骤1：部分通道精炼（仅1/4通道卷积增强）
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)  # 通道拆分
        x1 = self.partial_conv3(x1)  # 1/4通道卷积精炼
        x_refined = torch.cat((x1, x2), 1)  # 通道拼接恢复维度

        # 步骤2：展平为序列（BCHW → B×HW×C），适配线性层
        x_seq = rearrange(x_refined, 'b c h w -> b (h w) c')  # [B, HW, C]

        # 步骤3：升维与门控分支拆分
        x_seq = self.linear1(x_seq)  # [B, HW, C] → [B, HW, 2×hidden_dim]
        x1_gate, x2_gate = x_seq.chunk(2, dim=-1)  # 拆分双分支：各[B, HW, hidden_dim]

        # 步骤4：深度卷积精炼与门控融合
        # 序列 → BCHW：适配卷积
        x1_gate_bchw = rearrange(x1_gate, 'b (h w) c -> b c h w', h=H, w=W)
        x1_gate_refined = self.dwconv(x1_gate_bchw)  # 深度卷积空间精炼
        # BCHW → 序列：恢复适配门控
        x1_gate_seq = rearrange(x1_gate_refined, 'b c h w -> b (h w) c')
        x_gated = x1_gate_seq * x2_gate  # 门控融合：动态筛选

        # 步骤5：降维与重塑回BCHW
        x_out_seq = self.linear2(x_gated)  # [B, HW, hidden_dim] → [B, HW, C]
        x_out = rearrange(x_out_seq, 'b (h w) c -> b c h w', h=H, w=W)  # [B, C, H, W]

        return x_out


class Block(nn.Module):
    def __init__(self, dim, ffn_scale=2):
        super().__init__()

        self.rela = PELA(dim, num_heads=4, sr_ratio=2)
        self.rffn = RFFN(dim, ffn_scale * dim)
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.norm2 = LayerNorm(dim, data_format='channels_first')

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.rela(self.norm1(x), h, w) + x
        x = self.rffn(self.norm2(x)) + x
        return x


@ARCH_REGISTRY.register()
class PELASR(nn.Module):
    def __init__(self, dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[Block(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    upscale = 4
    height = (1280 // upscale)
    width = (720 // upscale)

    model = PELASR(dim=64, n_blocks=14, ffn_scale=2, upscaling_factor=upscale)
    input = torch.randn(1, 3, height, width)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    macs, params = profile(model.to(device), inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs:", macs)
    print("params:", params)
