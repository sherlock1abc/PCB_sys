import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()
        # 如果没有设置输出通道，就保持和输入通道一致
        if out_channels is None:
            out_channels = in_channels

        # 深度卷积：每个通道单独卷积，提取局部特征
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        # 逐点卷积：用 1x1 卷积整合通道信息
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        # 批归一化：加速收敛，提升训练稳定性
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 激活函数：LeakyReLU，避免神经元完全失效
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # 第一步：先做深度卷积 + BN + 激活
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        # 第二步：再做逐点卷积 + BN + 激活
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x
      
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 平均池化：取每个通道的平均值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化：取每个通道的最大值
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两层全连接（用1x1卷积代替）实现通道权重的学习
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),   # SiLU 激活，平滑版 ReLU
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出范围 [0,1]，作为权重系数
        )

    def forward(self, x):
        # 平均池化结果送入全连接
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化结果送入全连接
        max_out = self.fc(self.max_pool(x))
        # 两个结果相加，形成最终的通道注意力
        scale = avg_out + max_out
        # 输入特征乘以注意力权重
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 先用 7x7 卷积融合 4 个统计量通道，再用 1x1 卷积压缩
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, groups=1),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算每个像素点的通道均值
        mean_out = torch.mean(x, dim=1, keepdim=True)
        # 计算每个像素点的通道最大值
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 计算每个像素点的通道最小值
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        # 计算每个像素点的通道和
        sum_out = torch.sum(x, dim=1, keepdim=True)

        # 把四种统计量拼接在一起，形成 4 通道特征
        pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        # 卷积得到空间注意力图
        attention = self.conv(pool)
        # 输入特征乘以空间注意力权重
        return x * attention

class MYCASAB(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MYCASAB, self).__init__()
        # 卷积块（深度可分离卷积）
        self.convblock = ConvBlock(in_channels, in_channels)
        # 通道注意力模块
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空间注意力模块
        self.spatial_attention = SpatialAttention()

        # 门控融合模块：用于动态融合通道与空间注意力
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 先提取卷积特征
        residual = x  # 保留原始输入用于残差连接
        x = self.convblock(x)

        # 并行处理：通道注意力和空间注意力
        ca = self.channel_attention(x)  # 通道增强
        sa = self.spatial_attention(x)  # 空间增强

        # 门控融合：动态决定两个分支的权重
        fused = torch.cat([ca, sa], dim=1)  # [B, 2C, H, W]
        gate = self.sigmoid(self.gate_conv(fused))  # [B, C, H, W]，作为控制门
        out = gate * ca + (1 - gate) * sa  # 动态加权融合

        # 残差连接
        out += residual

        return out
class InceptionDWAttention(nn.Module):
    """通道注意力模块，用于动态加权多分支输出"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(in_channels // reduction, 4)  # 至少保留4维隐藏层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class InceptionDWConv2dDynamic(nn.Module):
    """
    改进版 Inception 风格动态深度卷积模块：
    - 支持方形卷积 (3x3)
    - 支持水平/垂直带状卷积 (1xK, Kx1)
    - 动态通道分配 + 注意力加权
    - 健壮的通道切分逻辑
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=7, branch_ratio=0.125):
        super().__init__()

        # 健壮的通道分配：确保 3 * gc <= in_channels
        gc = max(1, int(in_channels * branch_ratio))
        while 3 * gc > in_channels and gc > 1:
            gc -= 1
        total_gc = 3 * gc
        id_channels = in_channels - total_gc

        self.gc = gc
        self.split_indexes = (id_channels, gc, gc, gc)

        # 深度可分离卷积分支
        self.dwconv_hw = nn.Conv2d(
            gc, gc, kernel_size=square_kernel_size,
            padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0), groups=gc
        )

        # 批归一化
        self.bn_hw = nn.BatchNorm2d(gc)
        self.bn_w = nn.BatchNorm2d(gc)
        self.bn_h = nn.BatchNorm2d(gc)

        # 注意力与激活函数
        self.attention = InceptionDWAttention(3 * gc, reduction=8)
        self.act = nn.SiLU()  # SiLU = Swish，YOLO 系列常用

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming 初始化，适配 ReLU/SiLU 激活"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 切分输入通道：[identity, hw, w, h]
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        # 拼接三路特征用于注意力计算
        x_branches = torch.cat([x_hw, x_w, x_h], dim=1)
        attn = self.attention(x_branches)
        a_hw, a_w, a_h = torch.split(attn, [self.gc, self.gc, self.gc], dim=1)

        # 方形卷积分支
        x_hw = self.dwconv_hw(x_hw)
        x_hw = self.bn_hw(x_hw)
        x_hw = self.act(x_hw)
        x_hw = x_hw * a_hw

        # 水平带状卷积分支
        x_w = self.dwconv_w(x_w)
        x_w = self.bn_w(x_w)
        x_w = self.act(x_w)
        x_w = x_w * a_w

        # 垂直带状卷积分支
        x_h = self.dwconv_h(x_h)
        x_h = self.bn_h(x_h)
        x_h = self.act(x_h)
        x_h = x_h * a_h

        # 拼接所有分支（含 identity 路径）
        return torch.cat([x_id, x_hw, x_w, x_h], dim=1)

class C3k2_InceptionDWDynamic(nn.Module):
    """
    基于 CSP 思想的递归式特征增强模块，
    内部使用 ImprovedInceptionDWConv2dDynamic 作为处理单元。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, branch_ratio=0.2):
        super().__init__()
        self.c2 = c2
        self.add = shortcut and c1 == c2  # 是否启用残差连接
        c_ = int(c2 * e)  # 隐层通道数

        # 输入变换与初始分支
        self.cv1 = nn.Conv2d(c1, 2 * c_, 1, 1)
        # 输出融合层
        self.cv2 = nn.Conv2d((2 + n) * c_, c2, 1, 1)
        self.bn2 = nn.BatchNorm2d(c2)

        # 递归处理链
        self.m = nn.ModuleList([
            InceptionDWConv2dDynamic(
                c_,
                square_kernel_size=3,
                band_kernel_size=7,
                branch_ratio=branch_ratio
            ) for _ in range(n)
        ])

    def forward(self, x):
        # 初始切分：两路 c_ 通道
        y = list(torch.chunk(self.cv1(x), 2, dim=1))  # [y0, y1]

        # 递归增强：每次处理上一步输出
        for module in self.m:
            y.append(module(y[-1]))  # y2, y3, ..., y_{n+1}

        # 特征拼接 + 通道压缩
        out = self.cv2(torch.cat(y, dim=1))
        out = self.bn2(out)

        # 残差连接
        return out + x if self.add else out
  
