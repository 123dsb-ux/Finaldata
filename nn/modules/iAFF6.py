import torch
import torch.nn as nn
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=1, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class APConv(nn.Module):
    """非对称卷积分支"""
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步长（默认1）
        :param g: 分组卷积的组数（默认4）
        """
        super().__init__()
        
        # ------------------------------
        # 分支1: 非对称卷积替换5x5 (分解为1x5 + 5x1)
        # ------------------------------
        self.asym_conv5 = nn.Sequential(
            # 1x5 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, (1,5), stride=stride, padding=(0,2), groups=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 5x1 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, (5,1), stride=1, padding=(2,0), groups=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 1x1逐点卷积调整通道数
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        
        # ------------------------------
        # 分支2: 非对称卷积替换3x3 (分解为1x3 + 3x1)
        # ------------------------------
        self.asym_conv3 = nn.Sequential(
            # 1x3 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, (1,3), stride=stride, padding=(0,1), groups=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 3x1 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, (3,1), stride=1, padding=(1,0), groups=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 1x1逐点卷积调整通道数
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        
        # ------------------------------
        # 分支3: 1x1逐点卷积
        # ------------------------------
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        
        # ------------------------------
        # 批归一化与激活函数
        # ------------------------------
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 分支1: 非对称卷积（1x5 + 5x1）
        x1 = self.asym_conv5(x)
        
        # 分支2: 非对称卷积（1x3 + 3x1）
        x2 = self.asym_conv3(x)
        
        # 分支3: 1x1逐点卷积
        x3 = self.conv1x1(x)
        
        # 特征相加 → 批归一化 → 激活
        out = x1 + x2 + x3
        out = self.bn(out)
        return self.act(out)

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''
 
    def __init__(self, channels=64, r=2):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)
 
        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )
 
        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, residual):
 
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
 
        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
 
 
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
 
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
 
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
 
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class C2f_iAFF(nn.Module):
    """改进的CSP结构（用APConv替换原Conv）"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=4, e=0.5):  # 修改：新增g参数传递
        super().__init__()
        self.c = int(c2 * e)
        
        self.cv1 = APConv(c1, 2 * self.c, stride=1, g=g)  # 输入c1，输出2*self.c，stride=1
        self.cv2 = APConv((2 + n) * self.c, c2, stride=1, g=g) 
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # 分割通道
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
class Bottleneck(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.iAFF = iAFF(c2)
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        if self.add:
           results =  self.iAFF(x , self.cv2(self.cv1(x)))
        else:
            results = self.cv2(self.cv1(x))
        return results
 
 
if __name__ == '__main__':
    x = torch.ones(8, 64, 32, 32)
    channels = x.shape[1]
    model = C2f_iAFF(channels, channels, True)
    output = model(x)
    print(output.shape)
