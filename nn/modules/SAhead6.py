import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import dist2bbox, make_anchors
# classes
 
__all__ = ['Segment_SA', 'Pose_SA', 'Detect_SA']
 
def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.
    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
 
 
class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""
 
    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.
        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)
 
    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))
 
 
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
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# 添加 APConv 类
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
            nn.Conv2d(in_channels, in_channels, (1,5), stride=stride, padding=(0,2), groups=g, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 5x1 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, (5,1), stride=1, padding=(2,0), groups=g, bias=False),
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
            nn.Conv2d(in_channels, in_channels, (1,3), stride=stride, padding=(0,1), groups=g, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 3x1 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, (3,1), stride=1, padding=(1,0), groups=g, bias=False),
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
 
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
 
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
 
    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
 
class Self_Attn(nn.Module):
    def __init__(
            self,
            dim,
            downsample = False,
            activation=nn.ReLU(inplace=True)
    ):
        super().__init__()
        dim_out = dim
        # shortcut
        proj_factor = 1
        self.stride = 2 if downsample else 1
 
        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()
 
        # contraction and expansion
 
        attn_dim_in = dim_out // proj_factor
        # attn_dim_out = heads * dim_head
        attn_dim_out = attn_dim_in
 
        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias=False),
            nn.BatchNorm2d(attn_dim_in),
            activation,
            ATT(attn_dim_in),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)
        )
 
        # init last batch norm gamma to zero
 
        nn.init.zeros_(self.net[-1].weight)
 
        # final activation
 
        self.activation = activation
 
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.interpolate(x, size=(int(x.size(2)) // 2, int(x.size(3)) // 2), mode='bilinear', align_corners=True)
        out = self.net(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out += shortcut
        return self.activation(out)
 
 
class ATT(nn.Module):
    """ Self attention Layer"""
 
    def __init__(self, in_dim):
        super(ATT, self).__init__()
        self.chanel_in = in_dim
 
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax = nn.Softmax(dim=-1)  #
 
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
 
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
 
        out = self.gamma * out + x
        return out
 
 
# 修改 Detect_SA 中的卷积为 APConv
class Detect_SA(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=2, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(APConv(x, c2, stride=1),  Self_Attn(c2), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(APConv(x, c3, stride=1), Self_Attn(c3),  nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
 
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
 
        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size
 
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
 
    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
 
 
 
if __name__ == "__main__":
    # Generating Sample image
    image1 = (1, 64, 32, 32)
    image2 = (1, 128, 16, 16)
    image3 = (1, 256, 8, 8)
 
    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image = [image1, image2, image3]
    channel = (64, 128, 256)
    # Model
    mobilenet_v1 = Detect_SA(nc=2, ch=channel)
 
    out = mobilenet_v1(image)
    print(out)
 
