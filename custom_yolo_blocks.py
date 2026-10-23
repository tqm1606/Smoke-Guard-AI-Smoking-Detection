import torch
import torch.nn as nn

# ============================================================
# Stronger backbone blocks for thin/small objects.
# File .py riĂªng Ä‘á»ƒ Ultralytics DDP subprocess import Ä‘Æ°á»£c khi train 2 GPU.
# ============================================================

# ============================================================
# Stronger backbone blocks for thin/small objects.
# KhĂ¡c báº£n cÅ©: táº¥t cáº£ block Ä‘á»u residual + gamma=0,
# nĂªn lĂºc khá»Ÿi táº¡o gáº§n nhÆ° identity, Ă­t phĂ¡ pretrained YOLO11s.
# ============================================================

class LSKABlock(nn.Module):
    """Large Selective Kernel Attention.
    DĂ¹ng depthwise conv dáº¡ng dáº£i dĂ i Ä‘á»ƒ báº¯t ngá»¯ cáº£nh dĂ i/máº£nh cá»§a cigarette/vape.
    Residual gamma=0 Ä‘á»ƒ khá»Ÿi táº¡o khĂ´ng lĂ m há»ng pretrained features.
    """
    def __init__(self, c, k=7, dilated_k=11, dilation=2):
        super().__init__()
        pad1 = k // 2
        pad2 = ((dilated_k - 1) // 2) * dilation
        self.dw_local = nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c, bias=False)
        self.dw_h = nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, pad1), groups=c, bias=False)
        self.dw_w = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(pad1, 0), groups=c, bias=False)
        self.dw_dh = nn.Conv2d(c, c, kernel_size=(1, dilated_k), padding=(0, pad2), dilation=(1, dilation), groups=c, bias=False)
        self.dw_dw = nn.Conv2d(c, c, kernel_size=(dilated_k, 1), padding=(pad2, 0), dilation=(dilation, 1), groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn = self.dw_local(x)
        attn = self.dw_h(attn)
        attn = self.dw_w(attn)
        attn = self.dw_dh(attn)
        attn = self.dw_dw(attn)
        attn = self.bn(self.pw(attn))
        branch = x * torch.sigmoid(attn)
        return x + self.gamma * (branch - x)


class LightCBAM(nn.Module):
    def __init__(self, c, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        mx = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * torch.sigmoid(avg + mx)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))


class MSCBAMBlock(nn.Module):
    """Multi-Scale Context + CBAM.
    Parallel depthwise 3/5/7 giĂºp giá»¯ cáº£ chi tiáº¿t nhá» láº«n context xung quanh tay/miá»‡ng/khĂ³i.
    """
    def __init__(self, c):
        super().__init__()
        self.dw3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.dw7 = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
        self.pw = nn.Conv2d(c * 3, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.cbam = LightCBAM(c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        branch = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        branch = self.act(self.bn(self.pw(branch)))
        branch = self.cbam(branch)
        return x + self.gamma * branch


class CAAStripBlock(nn.Module):
    """Context Anchor / Strip Attention.
    Táº­p trung vĂ o cáº¥u trĂºc dĂ i theo ngang/dá»c, há»£p object máº£nh nhÆ° Ä‘iáº¿u thuá»‘c/vape.
    """
    def __init__(self, c, k=11):
        super().__init__()
        pad = k // 2
        self.local = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.conv_h = nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, pad), groups=c, bias=False)
        self.conv_w = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(pad, 0), groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        context = self.pool(self.local(x))
        gate = self.conv_h(context)
        gate = self.conv_w(gate)
        gate = torch.sigmoid(self.bn(self.pw(gate)))
        branch = x * gate
        return x + self.gamma * (branch - x)


def _choose_groups(c, max_groups=8):
    for g in [max_groups, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1


class EMABlock(nn.Module):
    """Efficient Multi-scale Attention style block.
    Nháº¹ hÆ¡n full self-attention, dĂ¹ng grouped spatial-channel gating.
    """
    def __init__(self, c, max_groups=8):
        super().__init__()
        self.groups = _choose_groups(c, max_groups)
        cg = c // self.groups
        self.conv1 = nn.Conv2d(cg, cg, 1, bias=False)
        self.conv3 = nn.Conv2d(cg, cg, 3, padding=1, groups=cg, bias=False)
        self.gn = nn.GroupNorm(1, cg)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        xg = x.reshape(b * g, c // g, h, w)

        hp = xg.mean(dim=3, keepdim=True)
        wp = xg.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        hw = self.conv1(torch.cat([hp, wp], dim=2))
        hp, wp = torch.split(hw, [h, w], dim=2)
        wp = wp.permute(0, 1, 3, 2)

        gated = self.gn(xg * torch.sigmoid(hp) * torch.sigmoid(wp))
        local = self.conv3(xg)
        spatial_gate = torch.sigmoid((gated + local).mean(dim=1, keepdim=True))
        out = (xg * spatial_gate).reshape(b, c, h, w)
        return x + self.gamma * (out - x)


class EMALSKABlock(nn.Module):
    """EMA + LSKA stack. Máº¡nh hÆ¡n nhÆ°ng váº«n identity-init."""
    def __init__(self, c):
        super().__init__()
        self.ema = EMABlock(c)
        self.lska = LSKABlock(c)

    def forward(self, x):
        return self.lska(self.ema(x))

# PyTorch >= 2.6 / Ultralytics safe-load compatibility.
try:
    import torch.serialization as _serialization
    _serialization.add_safe_globals([LSKABlock, LightCBAM, MSCBAMBlock, CAAStripBlock, EMABlock, EMALSKABlock])
except Exception:
    pass

__all__ = ["LSKABlock", "LightCBAM", "MSCBAMBlock", "CAAStripBlock", "EMABlock", "EMALSKABlock"]