"""Speed regressors compared in this project (input [B,T,H,W] → speed [B]).

  single2d  2D-CNN on the last frame only                 (lower bound: speed is not visible in one frame)
  early2d   2D-CNN, the T frames stacked as input channels (time is fused in the first layer)
  early2d_wide  early2d made wider so its compute (MACs) matches c3d at T=8
  c3d       3D-CNN, convolutions over (time, y, x)

All share the same 5-stage layout (stride 2 in space each stage, global average pooling, small MLP head)
and single2d / early2d / c3d have close parameter counts (see `count_params`).
"""
from __future__ import annotations

import torch
from torch import nn


def _block2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _block3d(cin, cout, t_stride):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=(t_stride, 2, 2), padding=1, bias=False),
                         nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


def _head(c):
    return nn.Sequential(nn.Flatten(), nn.Linear(c, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))


class Net2D(nn.Module):
    def __init__(self, T: int, single: bool, widths):
        super().__init__()
        self.single = single
        cin = 1 if single else T
        layers = []
        for w in widths:
            layers.append(_block2d(cin, w))
            cin = w
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = _head(cin)

    def forward(self, x):                       # x [B,T,H,W]
        if self.single:
            x = x[:, -1:]
        return self.head(self.features(x)).squeeze(1)


class Net3D(nn.Module):
    def __init__(self, T: int, widths, t_strides):
        super().__init__()
        cin, layers = 1, []
        for w, ts in zip(widths, t_strides):
            layers.append(_block3d(cin, w, ts))
            cin = w
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1))
        self.head = _head(cin)

    def forward(self, x):                       # x [B,T,H,W] -> [B,1,T,H,W]
        return self.head(self.features(x[:, None])).squeeze(1)


def build(name: str, T: int) -> nn.Module:
    if name == "single2d":
        return Net2D(T, True, (32, 64, 96, 128, 160))
    if name == "early2d":
        return Net2D(T, False, (32, 64, 96, 128, 160))
    if name == "early2d_wide":   # same compute (MACs) as c3d at T=8, ~4.5x the parameters
        return Net2D(T, False, (72, 136, 208, 272, 344))
    if name == "c3d":
        # 3x3x3 kernels have 3x the weights of 3x3: narrower layers give a similar parameter count.
        # Time is kept at full rate in the first two stages, then halved (stride 2) in stages 3 and 4;
        # the final global average pooling is over time and space.
        return Net3D(T, (16, 32, 56, 72, 96), (1, 1, 2, 2, 1))
    raise ValueError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_macs(m: nn.Module, T: int, H=140, W=180) -> int:
    """Multiply-accumulates of the conv/linear layers for one sample."""
    macs = 0

    def hook(mod, inp, out):
        nonlocal macs
        if isinstance(mod, (nn.Conv2d, nn.Conv3d)):
            k = mod.weight[0].numel()           # cin/groups * prod(kernel)
            macs += out.numel() // out.shape[0] * k
        elif isinstance(mod, nn.Linear):
            macs += mod.in_features * mod.out_features

    hs = [mm.register_forward_hook(hook) for mm in m.modules() if isinstance(mm, (nn.Conv2d, nn.Conv3d, nn.Linear))]
    with torch.no_grad():
        m.eval()(torch.zeros(1, T, H, W))
    for h in hs:
        h.remove()
    return macs
