"""Speed regressors compared in this project (input [B,T,H,W] → speed [B]).

  single2d  2D-CNN on the last frame only                 (lower bound: speed is not visible in one frame)
  early2d   2D-CNN, the T frames stacked as input channels (time is fused in the first layer)
  early2d_wide  early2d made wider so its compute (MACs) matches c3d at T=8
  c3d       3D-CNN, convolutions over (time, y, x)
  cnn_lstm / convlstm / r2p1d / tsm / vivit: other sequence models (see below)

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


# --------------------------------------------------------------------------------------------------
# Other ways to use an image sequence (all ~0.87-0.89 M parameters, like the models above)
#
#   cnn_lstm  shared 2D-CNN per frame → global feature vector per frame → LSTM over time (late fusion)
#   convlstm  shared 2D-CNN stem (1/8 resolution) → ConvLSTM over time keeps the spatial layout → 2D-CNN
#   r2p1d     (2+1)D CNN: each 3D conv factorized into a spatial 1x3x3 and a temporal 3x1x1 conv
#   tsm       Temporal Shift Module: per-frame 2D-CNN, 1/8 of the channels shifted ±1 frame in each block
#   vivit     Video Vision Transformer (factorized encoder): spatial transformer per frame, then a
#             temporal transformer over the frame tokens


class FrameEncoder(nn.Module):
    """2D-CNN applied to every frame with shared weights: [B,T,H,W] → [B,T,C,h,w]."""

    def __init__(self, widths, pool=True):
        super().__init__()
        cin, layers = 1, []
        for w in widths:
            layers.append(_block2d(cin, w))
            cin = w
        if pool:
            layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)
        self.out_ch = cin

    def forward(self, x):
        B, T = x.shape[:2]
        y = self.net(x.reshape(B * T, 1, *x.shape[2:]))
        return y.reshape(B, T, *y.shape[1:])


class CNNLSTM(nn.Module):
    channels_last_ok = True

    def __init__(self, widths=(24, 48, 80, 112, 152), hidden=144):
        super().__init__()
        self.enc = FrameEncoder(widths)
        self.lstm = nn.LSTM(self.enc.out_ch, hidden, batch_first=True)
        self.head = _head(hidden)

    def forward(self, x):
        f = self.enc(x).flatten(2)              # [B,T,C]
        out, _ = self.lstm(f.float())
        return self.head(out[:, -1]).squeeze(1)


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(cin + hid, 4 * hid, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMNet(nn.Module):
    channels_last_ok = True

    def __init__(self, stem=(32, 64, 80), hid=48, tail=(112, 144)):
        super().__init__()
        self.enc = FrameEncoder(stem, pool=False)
        self.cell = ConvLSTMCell(self.enc.out_ch, hid)
        layers, cin = [], hid
        for w in tail:
            layers.append(_block2d(cin, w))
            cin = w
        self.tail = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = _head(cin)

    def forward(self, x):
        f = self.enc(x)                          # [B,T,C,h,w]
        B, T, _, h, w = f.shape
        hc = (f.new_zeros(B, self.cell.hid, h, w), f.new_zeros(B, self.cell.hid, h, w))
        for t in range(T):
            hc = self.cell(f[:, t], hc)
        return self.head(self.tail(hc[0])).squeeze(1)


def _block2p1d(cin, cout, t_stride, mid_ratio=1.0):
    def conv(ci, co, ts, s):
        mid = max(8, int(round(mid_ratio * 27 * ci * co / (9 * ci + 3 * co))))   # R(2+1)D: same #params as 3x3x3
        return [nn.Conv3d(ci, mid, (1, 3, 3), stride=(1, s, s), padding=(0, 1, 1), bias=False), nn.BatchNorm3d(mid),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid, co, (3, 1, 1), stride=(ts, 1, 1), padding=(1, 0, 0), bias=False), nn.BatchNorm3d(co),
                nn.ReLU(inplace=True)]
    return nn.Sequential(*conv(cin, cout, t_stride, 2), *conv(cout, cout, 1, 1))


class R2Plus1D(nn.Module):
    channels_last_ok = False

    def __init__(self, widths=(16, 32, 56, 72, 96), t_strides=(1, 1, 2, 2, 1)):
        super().__init__()
        cin, layers = 1, []
        for w, ts in zip(widths, t_strides):
            layers.append(_block2p1d(cin, w, ts))
            cin = w
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1))
        self.head = _head(cin)

    def forward(self, x):
        return self.head(self.features(x[:, None])).squeeze(1)


class TemporalShift(nn.Module):
    """Shift 1/8 of the channels one frame back and 1/8 one frame forward (zero padded)."""

    def __init__(self, T, div=8):
        super().__init__()
        self.T, self.div = T, div

    def forward(self, x):                        # [B*T,C,H,W]
        bt, c, h, w = x.shape
        x = x.reshape(bt // self.T, self.T, c, h, w)
        k = c // self.div
        out = torch.zeros_like(x)
        out[:, 1:, :k] = x[:, :-1, :k]           # past → present
        out[:, :-1, k:2 * k] = x[:, 1:, k:2 * k]  # future → present (within the window: all frames are past)
        out[:, :, 2 * k:] = x[:, :, 2 * k:]
        return out.reshape(bt, c, h, w)


class TSM(nn.Module):
    channels_last_ok = True

    def __init__(self, T, widths=(32, 64, 96, 128, 160)):
        super().__init__()
        self.T = T
        cin, layers = 1, []
        for w in widths:
            layers += [TemporalShift(T), _block2d(cin, w)] if cin > 1 else [_block2d(cin, w)]
            cin = w
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = _head(cin)

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.features(x.reshape(B * T, 1, *x.shape[2:])).reshape(B, T, -1)
        return self.head(f.mean(1)).squeeze(1)


class ViViT(nn.Module):
    """Factorized encoder (ViViT model 2): spatial transformer per frame → temporal transformer."""
    channels_last_ok = False

    def __init__(self, T, H=140, W=180, patch=10, dim=108, depth_s=4, depth_t=2, heads=4):
        super().__init__()
        self.patch = patch
        n = (H // patch) * (W // patch)
        self.embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.cls_s = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_s = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.cls_t = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_t = nn.Parameter(torch.randn(1, T + 1, dim) * 0.02)

        def enc(depth):
            layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.spatial, self.temporal = enc(depth_s), enc(depth_t)
        self.norm = nn.LayerNorm(dim)
        self.head = _head(dim)

    def forward(self, x):
        B, T, H, W = x.shape
        p = self.patch
        x = x[:, :, :H // p * p, :W // p * p].reshape(B * T, 1, H // p * p, W // p * p)
        tok = self.embed(x).flatten(2).transpose(1, 2)                     # [B*T,N,D]
        tok = torch.cat([self.cls_s.expand(B * T, -1, -1), tok], 1) + self.pos_s
        fr = self.spatial(tok)[:, 0].reshape(B, T, -1)                     # one token per frame
        fr = torch.cat([self.cls_t.expand(B, -1, -1), fr], 1) + self.pos_t
        return self.head(self.norm(self.temporal(fr)[:, 0])).squeeze(1)


def build(name: str, T: int) -> nn.Module:
    other = {"cnn_lstm": lambda: CNNLSTM(), "convlstm": lambda: ConvLSTMNet(), "r2p1d": lambda: R2Plus1D(),
             "tsm": lambda: TSM(T), "vivit": lambda: ViViT(T)}
    if name in other:
        return other[name]()
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
            macs += out.numel() * k             # batch of 1 sample (per-frame models fold T into the batch)
        elif isinstance(mod, nn.Linear):        # applied to every token
            macs += out.numel() * mod.in_features
        elif isinstance(mod, nn.LSTM):
            seq = inp[0].shape[1]
            macs += 4 * mod.hidden_size * (mod.input_size + mod.hidden_size) * seq
        elif isinstance(mod, nn.TransformerEncoderLayer):   # attention projections and products (FFN = Linear)
            bn, n, d = inp[0].shape
            macs += bn * (4 * n * d * d + 2 * n * n * d)

    kinds = (nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM, nn.TransformerEncoderLayer)
    hs = [mm.register_forward_hook(hook) for mm in m.modules() if isinstance(mm, kinds)]
    was_training = m.training
    with torch.no_grad():
        m.eval()(torch.zeros(1, T, H, W, device=next(m.parameters()).device))
    m.train(was_training)
    for h in hs:
        h.remove()
    return macs
