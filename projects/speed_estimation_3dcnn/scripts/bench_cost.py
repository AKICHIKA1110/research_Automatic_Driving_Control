"""Compute cost of each model: sequential (streaming) inference and training.

Sequential inference: a new frame arrives every 0.1 s and the speed is estimated from the latest T frames.
  window  - run the model on the whole T-frame window again for every new frame
  stream  - keep what does not depend on the window position and compute only the new part.
            Exact (same output as `window`) for models whose per-frame part is independent of the other
            frames: cnn_lstm / convlstm (2D-CNN per frame), vivit (spatial transformer per frame).
            The others mix frames from the first layer (early2d), use zero padding / strides over time inside
            the window (c3d, r2p1d) or shift features both ways in time (tsm): they must recompute the window.
Training: forward + backward of a batch of 128 windows (bf16 autocast, as in train.py), peak GPU memory.

  python scripts/bench_cost.py --out results/report_cost
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speedest.models import build, count_macs, count_params  # noqa: E402

MODELS = ["single2d", "early2d", "early2d_wide", "c3d", "r2p1d", "tsm", "cnn_lstm", "convlstm", "vivit"]
T, H, W = 8, 140, 180


class Stream:
    """Sequential inference with a cache of per-frame features (exact for cnn_lstm / convlstm / vivit)."""

    def __init__(self, name, m):
        self.name, self.m, self.buf = name, m, collections.deque(maxlen=T)

    def frame_feat(self, x1):                      # x1 [1,1,H,W] → feature of that frame
        m = self.m
        if self.name in ("cnn_lstm", "convlstm"):
            return m.enc(x1)[:, 0]
        p = m.patch                                 # vivit: spatial transformer → frame token
        x1 = x1[:, :, :H // p * p, :W // p * p]
        tok = m.embed(x1).flatten(2).transpose(1, 2)
        tok = torch.cat([m.cls_s, tok], 1) + m.pos_s
        return m.spatial(tok)[:, 0]

    def head(self, f):                              # f [1,T,...]
        m = self.m
        if self.name == "cnn_lstm":
            out, _ = m.lstm(f.flatten(2).float())
            return m.head(out[:, -1]).squeeze(1)
        if self.name == "convlstm":
            B, _, _, h, w = f.shape
            hc = (f.new_zeros(B, m.cell.hid, h, w), f.new_zeros(B, m.cell.hid, h, w))
            for t in range(f.shape[1]):
                hc = m.cell(f[:, t], hc)
            return m.head(m.tail(hc[0])).squeeze(1)
        fr = torch.cat([m.cls_t, f], 1) + m.pos_t
        return m.head(m.norm(m.temporal(fr)[:, 0])).squeeze(1)

    def step(self, x1):
        self.buf.append(self.frame_feat(x1))
        if len(self.buf) < T:                       # not enough frames yet
            return None
        return self.head(torch.stack(list(self.buf), 1))

    def frame_macs(self):
        """MACs of computing one frame's feature (counted with the same hooks as count_macs)."""
        return _macs(lambda x: self.frame_feat(x[:, :1]), self.m, 1)


def _macs(fn, m, t):
    """count_macs for an arbitrary function of the model's input."""
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, x):
            return fn(x)
    return count_macs(Wrap(), t, H, W)


def timeit(fn, n, warm=5):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench(name, threads):
    m = build(name, T).eval()
    r = dict(model=name, params=count_params(m), macs_window=count_macs(m, T, H, W))
    st = Stream(name, m) if name in ("cnn_lstm", "convlstm", "vivit") else None
    if st:
        fm = st.frame_macs()
        r["macs_frame"] = fm
        r["macs_stream"] = r["macs_window"] - (T - 1) * fm
    else:
        r["macs_stream"] = r["macs_window"]

    # equivalence check of the streaming implementation
    seq = torch.rand(1, T + 4, H, W)
    with torch.inference_mode():
        if st:
            outs = [st.step(seq[:, i:i + 1]) for i in range(seq.shape[1])]
            ref = m(seq[:, -T:])
            r["stream_max_abs_diff"] = float((outs[-1] - ref).abs().max())

        # latency, batch 1: CPU (fp32, `threads` threads) and GPU (fp32)
        torch.set_num_threads(threads)
        x = seq[:, -T:]
        r["cpu_ms_window"] = 1e3 * timeit(lambda: m(x), 30)
        if st:
            frames = [seq[:, i:i + 1] for i in range(seq.shape[1])]
            it = iter(range(10 ** 9))
            r["cpu_ms_stream"] = 1e3 * timeit(lambda: st.step(frames[next(it) % len(frames)]), 30)
        else:
            r["cpu_ms_stream"] = r["cpu_ms_window"]
        mg = m.cuda()
        xg = x.cuda()
        r["gpu_ms_window"] = 1e3 * timeit(lambda: mg(xg), 100)
        if st:
            sg = Stream(name, mg)
            fg = [f.cuda() for f in frames]
            it = iter(range(10 ** 9))
            r["gpu_ms_stream"] = 1e3 * timeit(lambda: sg.step(fg[next(it) % len(fg)]), 100)
        else:
            r["gpu_ms_stream"] = r["gpu_ms_window"]

    # training step: batch 128, bf16 autocast, AdamW, same memory format as train.py
    m = build(name, T).cuda().train()
    if getattr(m, "channels_last_ok", name != "c3d"):
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    xb, yb = torch.rand(128, T, H, W, device="cuda"), torch.rand(128, device="cuda")
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    def train_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(m(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for _ in range(5):                              # warm up (cuDNN autotuning workspaces) before measuring memory
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    r["train_ms_batch"] = 1e3 * timeit(train_step, 30, warm=0)
    r["train_peak_mem_gb"] = (torch.cuda.max_memory_allocated() - base) / 1e9
    r["train_macs_sample"] = 3 * r["macs_window"]          # forward + backward (≈ 2 × forward)
    del m, opt, xb
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/report_cost")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (Raspberry Pi 5 has 4 cores)")
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.models:
        r = bench(name, a.threads)
        print(json.dumps(r), flush=True)
        rows.append(r)
    json.dump(rows, open(out / "cost.json", "w"), indent=1)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "cost.csv", index=False)


if __name__ == "__main__":
    main()
