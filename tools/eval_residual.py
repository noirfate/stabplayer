"""评估去抖效果: 整段视频、逐时间段比较「原始 vs 各档预设」的残余抖动。

参照量用画面里一块基本没有运动前景的区域(默认上部 35%, 如幕布/天空/墙面)重新估计运动, 与全局估计器的特征点集合不同,
不受前景干扰; 残余抖动 = 该路径去掉 σ=0.25s 高斯平滑后的 RMS(换算成 4K 像素), 反映 ≳0.5Hz 的抖动。
数值 <1 表示比原始更稳; 某时间段 >1 表示那段比原始更抖(会标出)。
用法: python tools/eval_residual.py 视频文件 [--strip 0.35] [--max-frames 700]
  --max-frames 限制分析帧数, 可用来排除片尾相机甩向无纹理区域、估计本就失效的段落。仅 8-bit 视频; 耗时约等于解码+运动估计一遍。
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av
import cv2
import numpy as np

import stabcore as sc
import stabplayer as sp

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("video")
ap.add_argument("--strip", type=float, default=0.35, help="作为参照的画面上部比例 (默认 0.35)")
ap.add_argument("--max-frames", type=int, default=700)
args = ap.parse_args()

info = sp.probe(args.video)
if info.bits != 8:
    sys.exit("本评估只支持 8-bit 视频")
W, H, fps = info.width, info.height, info.fps
c = av.open(args.video)
vs = c.streams.video[0]
vs.thread_type = "AUTO"
est = sc.MotionEstimator(W, H)
traj = sc.Trajectory()
small = []
for f in c.decode(vs):
    pl = f.planes[0]
    y = np.frombuffer(pl, np.uint8).reshape(-1, pl.line_size)[:H, :W]
    z, t, ok = est.push(y)
    traj.append(z, t)
    small.append(est.prev.copy())
    if len(small) >= args.max_frames:
        break
traj.eof = True
N = len(small)
ow, oh = est.size
S_OUT_INV = np.array([[1 / ow, 0, 0.5 / ow], [0, 1 / oh, 0.5 / oh], [0, 0, 1]])
S_SRC = np.array([[ow, 0, -0.5], [0, oh, -0.5], [0, 0, 1]])


def warp(img, m_uv):
    p = S_SRC @ m_uv @ S_OUT_INV
    return cv2.warpAffine(img, p[:2], (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


STRIP = int(oh * args.strip)


def strip_path(frames):
    e = sc.MotionEstimator(ow, STRIP)
    tr = sc.Trajectory()
    for fr in frames:
        z, t, _ = e.push(np.ascontiguousarray(fr[:STRIP]))
        tr.append(z, t)
    return tr.p[: tr.n]


HALF = int(round(fps * 0.25 * 3))              # 高斯核半宽 = 3σ


def residual(p, seg):
    """路径去掉 σ=0.25s 的高斯平滑后的 RMS(原始分辨率像素)。"""
    half = HALF
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / (0.25 * fps)) ** 2)
    k /= k.sum()
    out = []
    for c_ in (0, 1):
        x = np.pad(p[:, c_], half, mode="reflect")
        out.append((p[:, c_] - np.convolve(x, k, mode="valid"))[seg[0]:seg[1]])
    return (W / ow) * math.sqrt(np.mean(np.concatenate(out) ** 2))


CONFIGS = [("原始", None, 1.05)] + [(f"{i + 1} {name} σ{sig:.2f} 范围{zoom:.2f}", sig, zoom) for i, (name, sig, zoom) in enumerate(sp.PRESETS)]
paths, lim = {}, {}
for name, sig, zoom in CONFIGS:
    st = sc.Stabilizer(traj, W, H, fps, max_lookahead=400)
    if sig:
        st.configure(sig, zoom)
    outs, ls = [], []
    for i in range(N):
        if sig:
            z, t = st.correction(i)
            ls.append(st.limit)
        else:
            z, t = 1 + 0j, 0j
        outs.append(warp(small[i], sc.view_matrix(z, t, W, H, zoom)))
    paths[name] = strip_path(outs)
    lim[name] = float(np.mean(np.array(ls) > 0.02)) if ls else 0.0

seg_len = int(round(fps * 2.5))
segs = [(a, min(a + seg_len, N)) for a in range(0, N, seg_len)]
print(f"{os.path.basename(args.video)}: {W}x{H} {fps:.2f}fps, 分析 {N} 帧; 参照区域 = 画面上部 {args.strip * 100:.0f}%; 单位: 像素 RMS")
print(f"{'配置':<28}" + "".join(f"{a / fps:4.0f}-{b / fps:<4.0f}" for a, b in segs) + "   全程   受限帧%")
base = paths[CONFIGS[0][0]]
row = [residual(base, s) for s in segs]
base_all = residual(base, (0, N))
print(f"{CONFIGS[0][0]:<28}" + "".join(f"{v:9.1f}" for v in row) + f"{base_all:8.1f}")
for name, sig, zoom in CONFIGS[1:]:
    p_ = paths[name]
    vals = [residual(p_, s) for s in segs]
    ratio = [v / max(b, 1e-6) for v, b in zip(vals, row)]
    print(f"{name:<28}" + "".join(f"{v:9.1f}" for v in vals) + f"{residual(p_, (0, N)):8.1f}   {lim[name] * 100:5.1f}")
    print(f"{'   相对原始':<28}" + "".join(f"{r:9.2f}" for r in ratio) + f"{residual(p_, (0, N)) / base_all:8.2f}"
          + ("   <-- 有时间段比原始更抖" if any(r > 1.0 for r in ratio) else ""))
