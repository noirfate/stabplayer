"""相邻帧映射的几何校验: 把相邻帧按 Stabilizer.neighbor_maps 算出的矩阵映射到当前帧的输出坐标, 与当前帧在重叠区域比较。
正确映射的差异应明显小于: ① 错误映射(位姿取反 / 忽略平滑) ② 完全不做运动补偿。改了 stabcore 的坐标系/矩阵推导后跑一遍。
用法: python tools/check_fill_geometry.py 视频文件 [--frames 420]    (仅 8-bit; 视频需 ≥ 120 帧, 手持抖动明显的素材才有区分度)
"""
import argparse
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
ap.add_argument("--frames", type=int, default=420, help="最多分析多少帧")
args = ap.parse_args()

info = sp.probe(args.video)
if info.bits != 8:
    sys.exit("本检查只支持 8-bit 视频")
W, H = info.width, info.height
c = av.open(args.video)
vs = c.streams.video[0]
vs.thread_type = "AUTO"
est = sc.MotionEstimator(W, H)
traj = sc.Trajectory()
small = []
for n, f in enumerate(c.decode(vs)):
    pl = f.planes[0]
    y = np.frombuffer(pl, np.uint8).reshape(-1, pl.line_size)[:H, :W]
    z, t, ok = est.push(y)
    traj.append(z, t)
    small.append(est.prev.copy())
    if n + 1 >= args.frames:
        break
traj.eof = True
if len(small) < 120:
    sys.exit("视频帧数不足 120")
ow, oh = est.size
S_OUT_INV = np.array([[1 / ow, 0, 0.5 / ow], [0, 1 / oh, 0.5 / oh], [0, 0, 1]])
S_SRC = np.array([[ow, 0, -0.5], [0, oh, -0.5], [0, 0, 1]])


def warp(img, m_uv, fill=0):
    p = S_SRC @ m_uv @ S_OUT_INV
    return cv2.warpAffine(img, p[:2], (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


ones = np.full((oh, ow), 255, np.uint8)
st = sc.Stabilizer(traj, W, H, info.fps, max_lookahead=400)
st.configure(0.6, 1.15)
VIEW = (W, H)
offs = (-4, 4, -8, 8, -16, 16)
rows = {"正确映射": [], "错误: 取反偏移": [], "不补偿运动(同 uv)": [], "错误: 忽略平滑 H": []}
cover = []
for i in range(40, len(small) - 40, 5):
    z_w, t_w = st.correction(i)
    cur = warp(small[i], sc.view_matrix(z_w, t_w, W, H, view=VIEW))
    cmask = warp(ones, sc.view_matrix(z_w, t_w, W, H, view=VIEW)) > 250
    for d, a, b in st.neighbor_maps(i, z_w, t_w, offs):
        j = i + d
        nb = warp(small[j], sc.forward_matrix(a, b, W, H, VIEW))
        nmask = warp(ones, sc.forward_matrix(a, b, W, H, VIEW)) > 250
        ov = cv2.erode((cmask & nmask).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if ov.mean() < 0.2:
            continue

        def mae(img):
            return float(np.abs(img.astype(np.float32) - cur.astype(np.float32))[ov].mean())

        rows["正确映射"].append(mae(nb))
        zg, tg = traj.pose(i)                       # 对照 1: 用 i-d 的位姿算 j=i+d 的映射(方向取反)
        zj, tj = traj.pose(i - d)
        zh, th = z_w * zg, z_w * tg + t_w
        a2 = zj / zh
        rows["错误: 取反偏移"].append(mae(warp(small[j], sc.forward_matrix(a2, tj - a2 * th, W, H, VIEW))))
        rows["不补偿运动(同 uv)"].append(mae(warp(small[j], sc.view_matrix(1 + 0j, 0j, W, H, view=VIEW))))
        zj2, tj2 = traj.pose(j)                     # 对照 3: 把 H_i 当作 G_i(忽略平滑造成的校正)
        a3 = zj2 / zg
        rows["错误: 忽略平滑 H"].append(mae(warp(small[j], sc.forward_matrix(a3, tj2 - a3 * tg, W, H, VIEW))))
    cover.append((~cmask).mean())
print(f"评估 {len(cover)} 帧, 当前帧自身缺口平均 {np.mean(cover) * 100:.1f}% (全幅不裁切, σ0.6 校正范围 1.15)")
means = {k: float(np.mean(v)) for k, v in rows.items() if v}
for k, v in means.items():
    print(f"  {k:<22} 重叠区域平均绝对差 {v:6.2f} 灰度级  (样本 {len(rows[k])})")
if not means:
    sys.exit("没有足够的重叠样本")
good = means["正确映射"]
worst_ratio = good / min(v for k, v in means.items() if k != "正确映射")
ok = worst_ratio < 0.7
print(f"{'通过' if ok else '失败'}: 正确映射的差异是最好的错误对照的 {worst_ratio * 100:.0f}% (期望 < 70%)")
sys.exit(0 if ok else 1)
