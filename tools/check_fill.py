"""全幅稳定(相邻帧补缺口)的 GPU 输出 vs CPU 参考: 同一帧、同一批矩阵, 比较整体与缺口区域, 并可保存对比图供目视。

流程: 逐帧估计运动 → 取第 N 帧的校正量与相邻帧映射 → 把 N±16 帧传进纹理池 → GPU 绘制 → 用 OpenCV 按同样的级联羽化逻辑算参考。
用法: python tools/check_fill.py 视频文件 [--frame N] [--sigma 0.6] [--range 1.15] [--out 对比图.png] [--identity]
  --identity  用恒等校正(无缺口): 检查画面最外圈像素不会被无谓地混入相邻帧。
图(--out): 左=GPU 输出(缩小), 右=缺口标注(红=相邻帧补, 蓝=边缘延伸); 另存 *_zoom.png 为缺口边缘 1:1 放大。
"""
import argparse
import math
import os
import sys

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av
import cv2
import moderngl
import numpy as np
import pygame

import stabcore
import stabplayer as sp

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("video")
ap.add_argument("--frame", type=int, default=100)
ap.add_argument("--sigma", type=float, default=0.6, help="平滑 σ(秒)")
ap.add_argument("--range", type=float, default=1.15, dest="zoom", help="校正范围倍数")
ap.add_argument("--out", help="保存目视对比图 PNG")
ap.add_argument("--identity", action="store_true")
args = ap.parse_args()
I = args.frame

info = sp.probe(args.video)
W, H = info.width, info.height
c = av.open(args.video)
vs = c.streams.video[0]
vs.thread_type = "AUTO"
est = stabcore.MotionEstimator(W, H, 960, info.bits)
traj = stabcore.Trajectory()
frames = {}
for n, f in enumerate(c.decode(vs)):
    planes = sp.frame_planes(f, info.bits)
    z, t, ok = est.push(planes[0])
    traj.append(z, t)
    if I - sp.FILL_RADIUS <= n <= I + sp.FILL_RADIUS:
        frames[n] = tuple(p.copy() for p in planes)
    if n >= I + sp.FILL_RADIUS + 70:
        break
if I not in frames:
    sys.exit("视频帧数不够")
traj.eof = True
stab = stabcore.Stabilizer(traj, W, H, info.fps, max_lookahead=400)
stab.configure(args.sigma, args.zoom)

sp.enable_dpi_awareness()
pygame.init()
for k, v in ((pygame.GL_CONTEXT_MAJOR_VERSION, 3), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
             (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)):
    pygame.display.gl_set_attribute(k, v)
pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
ctx = moderngl.create_context()
r = sp.Renderer(ctx, info)
print("纹理池可用:", r.pool_ok)
for idx, pl in frames.items():
    r.pool_put(idx, pl)
r.upload(frames[I])

VIEW = (W, H)
z, t = stab.correction(I)
m = stabcore.view_matrix(z, t, W, H, view=VIEW)
if args.identity:
    m = stabcore.view_matrix(1 + 0j, 0j, W, H, view=VIEW)
nb = []
for d, a, b in stab.neighbor_maps(I, z, t, sp.FILL_OFFSETS):
    if r.pool_has(I + d):
        nb.append(((I + d) % sp.POOL_LAYERS, stabcore.forward_matrix(a, b, W, H, VIEW), d, a, b))
print(f"校正平移 ({t.real:.1f},{t.imag:.1f}) px, 转角 {math.degrees(math.atan2(z.imag, z.real)):.3f}°; 可用相邻帧 {len(nb)} 个")

fbo = ctx.simple_framebuffer((W, H))
fbo.use()
ctx.clear(0, 0, 0, 1)
r.draw((0, 0, W, H), m, m, 0.0, 0.0, 2, [(l, mm) for l, mm, *_ in nb])
gl = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(H, W, 3)[::-1].astype(np.float32)


# ---- CPU 参考(与着色器里的 cover_weight / 级联羽化逻辑一致) ----
def to_rgb(planes):
    top = (1 << info.bits) - 1
    y, u, v = (p.astype(np.float32) for p in planes)
    U = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
    V = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
    mat, off, scale = sp.yuv_params(info.colorspace, info.full_range, info.bits, H)
    return np.clip(((np.stack([y, U, V], -1) / top - np.array(off)) * np.array(scale)) @ mat.T, 0, 1) * 255


S_OUT_INV = np.array([[1 / W, 0, 0.5 / W], [0, 1 / H, 0.5 / H], [0, 0, 1]])
S_SRC = np.array([[W, 0, -0.5], [0, H, -0.5], [0, 0, 1]])


def warp(img, m_uv, flags):
    pm = S_SRC @ m_uv @ S_OUT_INV
    return cv2.warpAffine(img, pm[:2], (W, H), flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


ones = np.full((H, W), 255, np.uint8)


def mask(m_uv):
    pm = S_SRC @ m_uv @ S_OUT_INV
    return cv2.warpAffine(ones, pm[:2], (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


uu = (np.arange(W) + 0.5) / W
vv = (np.arange(H) + 0.5) / H


def cover_w(mm, cov):
    """帧外 0; 只有边缘落在输出画面内的那一侧才在 FEATHER_PX 宽度内渐变。"""
    sx = mm[0, 0] * uu[None, :] + mm[0, 1] * vv[:, None] + mm[0, 2]
    sy = mm[1, 0] * uu[None, :] + mm[1, 1] * vv[:, None] + mm[1, 2]
    sl, sr = (mm[0, 1] * vv + mm[0, 2])[:, None], (mm[0, 0] + mm[0, 1] * vv + mm[0, 2])[:, None]
    st, sb = (mm[1, 0] * uu + mm[1, 2])[None, :], (mm[1, 0] * uu + mm[1, 1] + mm[1, 2])[None, :]

    def ss(x, e0, e1):
        x = np.clip((x - e0) / (e1 - e0), 0, 1)
        return x * x * (3 - 2 * x)

    fx, fy, gx, gy = sp.FEATHER_PX / W, sp.FEATHER_PX / H, 2.0 / W, 2.0 / H
    w = ((1 - ss(-sl, 0, gx) * (1 - ss(sx, 0, fx))) * (1 - ss(sr - 1, 0, gx) * (1 - ss(1 - sx, 0, fx))) *
         (1 - ss(-st, 0, gy) * (1 - ss(sy, 0, fy))) * (1 - ss(sb - 1, 0, gy) * (1 - ss(1 - sy, 0, fy))))
    return w * cov


cur_w = warp(to_rgb(frames[I]), m, cv2.INTER_CUBIC)
covered = mask(m)
wcur = cover_w(m, covered)
need = wcur < 1                                 # 需要级联的像素(缺口 + 当前帧里确有接缝的羽化带)
ref = cur_w.copy()                              # 底色: 延伸边缘(BORDER_REPLICATE 的当前帧)
any_nb = np.zeros((H, W), bool)
for k in range(len(nb) - 1, -1, -1):
    layer, mm, d, a_, b_ = nb[k]
    nmask = mask(mm)
    w = (cover_w(mm, nmask) * need)[..., None]
    ref = ref * (1 - w) + warp(to_rgb(frames[I + d]), mm, cv2.INTER_LINEAR) * w
    any_nb |= nmask & ~covered
wc3 = wcur[..., None]
ref = ref * (1 - wc3) + cur_w * wc3
hole = ~covered
filled = hole & any_nb
band = covered & need

failed = False
if args.identity:
    ring = np.zeros((H, W), bool)
    ring[:20] = ring[-20:] = True
    ring[:, :20] = ring[:, -20:] = True
    diff = np.abs(gl - cur_w).max(-1)
    changed = (diff > 3)[ring].mean() * 100
    print(f"恒等校正(无缺口): 最外 20px 圈内 GPU 与单帧参考 MAE {np.abs(gl - cur_w)[ring].mean():.3f}, 与单帧结果明显不同的像素占 {changed:.2f}%")
    failed |= changed > 0
print(f"缺口面积 {hole.mean() * 100:.2f}% ; 其中被相邻帧补上 {filled.sum() / max(hole.sum(), 1) * 100:.1f}% ; "
      f"剩余 {(hole & ~filled).mean() * 100:.3f}% 用边缘延伸 ; 羽化带 {band.mean() * 100:.2f}%")
d_all = np.abs(gl - ref)
print(f"GPU vs CPU 参考: 整幅平均绝对误差 {d_all.mean():.3f}  p99 {np.percentile(d_all, 99):.1f}  (期望 平均<0.8)")
failed |= d_all.mean() >= 0.8
if hole.any():
    nan = float("nan")
    print(f"  缺口区域内: 平均 {d_all[hole].mean():.3f}  (相邻帧补的部分 {d_all[filled].mean() if filled.any() else nan:.3f}; "
          f"边缘延伸部分 {d_all[hole & ~filled].mean() if (hole & ~filled).any() else nan:.3f})")

if args.out:
    vis = gl.astype(np.uint8).copy()
    vis[filled] = (0.6 * vis[filled] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    vis[hole & ~filled] = (0.4 * vis[hole & ~filled] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    sc_ = 960 / W
    cv2.imwrite(args.out, cv2.cvtColor(np.hstack([cv2.resize(gl.astype(np.uint8), None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA),
                                                  cv2.resize(vis, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA)]), cv2.COLOR_RGB2BGR))
    if hole.any() and W >= 500 and H >= 300:
        ys, xs = np.nonzero(hole)
        cy, cx = int(np.clip(np.median(ys), 150, H - 150)), int(np.clip(np.median(xs), 250, W - 250))
        crop = np.hstack([gl[cy - 150:cy + 150, cx - 250:cx + 250], vis[cy - 150:cy + 150, cx - 250:cx + 250]]).astype(np.uint8)
        cv2.imwrite(os.path.splitext(args.out)[0] + "_zoom.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
print("失败" if failed else "通过")
sys.exit(1 if failed else 0)
