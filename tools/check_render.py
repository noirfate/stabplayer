"""GL 渲染正确性: 与 numpy/OpenCV 独立算出的参考图逐像素对比, 改着色器或渲染器后跑一遍。

检查项: (a) 颜色矩阵/量程/纹理方向/位深缩放  (b) 变换(平移+旋转+放大)与符号  (c) 缩小显示的 mip 路径  (d) 1:1 原生像素。
用法: python tools/check_render.py 视频文件 [--frame N] [--out 缩小对比图.png]
需要 OpenGL 3.3 环境; 视频建议 720p~1080p(会开一个与视频同尺寸的窗口)。任一项超出阈值则退出码为 1。
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
ap.add_argument("--frame", type=int, default=10, help="用第几帧做检查 (默认 10)")
ap.add_argument("--out", help="保存「面积平均缩小 | GL 缩小」对比图 PNG")
args = ap.parse_args()

info = sp.probe(args.video)
print(f"{os.path.basename(args.video)}: {info.width}x{info.height} bits={info.bits} cs={info.colorspace} full={info.full_range}")
planes = None
with av.open(args.video) as c:
    vs = c.streams.video[0]
    for i, f in enumerate(c.decode(vs)):
        if i == args.frame:
            planes = sp.frame_planes(f, info.bits)
            break
if planes is None:
    sys.exit(f"视频不足 {args.frame + 1} 帧")
W, H = info.width, info.height
failed = []


def check(name: str, ok: bool, text: str):
    print(("通过 " if ok else "失败 ") + f"{name}: {text}")
    if not ok:
        failed.append(name)


def open_gl(w: int, h: int):
    pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF)
    c = moderngl.create_context()
    rr = sp.Renderer(c, info)
    rr.upload(planes)
    return c, rr


def read(c, w: int, h: int) -> np.ndarray:
    data = c.screen.read(viewport=(0, 0, w, h), components=3, alignment=1)
    return np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1].astype(np.float32)


sp.enable_dpi_awareness()
pygame.init()
for k, v in ((pygame.GL_CONTEXT_MAJOR_VERSION, 3), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
             (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)):
    pygame.display.gl_set_attribute(k, v)
ctx, r = open_gl(W, H)


def render(m, quality, lod=0.0):
    ctx.screen.use()
    ctx.viewport = (0, 0, W, H)
    ctx.clear(0, 0, 0, 1)
    r.draw((0, 0, W, H), m, m, 0.0, lod, quality)
    return read(ctx, W, H)


# 参考: numpy 做 YUV→RGB (色度双线性上采样到全分辨率)
top = (1 << info.bits) - 1
y, u, v = (p.astype(np.float32) for p in planes)
U = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
V = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
mat, off, scale = sp.yuv_params(info.colorspace, info.full_range, info.bits, H)
ref = np.clip(((np.stack([y, U, V], -1) / top - np.array(off)) * np.array(scale)) @ mat.T, 0, 1) * 255

# (a) 恒等变换 + 双线性
ident = stabcore.view_matrix(1 + 0j, 0j, W, H, view=(W, H))
d = np.abs(render(ident, 0) - ref)
check("(a) 恒等/色彩", d.mean() < 0.8 and d.max() <= 3, f"平均绝对误差 {d.mean():.3f} 最大 {d.max():.0f} (抖动 ±0.5 + 取整, 期望 平均<0.8 最大<=3)")

# (b) 几何: 平移 + 旋转 + 放大, 对比 cv2.warpAffine 的三次插值
z = complex(math.cos(math.radians(0.6)), math.sin(math.radians(0.6)))
t = complex(23.5, -17.25) * W / 1280
zoom = 1.05
m = stabcore.view_matrix(z, t, W, H, zoom)
s_out_inv = np.array([[1 / W, 0, 0.5 / W], [0, 1 / H, 0.5 / H], [0, 0, 1]])
s_src = np.array([[W, 0, -0.5], [0, H, -0.5], [0, 0, 1]])
P = s_src @ m @ s_out_inv
cpu = cv2.warpAffine(ref, P[:2], (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
errs = {}
for q, name, limit in ((2, "high", 0.8), (0, "fast", 1.2)):
    d = np.abs(render(m, q) - cpu)
    errs[name] = d.mean()
    check(f"(b) 变换 quality={name}", d.mean() < limit, f"平均绝对误差 {d.mean():.3f} p99 {np.percentile(d, 99):.1f}")
d_bad = np.abs(render(stabcore.view_matrix(z, -t, W, H, zoom), 2) - cpu)      # 符号检验: 反向平移误差应显著变大
check("(b) 反向平移对照", d_bad.mean() > 5 * errs["high"], f"平均绝对误差 {d_bad.mean():.3f} (应远大于上面)")

# (c) 缩小显示: lod=1 的 mip 路径, 与面积平均缩小对比
if W >= 1280:
    W2, H2 = W // 2, H // 2
    ctx, r = open_gl(W2, H2)
    ctx.screen.use()
    ctx.viewport = (0, 0, W2, H2)
    ctx.clear(0, 0, 0, 1)
    r.draw((0, 0, W2, H2), ident, ident, 0.0, 1.0, 2)
    out = read(ctx, W2, H2)
    area = cv2.resize(ref, (W2, H2), interpolation=cv2.INTER_AREA)
    d = np.abs(out - area)
    check("(c) 2x 缩小(lod=1) vs 面积平均", d.mean() < 1.0, f"平均绝对误差 {d.mean():.3f} p99 {np.percentile(d, 99):.1f}")
    if args.out:
        cv2.imwrite(args.out, cv2.cvtColor(np.hstack([area, out]).astype(np.uint8), cv2.COLOR_RGB2BGR))

# (d) 1:1 原生像素: 640x360 窗口显示源画面正中央, 与 numpy 裁切对比; 再加整数平移校正
if W >= 640 and H >= 360:
    ctx, r = open_gl(640, 360)
    for tx, ty in ((0, 0), (8, -6)):
        m = stabcore.view_matrix(1 + 0j, complex(tx, ty), W, H, view=(640, 360))
        ctx.screen.use()
        ctx.viewport = (0, 0, 640, 360)
        ctx.clear(0, 0, 0, 1)
        r.draw((0, 0, 640, 360), m, m, 0.0, 0.0, 2)
        y0, x0 = H // 2 - 180 - ty, W // 2 - 320 - tx
        d = np.abs(read(ctx, 640, 360) - ref[y0:y0 + 360, x0:x0 + 640])
        check(f"(d) 1:1 中心裁切 校正平移({tx},{ty})", d.mean() < 0.6, f"平均绝对误差 {d.mean():.3f} 最大 {d.max():.0f} (期望 ≈0.35)")

print("全部通过" if not failed else f"失败 {len(failed)} 项: {failed}")
sys.exit(1 if failed else 0)
