"""像素对齐模式(--pixel-perfect / 键 Z)检查, 不需要真实 4K 屏。

1) 规则表: fit_rect_integer 对各种 窗口 × 视频 尺寸给出的倍率/矩形(纯计算)。
   另外在每个窗口里按 Z 键(走真实按键事件)检查开关: 关闭后矩形回到按比例填满, 再按恢复。
2) 端到端: 用真实 Player 在指定大小的窗口里渲染(--mode orig 恒等变换), 读回整个窗口, 与独立算出的参考逐像素比较:
   倍率 1x → 输出应与源帧一致(只差 ±0.5 LSB 抖动); 1/2x → 与 2×2 面积平均一致; 2x → 与双三次放大一致; 矩形以外必须严格为黑。
   窗口可以比屏幕大(GL 输出尺寸是真实的), 所以能在没有 4K 屏的机器上验证 3840×2160 / 3840×2054 这类场景。
用法: python tools/check_pixel_perfect.py 视频文件      需要 OpenGL; 会弹出几个窗口; 任一项不符合则退出码为 1。
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cv2
import numpy as np
import pygame

import stabplayer as sp

RULES = [  # (窗口, 视频显示尺寸, 期望输出宽高, 倍率)
    ((3840, 2160), (3840, 2160), (3840, 2160), "1x"),
    ((3840, 2054), (3840, 2160), (1920, 1080), "1/2x"),       # 4K 屏上最大化窗口(客户区约 2054 高)
    ((3824, 2040), (3840, 2160), (1920, 1080), "1/2x"),       # 4K 屏上的默认窗口上限
    ((2560, 1440), (3840, 2160), (1920, 1080), "1/2x"),
    ((1920, 1080), (3840, 2160), (1920, 1080), "1/2x"),
    ((1366, 768), (3840, 2160), (1280, 720), "1/3x"),
    ((800, 600), (3840, 2160), (768, 432), "1/5x"),
    ((3840, 2160), (1920, 1080), (3840, 2160), "2x"),
    ((3840, 2054), (1920, 1080), (1920, 1080), "1x"),
    ((2560, 1440), (1920, 1080), (1920, 1080), "1x"),
    ((7680, 4320), (1920, 1080), (7680, 4320), "4x"),
    ((1500, 1100), (1440, 1080), (1440, 1080), "1x"),
]


def check_rules() -> int:
    bad = 0
    for (ww, wh), (dw, dh), (ew, eh), el in RULES:
        (x, y, w, h), label = sp.fit_rect_integer(ww, wh, dw, dh)
        ok = (w, h) == (ew, eh) and label == el and x == (ww - w) // 2 and y == (wh - h) // 2 and w <= ww and h <= wh
        bad += not ok
        print(("通过 " if ok else "失败 ") + f"规则: 窗口 {ww}x{wh} 视频 {dw}x{dh} -> 矩形 ({x},{y},{w},{h}) {label}")
    return bad


def reference_rgb(info, planes) -> np.ndarray:
    top = (1 << info.bits) - 1
    y, u, v = (p.astype(np.float32) for p in planes)
    W, H = info.width, info.height
    U = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
    V = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
    mat, off, scale = sp.yuv_params(info.colorspace, info.full_range, info.bits, H)
    return np.clip(((np.stack([y, U, V], -1) / top - np.array(off)) * np.array(scale)) @ mat.T, 0, 1) * 255


def compare(info, ww, wh, got) -> int:
    (w, h), (x, y, rw, rh), label, img = got["win"], got["rect"], got["label"], got["img"]
    name = f"窗口 {w}x{h} -> 输出 {rw}x{rh} ({label})"
    if (w, h) != (ww, wh):
        print(f"失败 {name}: 窗口没有创建成请求的 {ww}x{wh} 大小")
        return 1
    top = h - (y + rh)                                  # 读回的图像自上而下, 矩形原点在左下
    inside = img[top:top + rh, x:x + rw]
    outside = img.copy()
    outside[top:top + rh, x:x + rw] = 0
    ref = reference_rgb(info, got["planes"])
    if label == "1x":
        expect, limit = ref, 0.8
    elif label.startswith("1/"):
        expect, limit = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_AREA), 1.0
    else:
        expect, limit = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_CUBIC), 1.5
    d = np.abs(inside - expect)
    border_ok = outside.max() == 0
    far = float((d.max(-1) > 3).mean())                 # 与参考相差超过 3 的像素占比(high 质量的色度是双三次、参考是双线性, 色块边缘会有个别差异)
    ok = d.mean() < limit and border_ok and (label != "1x" or far < 1e-4)
    if not ok:                                          # 失败时存下读回的整窗图, 并标出与参考不一致/应为黑的位置
        vis = img.copy()
        diff = np.zeros(img.shape[:2], bool)
        diff[top:top + rh, x:x + rw] = d.max(-1) > 3
        diff |= outside.max(-1) > 0
        vis[diff] = (255, 0, 0)
        path = os.path.join(tempfile.gettempdir(), f"pixel_perfect_fail_{w}x{h}.png")
        cv2.imwrite(path, cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(path.replace(".png", "_raw.png"), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        ys, xs = np.nonzero(diff)
        print(f"     失败截图(红=不一致): {path}; 不一致像素 {diff.sum()} 个, 范围 x {xs.min() if len(xs) else '-'}~{xs.max() if len(xs) else '-'} "
              f"y {ys.min() if len(ys) else '-'}~{ys.max() if len(ys) else '-'}")
    print(("通过 " if ok else "失败 ") + f"{name}: 与参考的平均绝对误差 {d.mean():.3f} 最大 {d.max():.0f} (期望 平均<{limit}"
          + (f"; 差>3 的像素 {far * 100:.4f}% , 期望 <0.01%" if label == "1x" else "") + f"); 黑边{'严格为 0' if border_ok else '不是全黑!'}", flush=True)
    return 0 if ok else 1


def run_case(video: str, ww: int, wh: int) -> int:
    """子进程: 在 ww×wh 的窗口里渲染一帧, 与参考比较(比较在播放器还在运行时做, 退出后不再做计算)。"""
    sp.enable_dpi_awareness()
    sp.setup_logging()
    info = sp.probe(video)
    args = sp.build_parser().parse_args([video, "--mute", "--prebuffer", "1", "--mode", "orig", "--pixel-perfect", "--size", f"{ww}x{wh}"])
    p = sp.Player(info, args)
    result = []
    state = {"phase": 0}

    def key_z():
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=0, unicode="z", scancode=0))

    def hook(pl):
        if pl.buffering or pl.cur is None:
            return False
        if state["phase"] == 1:                         # 刚按过 Z: 应已关闭像素对齐, 矩形回到「按比例填满」
            pl.draw()
            ok = (not pl.pixel_perfect) and pl.rect == sp.fit_rect(ww, wh, info.aspect)
            print(("通过 " if ok else "失败 ") + f"Z 键关闭像素对齐: 输出矩形 {pl.rect}", flush=True)
            result.append(0 if ok else 1)
            state["phase"] = 2
            key_z()
            return False
        if state["phase"] == 2:                         # 再按 Z: 应恢复
            pl.draw()
            ok = pl.pixel_perfect and pl.pp_label == state["label"]
            print(("通过 " if ok else "失败 ") + f"Z 键重新开启像素对齐: 倍率 {pl.pp_label}", flush=True)
            result.append(0 if ok else 1)
            return True
        pl.mouse_t, pl.toast_until = 0.0, 0.0           # 关掉进度条/提示/帮助/统计等叠加层(鼠标在窗口上或误触按键会触发它们)
        pl.help_on = pl.stats_on = False
        pl.draw()
        w, h = pygame.display.get_window_size()
        data = pl.ctx.screen.read(viewport=(0, 0, w, h), components=3, alignment=1)
        got = dict(img=np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1].astype(np.float32), planes=pl.cur.planes,
                   rect=pl.rect, label=pl.pp_label, win=(w, h))
        result.append(compare(info, ww, wh, got))
        state["phase"], state["label"] = 1, got["label"]
        key_z()
        return False

    p.run(hook)
    return max(result) if len(result) == 3 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--case", help="(内部) 只跑一个 WxH 窗口")
    args = ap.parse_args()
    if args.case:
        ww, wh = map(int, args.case.lower().split("x"))
        sys.exit(run_case(args.video, ww, wh))
    bad = check_rules()
    info = sp.probe(args.video)
    W, H = sp.display_size(info)
    cases = [(W, H), (W, H - 106)]                      # 刚好放得下 → 1x; 少一点高度(像最大化窗口) → 1/2x
    if W <= 1920:
        cases.append((2 * W, 2 * H))                    # 窗口大一倍 → 2x
    for ww, wh in cases:
        bad += subprocess.run([sys.executable, os.path.abspath(__file__), args.video, "--case", f"{ww}x{wh}"]).returncode != 0
    print("全部通过" if not bad else f"失败 {bad} 项")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
