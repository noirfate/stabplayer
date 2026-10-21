"""界面输入检查(需要真实显示器, 窗口会弹出、全屏会盖住屏幕几秒): 全屏快捷键走完整事件路径, 输入法(IME)是否被关掉, 鼠标移动是否扰乱播放节奏。

1) 文字输入应处于关闭状态: 打开时, 中文输入法会吃掉字母键, F/S/C 等快捷键全部失效。
2) 全屏切换: F 开 → F11 关 → Alt+Enter 开 → 双击关 → F 开 → Esc 关。
3) 持续发鼠标移动事件的 2 秒里, 重绘次数应与新帧数相当(鼠标移动不应额外重绘, 否则垂直同步下节奏不匀)。
用法: python tools/check_ui.py 视频文件 [--out-dir 截图目录]       任一项不符合则退出码为 1。
"""
import argparse
import ctypes
import faulthandler
import os
import sys
import tempfile
import time

faulthandler.dump_traceback_later(90, exit=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame

import stabplayer as sp

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("video")
ap.add_argument("--out-dir", default=tempfile.gettempdir())
opts = ap.parse_args()

sp.setup_logging()
sp.enable_dpi_awareness()      # 与播放器一致: 按真实像素创建窗口(高分屏缩放下不被系统拉伸)
args = sp.build_parser().parse_args([opts.video, "--mute", "--prebuffer", "2"])
p = sp.Player(sp.probe(opts.video), args)
lib = sp.sdl_lib()
lib.SDL_IsTextInputActive.restype = ctypes.c_int
failed = []
counts = {"draw": 0, "new": 0}
_draw, _update = p.draw, p.update
p.draw = lambda: (counts.__setitem__("draw", counts["draw"] + 1), _draw())[1]


def counted_update():
    new = _update()
    counts["new"] += bool(new)
    return new


p.update = counted_update


def check(name, ok, text=""):
    print(("通过 " if ok else "失败 ") + name + (f": {text}" if text else ""), flush=True)
    if not ok:
        failed.append(name)


def key(k, mod=0):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode="", scancode=0))


def double_click():
    for _ in range(2):
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(400, 300), button=1))


# (时刻 s, 动作)。动作 = 函数, 或 ("expect", 期望的 fullscreen 状态)
steps = [(1.0, lambda: check("文字输入已关闭", lib.SDL_IsTextInputActive() == 0, f"SDL_IsTextInputActive={lib.SDL_IsTextInputActive()}")),
         (1.5, lambda: key(pygame.K_f)), (2.5, ("expect", True, "F 进入全屏")),
         (3.0, lambda: key(pygame.K_F11)), (4.0, ("expect", False, "F11 退出全屏")),
         (4.5, lambda: key(pygame.K_RETURN, pygame.KMOD_ALT)), (5.5, ("expect", True, "Alt+Enter 进入全屏")),
         (6.0, double_click), (7.0, ("expect", False, "双击退出全屏")),
         (7.5, lambda: key(pygame.K_f)), (8.5, ("expect", True, "F 再次进入全屏")),
         (9.0, lambda: key(pygame.K_ESCAPE)), (10.0, ("expect", False, "Esc 退出全屏")),
         (10.5, lambda: counts.update(draw=0, new=0)), (12.5, "mouse_check"), (13.0, "end")]
done = set()
state = {"t0": None, "mouse": 0.0}


def hook(pl):
    if pl.buffering:
        return False
    now = time.perf_counter()
    state["t0"] = state["t0"] or now
    t = now - state["t0"]
    if 10.5 <= t < 12.5 and now - state["mouse"] > 0.01:        # 约 100Hz 鼠标移动
        state["mouse"] = now
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(300, 300), rel=(1, 1), buttons=(0, 0, 0)))
    for i, (at, act) in enumerate(steps):
        if i in done or t < at:
            continue
        done.add(i)
        if act == "end":
            return True
        if act == "mouse_check":
            ratio = counts["draw"] / max(counts["new"], 1)
            check("鼠标移动不额外重绘", ratio < 1.3, f"2 秒内重绘 {counts['draw']} 次 / 新帧 {counts['new']} 个 = {ratio:.2f} (期望 < 1.3)")
        elif isinstance(act, tuple):
            check(act[2], pl.fullscreen == act[1], f"fullscreen={pl.fullscreen}")
            if act[1]:
                pl.draw()
                pl.screenshot(os.path.join(opts.out_dir, "ui_fullscreen.png"))
                print("     全屏输出", pl.rect[2], "x", pl.rect[3], "(窗口", pygame.display.get_window_size(), ")", flush=True)
        else:
            act()
    return False


p.run(hook)
print("全部通过" if not failed else f"失败 {len(failed)} 项: {failed}")
sys.exit(1 if failed else 0)
