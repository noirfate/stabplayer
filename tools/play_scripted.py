"""脚本驱动真实播放器: 按时间表模拟按键/截图, 最后打印媒体时钟/音频位置的时间线, 并检查播放确实在推进。

时间表: 4.0s 截图 → 4.2s 按 C(分屏) → 5.5s 截图 → 5.7s 按 C → 7.0s 暂停 → 8.0s 继续 → 9.0s 按 → (快进 5 秒) → 13s 结束。
用法: python tools/play_scripted.py 视频文件 [--out-dir 截图目录] [播放器选项, 如 --prebuffer 3 --mute --no-audio]
结果: 媒体时钟在结尾应已越过 8 秒(没有冻住); 否则退出码为 1。窗口会弹出来。
"""
import argparse
import faulthandler
import os
import sys
import tempfile
import time

faulthandler.dump_traceback_later(60, exit=True)       # 卡死时打印所有线程栈并退出
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame

import stabplayer as sp

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("video")
ap.add_argument("--out-dir", default=tempfile.gettempdir(), help="截图保存目录")
opts, extra = ap.parse_known_args()

sp.setup_logging()
sp.enable_dpi_awareness()      # 与播放器一致: 按真实像素创建窗口(高分屏缩放下不被系统拉伸)
args = sp.build_parser().parse_args([opts.video, "--stats", "--size", "1280x720"] + extra)
p = sp.Player(sp.probe(opts.video), args)
t0 = time.perf_counter()
script = [(4.0, "shot", "a_stab"), (4.2, "key", pygame.K_c), (5.5, "shot", "b_split"), (5.7, "key", pygame.K_c),
          (7.0, "key", pygame.K_SPACE), (7.6, "shot", "c_paused"), (8.0, "key", pygame.K_SPACE),
          (9.0, "key", pygame.K_RIGHT), (13.0, "end", None)]
done = set()
marks = []


def hook(pl):
    now = time.perf_counter() - t0
    for i, (at, kind, arg) in enumerate(script):
        if i in done or now < at:
            continue
        done.add(i)
        if kind == "key":
            pl.on_key(arg, 0)
        elif kind == "shot":
            pl.draw()
            pl.screenshot(os.path.join(opts.out_dir, f"{arg}.png"))
        elif kind == "end":
            return True
    if not marks or now - marks[-1][0] > 0.9:
        marks.append((now, pl.media_time(), pl.audio.position() if pl.audio else None, pl.buffering, pl.paused))
    return False


p.run(hook)
print("时间线(墙钟s, 媒体时钟s, 音频位置s, buffering, paused):")
for m in marks:
    print("  %.1f  media %.2f  audio %s  buf=%s paused=%s" % (m[0], m[1], "%.2f" % m[2] if m[2] is not None else "-", m[3], m[4]))
final = marks[-1][1] if marks else 0.0
ok = final > 8.0
print(f"{'通过' if ok else '失败'}: 结尾媒体时钟 {final:.2f}s (期望 > 8s, 播放在推进)")
sys.exit(0 if ok else 1)
