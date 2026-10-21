"""音频死锁压力测试: 持有 GIL 的计算 + 高频 pause/play/start/switch, 最后 close。
曾经的死锁: pygame._sdl2.audio 在持有 GIL 时调用 SDL_PauseAudioDevice, 与 SDL 音频线程(持设备锁、等 GIL 跑回调)互等。
改音频代码(AudioOut)后跑一遍。卡住则 60 秒后打印所有线程栈并退出(退出码 1)。
用法: python tools/check_audio.py 带音轨的视频 [--seconds 25]      需要一个可用的音频输出设备(声音会被播放)。
"""
import argparse
import faulthandler
import os
import random
import sys
import threading
import time

faulthandler.dump_traceback_later(90, exit=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pygame

import stabplayer as sp

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("video")
ap.add_argument("--seconds", type=float, default=25.0)
args = ap.parse_args()

pygame.init()
a = sp.AudioOut(args.video)
if not a.ok:
    sys.exit("音频设备打不开, 无法测试")
print("设备列表:", a.names, "| 当前:", a.name)
stop = False


def burn():
    """制造 GIL 争用: 大量小的 Python 级操作"""
    x = np.zeros(1000)
    while not stop:
        for _ in range(2000):
            x = x * 0.999 + 1.0


threading.Thread(target=burn, daemon=True).start()
rng = random.Random(1)
a.start(0.0)
a.play()
t0 = time.perf_counter()
n = {"pause": 0, "play": 0, "start": 0, "switch": 0}
while time.perf_counter() - t0 < args.seconds:
    warm = time.perf_counter() - t0 > 5
    op = rng.choice(["pause", "play", "start", "start", "pause", "play", "switch"] if warm else ["pause", "play", "start"])
    if op == "pause":
        a.pause()
    elif op == "play":
        a.play()
    elif op == "start":
        a.start(rng.uniform(0, 20))
    else:
        a.switch_device()
    n[op] += 1
    time.sleep(rng.uniform(0, 0.03))
stop = True
print("操作次数:", n, " 欠载:", a.underruns)
t = time.perf_counter()
a.close()
print("close 用时 %.0f ms" % ((time.perf_counter() - t) * 1000))
print("通过: 无死锁")
