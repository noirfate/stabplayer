#!/usr/bin/env python3
"""stabplayer: 实时去抖视频播放器, 面向 4K、画质优先。

  解码线程(PyAV)  → YUV420 平面 + 降采样亮度估计运动(stabcore) → 前瞻缓冲队列
  主线程(OpenGL)  → YUV 平面直传 GPU, 一个着色器完成「去抖变换 + 缩放 + 补缺口 + YUV→RGB」
  音频(SDL)       → 音频时钟驱动视频时间戳; 无音轨则用系统时钟

只依赖 OpenGL 3.3, 与显卡厂商无关(NVIDIA / AMD / Intel 核显均可)。
用法: python stabplayer.py 视频文件 [选项]      按 H 查看快捷键
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import ctypes.util
import faulthandler
import glob
import logging
import math
import os
import platform
import re
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP_VERSION = "0.1.0"
log = logging.getLogger("stab")
_crash_file = None


def app_dir() -> str:
    return os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))


def enable_dpi_awareness():
    """让 Windows 用真实像素创建窗口, 否则高分屏(缩放 125% 等)上画面会被系统二次缩放而变糊。必须在创建任何窗口之前调用。"""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def message_box(title: str, text: str):
    """无控制台的 exe 里错误看不见, 用系统弹窗提示; 其他平台打到 stderr。"""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
            return
        except Exception:
            pass
    print(f"{title}: {text}", file=sys.stderr)


def setup_logging() -> str | None:
    """日志写到 exe(或脚本)同目录的 stabplayer.log, 不可写时退到临时目录; 每次启动追加, 超过 3MB 先轮转。
    同时打开 faulthandler: 原生崩溃(显卡驱动、解码库)时把所有线程栈写进同一个文件。返回日志路径。"""
    global _crash_file
    if log.handlers:
        return log.handlers[0].baseFilename
    for d in (app_dir(), tempfile.gettempdir()):
        path = os.path.join(d, "stabplayer.log")
        try:
            if os.path.exists(path) and os.path.getsize(path) > 3_000_000:
                os.replace(path, path + ".old")
            handler = logging.FileHandler(path, encoding="utf-8")
            break
        except OSError:
            continue
    else:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    _crash_file = open(path, "ab", buffering=0)
    faulthandler.enable(file=_crash_file, all_threads=True)

    def excepthook(et, ev, tb):
        log.error("未捕获异常:\n%s", "".join(traceback.format_exception(et, ev, tb)))
    sys.excepthook = excepthook
    threading.excepthook = lambda a: log.error("线程 %s 未捕获异常:\n%s", a.thread.name if a.thread else "?",
                                              "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
    log.info("=" * 70)
    log.info("stabplayer %s 启动 %s  日志: %s", APP_VERSION, time.strftime("%Y-%m-%d %H:%M:%S"), path)
    return path


def fatal(text: str, exc: BaseException | None = None):
    """记日志 + 弹窗, 然后退出。"""
    path = setup_logging()
    if exc is not None:
        log.error("%s\n%s", text, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    else:
        log.error(text)
    message_box("stabplayer 出错", f"{text}\n\n详细信息见日志:\n{path}")
    sys.exit(1)


try:
    import av
    import cv2
    import moderngl
    import numpy as np
    import pygame
    import stabcore
    import stabres
except ImportError as e:
    fatal(f"缺少依赖模块 {e.name}。源码运行请先安装: pip install numpy opencv-python av pygame moderngl", e)

# 强度预设: (名称, 平滑核 σ 秒, 校正范围倍数)。平滑越强, 校正量越大, 允许的最大校正量(校正范围)也要越大, 所以两者联动。
# 画面不裁切: 校正把画面移开后露出的边缘缺口用相邻帧补上。
# 依据: 在手持抖动很大的 4K 样例上, 以幕布区域为独立参照, 残余抖动相对原始约为 0.92 / 0.85 / 0.69 / 0.51 / 0.36;
# σ 超过 1s 不再有收益, 上限由校正范围决定。
PRESETS = (("light", 0.15, 1.04), ("standard", 0.25, 1.06), ("strong", 0.40, 1.10),
           ("stronger", 0.60, 1.15), ("max", 1.00, 1.25))
DEFAULT_PRESET = 4
CROP_MIN, CROP_MAX, CROP_STEP = 1.01, 1.30, 0.01       # 「校正范围」档位: 允许的最大校正量 ≈ 画面宽 × (1-1/z)/2
FILL_RADIUS = 16                                        # 补缺口用的相邻帧: 前后各这么多帧内
FILL_OFFSETS = (4, -4, 8, -8, 16, -16)                  # 按时间距离从近到远依次尝试
POOL_LAYERS = 2 * FILL_RADIUS + 1                       # 相邻帧纹理池层数
FEATHER_PX = 16                                         # 补缺口交界处的羽化宽度(源像素)
QUALITIES = ("high", "normal", "fast")          # 对应 着色器 quality = 2 / 1 / 0


# ───────────────────────────── 媒体信息与像素格式 ─────────────────────────────

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    aspect: float           # 显示宽高比(已含像素宽高比)
    bits: int               # 8 / 10 / 12
    colorspace: int         # AVColorSpace
    full_range: bool
    has_audio: bool


def probe(path: str) -> VideoInfo:
    with av.open(path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        if vs is None:
            raise ValueError("文件中没有视频流")
        frame = next(c.decode(vs))
        name = frame.format.name
        m = re.search(r"p(\d+)(?:le|be)$", name)
        bits = min(int(m.group(1)), 12) if m else 8
        if bits not in (8, 10, 12):
            bits = 8 if bits < 8 else 10
        sar = float(vs.sample_aspect_ratio or 1) or 1.0
        duration = c.duration / 1e6 if c.duration else (float(vs.duration * vs.time_base) if vs.duration else 0.0)
        return VideoInfo(
            path=path, width=frame.width, height=frame.height,
            fps=float(vs.average_rate or vs.guessed_rate or 30), duration=duration,
            aspect=frame.width * sar / frame.height, bits=bits,
            colorspace=int(frame.colorspace), full_range=int(frame.color_range) == 2 or name.startswith("yuvj"),
            has_audio=any(s.type == "audio" for s in c.streams))


def frame_planes(frame, bits: int):
    """取出 (Y, U, V) 三个 C 连续的 numpy 数组; 格式不是 4:2:0 或位深不符时先用 swscale 转换。"""
    name = frame.format.name
    target = "yuv420p" if bits == 8 else f"yuv420p{bits}le"
    if name != target and not (bits == 8 and name == "yuvj420p"):
        frame = frame.reformat(format=target)
    dtype = np.uint8 if bits == 8 else np.uint16
    out = []
    for p in frame.planes[:3]:
        stride = p.line_size // np.dtype(dtype).itemsize
        a = np.frombuffer(p, dtype=dtype)[: stride * p.height].reshape(p.height, stride)
        out.append(a if stride == p.width else np.ascontiguousarray(a[:, : p.width]))
    return tuple(out)


def yuv_params(colorspace: int, full_range: bool, bits: int, height: int):
    """返回 (3×3 矩阵, 偏移, 缩放): rgb = 矩阵 · ((码值/满量程 − 偏移) · 缩放)。色彩空间未标注时按分辨率猜。"""
    kr, kb = {1: (0.2126, 0.0722), 5: (0.299, 0.114), 6: (0.299, 0.114),
              9: (0.2627, 0.0593), 10: (0.2627, 0.0593)}.get(colorspace, (0.2126, 0.0722) if height >= 720 else (0.299, 0.114))
    kg = 1 - kr - kb
    mat = np.array([[1, 0, 2 * (1 - kr)],
                    [1, -2 * kb * (1 - kb) / kg, -2 * kr * (1 - kr) / kg],
                    [1, 2 * (1 - kb), 0]])
    top, s = (1 << bits) - 1, 1 << (bits - 8)
    if full_range:
        y_off, y_sc, c_sc = 0.0, 1.0, 1.0
    else:
        y_off, y_sc, c_sc = 16 * s / top, top / (219 * s), top / (224 * s)
    c_off = 128 * s / top
    return mat, (y_off, c_off, c_off), (y_sc, c_sc, c_sc)


# ───────────────────────────── 解码 + 运动分析线程 ─────────────────────────────

class FrameItem:
    __slots__ = ("idx", "pts", "planes")

    def __init__(self, idx: int, pts: float, planes):
        self.idx, self.pts, self.planes = idx, pts, planes


class VideoPipeline(threading.Thread):
    """从 start 秒起解码并逐帧分析运动, 领先播放头若干帧, 结果放进有界队列(反压)。"""

    def __init__(self, info: VideoInfo, start: float, analysis_width: int, sigma_sec: float, zoom: float, buffer_frames: int,
                 decode_threads: int = 0):
        super().__init__(daemon=True, name="video-pipeline")
        self.info, self.start_t, self.analysis_width, self.buffer_frames = info, start, analysis_width, buffer_frames
        self.decode_threads = decode_threads
        self.traj = stabcore.Trajectory()
        self.stab = stabcore.Stabilizer(self.traj, info.width, info.height, info.fps, max_lookahead=buffer_frames - 6)
        self.queue: collections.deque[FrameItem] = collections.deque()
        self.cond = threading.Condition()
        self.stopped = False
        self.error: Exception | None = None
        self.ms_analysis = 0.0          # 每帧运动分析耗时(指数滑动平均)
        self.n_frames = self.n_fail = 0
        self.configure(sigma_sec, zoom)

    def configure(self, sigma_sec: float, zoom: float):
        """更新平滑强度/校正范围; 帧缓冲容量 = max(缓冲目标, 前瞻需求 + 余量)。缓冲越深, 机器偶尔跟不上时稳定窗口越不容易被截断。"""
        self.stab.configure(sigma_sec, zoom)
        with self.cond:
            self.capacity = max(self.buffer_frames, self.stab.lookahead() + 6)
            self.cond.notify_all()

    def stop(self, join: bool = True):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        if join and self.is_alive():
            self.join(timeout=3)

    def run(self):
        stabres.lower_thread_priority()     # 机器忙时让界面/渲染/音频先拿到 CPU
        try:
            self._run()
        except Exception as e:      # 解码/分析出错: 记录并让播放器显示, 不让线程静默死掉
            self.error = e
            log.exception("解码/分析线程出错")
        finally:
            self.traj.eof = True
            with self.cond:
                self.cond.notify_all()

    def _run(self):
        info = self.info
        with av.open(info.path) as c:
            vs = next(s for s in c.streams if s.type == "video")
            vs.thread_type = "AUTO"
            if self.decode_threads:
                vs.thread_count = self.decode_threads
            if self.start_t > 0.05:
                c.seek(int(self.start_t / float(vs.time_base)), stream=vs, backward=True)
            est = stabcore.MotionEstimator(info.width, info.height, self.analysis_width, info.bits)
            half, idx, last_pts = 0.5 / info.fps, 0, None
            for packet in c.demux(vs):
                if self.stopped:
                    return
                try:
                    frames = packet.decode()
                except av.error.FFmpegError:
                    continue            # 个别损坏的包: 跳过
                for frame in frames:
                    pts = frame.time if frame.time is not None else (0.0 if last_pts is None else last_pts + 1 / info.fps)
                    last_pts = pts
                    if pts < self.start_t - half:
                        continue        # seek 落在关键帧上, 前面的帧只解码不分析
                    t0 = time.perf_counter()
                    planes = frame_planes(frame, info.bits)
                    z, t, ok = est.push(planes[0])
                    self.n_frames += 1
                    self.n_fail += not ok
                    self.traj.append(z, t)
                    ms = (time.perf_counter() - t0) * 1000
                    self.ms_analysis = ms if not self.ms_analysis else 0.95 * self.ms_analysis + 0.05 * ms
                    item, idx = FrameItem(idx, pts, planes), idx + 1
                    with self.cond:
                        while len(self.queue) >= self.capacity and not self.stopped:
                            self.cond.wait(0.1)
                        if self.stopped:
                            return
                        self.queue.append(item)

    def prefilled(self) -> bool:
        """第 0 帧所需的前瞻是否就绪(或已到片尾), 即可以开始播放。"""
        return self.traj.ready(0, self.stab.lookahead())

    def take(self, t: float):
        """取出 pts ≤ t 的最后一帧(更早的视为已错过而丢弃)。返回 (帧或 None, 丢弃数)。"""
        item, n = None, 0
        with self.cond:
            while self.queue and self.queue[0].pts <= t:
                item = self.queue.popleft()
                n += 1
            if n:
                self.cond.notify_all()
        return item, max(n - 1, 0)

    def peek_upto(self, last_idx: int):
        """队列里 idx ≤ last_idx 的帧(不取出), 供全幅稳定把即将用到的相邻帧提前传进 GPU。"""
        out = []
        with self.cond:
            for it in self.queue:
                if it.idx > last_idx:
                    break
                out.append(it)
        return out

    def drained(self) -> bool:
        with self.cond:
            return self.traj.eof and not self.queue


# ───────────────────────────── 音频 ─────────────────────────────

class _SDLAudioSpec(ctypes.Structure):
    _fields_ = [("freq", ctypes.c_int), ("format", ctypes.c_uint16), ("channels", ctypes.c_uint8),
                ("silence", ctypes.c_uint8), ("samples", ctypes.c_uint16), ("padding", ctypes.c_uint16),
                ("size", ctypes.c_uint32), ("callback", ctypes.c_void_p), ("userdata", ctypes.c_void_p)]


_SDL_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int)
_sdl = None


def sdl_lib():
    """pygame 自带的 SDL2 动态库。

    不用 pygame._sdl2.audio: 它在持有 GIL 时调用 SDL_PauseAudioDevice / SDL_CloseAudioDevice, 而 SDL 音频线程握着设备锁、
    等 GIL 去执行 Python 回调, 两边互等即死锁(暂停、seek、退出都可能撞上)。ctypes 调用外部函数时会释放 GIL, 不存在这个问题。
    同时 SDL_OpenAudioDevice(NULL) 能直接打开系统默认设备。
    """
    global _sdl
    if _sdl is not None:
        return _sdl
    base = os.path.dirname(pygame.__file__)
    roots = [base, os.path.dirname(base), os.path.join(os.path.dirname(base), "pygame.libs"),
             getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)]
    cands = []
    for r in filter(None, roots):
        for pat in ("SDL2.dll", "libSDL2-2*", "libSDL2.so*", "libSDL2*.dylib"):
            cands += glob.glob(os.path.join(r, pat))
    cands.append("SDL2.dll" if sys.platform == "win32" else (ctypes.util.find_library("SDL2") or "libSDL2-2.0.so.0"))
    lib, errors = None, []
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            log.info("SDL2 库: %s", c)
            break
        except OSError as e:
            errors.append(f"{c}: {e}")
    if lib is None:
        raise OSError("找不到 SDL2 动态库: " + "; ".join(errors))
    lib.SDL_InitSubSystem.argtypes = [ctypes.c_uint32]
    lib.SDL_GetNumAudioDevices.argtypes = [ctypes.c_int]
    lib.SDL_GetAudioDeviceName.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.SDL_GetAudioDeviceName.restype = ctypes.c_char_p
    lib.SDL_OpenAudioDevice.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_SDLAudioSpec),
                                        ctypes.POINTER(_SDLAudioSpec), ctypes.c_int]
    lib.SDL_OpenAudioDevice.restype = ctypes.c_uint32
    lib.SDL_PauseAudioDevice.argtypes = [ctypes.c_uint32, ctypes.c_int]
    lib.SDL_CloseAudioDevice.argtypes = [ctypes.c_uint32]
    lib.SDL_GetError.restype = ctypes.c_char_p
    lib.SDL_InitSubSystem(0x10)         # SDL_INIT_AUDIO
    _sdl = lib
    return lib


def list_audio_devices() -> list[str]:
    lib = sdl_lib()
    return [(lib.SDL_GetAudioDeviceName(i, 0) or b"?").decode("utf-8", "replace")
            for i in range(max(lib.SDL_GetNumAudioDevices(0), 0))]


class AudioOut:
    """独立解码线程 → 环形缓冲 → SDL 回调拉取。播放位置(以实际送入设备的样本数计)作为主时钟。"""

    RATE, CHUNK = 48000, 1024

    def __init__(self, path: str, device: str | None = None):
        self.path, self.ok = path, False
        self.lock = threading.Lock()
        self.chunks: collections.deque = collections.deque()
        self.avail = self.played = self.gen = 0
        self.base, self.eof, self.running = 0.0, False, False
        self.volume, self.muted = 1.0, False
        self.cb_wall = time.perf_counter()
        self.underruns = 0              # 缓冲被抽干(解码跟不上)的回调次数
        self.name = None
        self.dev = 0
        self._cb_ref = _SDL_CALLBACK(self._callback)        # 必须一直持有, 否则被回收后 SDL 回调成野指针
        try:
            self.names = list_audio_devices()
            log.info("音频输出设备: %s", self.names)
            self._open(device)                              # None = 系统默认设备
        except Exception:
            log.exception("无法打开音频输出设备, 将静音播放")
            self.names = []

    def _open(self, name: str | None):
        lib = sdl_lib()
        want = _SDLAudioSpec(freq=self.RATE, format=0x8010, channels=2, silence=0, samples=self.CHUNK, padding=0,
                             size=0, callback=ctypes.cast(self._cb_ref, ctypes.c_void_p), userdata=None)
        have = _SDLAudioSpec()
        dev = lib.SDL_OpenAudioDevice(name.encode("utf-8") if name else None, 0, ctypes.byref(want), ctypes.byref(have), 0)
        if not dev:
            raise RuntimeError((lib.SDL_GetError() or b"").decode("utf-8", "replace"))
        self.dev, self.name, self.ok = dev, name, True
        lib.SDL_PauseAudioDevice(dev, 1)
        log.info("已打开音频设备 %s: %d Hz, %d 声道, 块 %d 样本", name or "(系统默认)", have.freq, have.channels, have.samples)

    def _close_device(self):
        if self.dev:
            lib = sdl_lib()
            lib.SDL_PauseAudioDevice(self.dev, 1)
            lib.SDL_CloseAudioDevice(self.dev)
            self.dev = 0

    def switch_device(self) -> int:
        """切到下一个输出设备(系统默认 → 列表里的各设备 → 回到默认); 听不到声音时用。返回当前序号(0=系统默认)。"""
        if not self.names:
            return 0
        order = [None] + self.names
        i = (order.index(self.name) + 1) % len(order) if self.name in order else 0
        was_running = self.running
        self._close_device()
        try:
            self._open(order[i])
        except Exception:
            log.exception("切换音频设备失败")
            self._open(None)
            i = 0
        if was_running:
            self.play()
        return i

    def start(self, t: float):
        """从 t 秒开始(重新)填充缓冲; 设备保持暂停, 由 play() 启动。"""
        with self.lock:
            self.gen += 1
            self.chunks.clear()
            self.avail = self.played = 0
            self.base, self.eof = t, False
            gen = self.gen
        threading.Thread(target=self._decode, args=(gen, t), daemon=True, name="audio-decode").start()

    def play(self):
        self.running = True
        self.cb_wall = time.perf_counter()
        sdl_lib().SDL_PauseAudioDevice(self.dev, 0)

    def pause(self):
        self.running = False
        sdl_lib().SDL_PauseAudioDevice(self.dev, 1)

    def stalled(self) -> bool:
        """设备在运行却长时间不来取数据(被独占、设备被拔、远程音频没起来等): 正常时回调约每 21ms 一次。"""
        with self.lock:
            wall = self.cb_wall
        return self.running and time.perf_counter() - wall > 1.0

    def position(self) -> float:
        """当前正在听到的媒体时间(秒): 已送入样本数 + 两次回调之间按墙钟插值 − 设备缓冲延迟估计。"""
        with self.lock:
            played, wall = self.played, self.cb_wall
        extra = min(time.perf_counter() - wall, self.CHUNK / self.RATE) if self.running else 0.0
        return self.base + played / self.RATE + extra - 2 * self.CHUNK / self.RATE

    def close(self):
        with self.lock:
            self.gen += 1
        self.running = False
        try:
            self._close_device()
        except Exception:
            log.exception("关闭音频设备出错")

    def _callback(self, userdata, stream, length):
        out = np.ctypeslib.as_array(stream, shape=(length,)).view(np.int16).reshape(-1, 2)
        need, got = len(out), 0
        vol = 0.0 if self.muted else self.volume
        with self.lock:
            while got < need and self.chunks:
                c = self.chunks[0]
                take = min(len(c), need - got)
                out[got:got + take] = c[:take] if vol == 1.0 else (c[:take] * vol).astype(np.int16)
                got += take
                if take == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[take:]
            if got < need:
                out[got:] = 0
                if self.eof:                # 音频比视频短: 片尾的静音也计入时钟, 否则视频时钟会停住
                    got = need
                else:
                    self.underruns += 1
            self.avail -= min(got, self.avail)
            self.played += got
            self.cb_wall = time.perf_counter()

    def _decode(self, gen: int, t: float):
        try:
            with av.open(self.path) as c:
                a = next(s for s in c.streams if s.type == "audio")
                if t > 0.05:
                    c.seek(int(t / float(a.time_base)), stream=a, backward=True)
                rs = av.AudioResampler(format="s16", layout="stereo", rate=self.RATE)
                first, skip = True, 0
                for packet in c.demux(a):
                    if gen != self.gen:
                        return
                    try:
                        frames = packet.decode()
                    except av.error.FFmpegError:
                        continue
                    for frame in frames:
                        for out in rs.resample(frame):
                            arr = out.to_ndarray().reshape(-1, 2)
                            if first and out.time is not None:      # 对齐到 t: 多出的前导样本丢掉, 缺的补静音
                                lead = int(round((out.time - t) * self.RATE))
                                skip = max(-lead, 0)
                                if lead > 0:
                                    arr = np.concatenate([np.zeros((lead, 2), np.int16), arr])
                            first = False
                            if skip:
                                n = min(skip, len(arr))
                                arr, skip = arr[n:], skip - n
                            if not len(arr):
                                continue
                            while self.avail > self.RATE * 2 and gen == self.gen:
                                time.sleep(0.02)
                            with self.lock:
                                if gen != self.gen:
                                    return
                                self.chunks.append(arr)
                                self.avail += len(arr)
        except Exception:
            log.exception("音频解码出错")
        finally:
            with self.lock:
                if gen == self.gen:
                    self.eof = True


# ───────────────────────────── OpenGL 渲染 ─────────────────────────────

VERT = """
#version 330
in vec2 pos;
out vec2 uv;
void main() {
    uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);     // 左上角为原点
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

FRAG_VIDEO = """
#version 330
uniform sampler2D tex_y;
uniform sampler2D tex_u;
uniform sampler2D tex_v;
uniform sampler2DArray pool_y;      // 相邻帧纹理池(用来补去抖后画面边缘的缺口)
uniform sampler2DArray pool_u;
uniform sampler2DArray pool_v;
uniform mat3 m_left;        // 输出 uv → 当前帧 uv, 用于 uv.x < split (原片)
uniform mat3 m_right;       // 同上, 用于 uv.x >= split (去抖)
uniform mat3 nb_mat[6];     // 输出 uv → 相邻帧 uv (仅去抖一侧使用)
uniform int nb_layer[6];    // 相邻帧在纹理池里的层号
uniform int nb_count;
uniform vec2 edge;          // 判断「被某一帧覆盖」时的边缘余量(半个纹素)
uniform vec2 feather;       // 羽化带宽度(纹理坐标): 帧边缘内侧这一段里权重渐变到 0, 让帧与帧之间的接缝不是一条硬线
uniform float split;
uniform float lod;          // 缩小显示时的 mip 层级, 放大/1:1 时为 0
uniform int quality;        // 2: 亮度+色度双三次  1: 仅亮度双三次  0: 双线性
uniform float tex_scale;    // 纹理归一化值 → 码值/满量程
uniform vec3 yuv_off;
uniform vec3 yuv_scale;
uniform mat3 yuv_mat;
in vec2 uv;
out vec4 frag;

// Catmull-Rom 双三次, 利用硬件双线性做 9 次采样
float catmull(sampler2D t, vec2 p) {
    vec2 sz = vec2(textureSize(t, 0));
    vec2 pos = p * sz;
    vec2 c1 = floor(pos - 0.5) + 0.5;
    vec2 f = pos - c1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 o12 = w2 / w12;
    vec2 p0 = (c1 - 1.0) / sz;
    vec2 p3 = (c1 + 2.0) / sz;
    vec2 p12 = (c1 + o12) / sz;
    float r = 0.0;
    r += textureLod(t, vec2(p0.x, p0.y), 0.0).r * (w0.x * w0.y);
    r += textureLod(t, vec2(p12.x, p0.y), 0.0).r * (w12.x * w0.y);
    r += textureLod(t, vec2(p3.x, p0.y), 0.0).r * (w3.x * w0.y);
    r += textureLod(t, vec2(p0.x, p12.y), 0.0).r * (w0.x * w12.y);
    r += textureLod(t, vec2(p12.x, p12.y), 0.0).r * (w12.x * w12.y);
    r += textureLod(t, vec2(p3.x, p12.y), 0.0).r * (w3.x * w12.y);
    r += textureLod(t, vec2(p0.x, p3.y), 0.0).r * (w0.x * w3.y);
    r += textureLod(t, vec2(p12.x, p3.y), 0.0).r * (w12.x * w3.y);
    r += textureLod(t, vec2(p3.x, p3.y), 0.0).r * (w3.x * w3.y);
    return r;
}

vec3 sample_cur(vec2 s) {
    float y, cb, cr;
    if (lod > 0.0) {
        y = textureLod(tex_y, s, lod).r;
        cb = textureLod(tex_u, s, max(lod - 1.0, 0.0)).r;
        cr = textureLod(tex_v, s, max(lod - 1.0, 0.0)).r;
    } else if (quality >= 1) {
        y = catmull(tex_y, s);
        if (quality >= 2) { cb = catmull(tex_u, s); cr = catmull(tex_v, s); }
        else { cb = textureLod(tex_u, s, 0.0).r; cr = textureLod(tex_v, s, 0.0).r; }
    } else {
        y = textureLod(tex_y, s, 0.0).r;
        cb = textureLod(tex_u, s, 0.0).r;
        cr = textureLod(tex_v, s, 0.0).r;
    }
    return vec3(y, cb, cr);
}

vec3 sample_pool(vec2 s, int layer) {
    vec3 c = vec3(s, float(layer));
    return vec3(textureLod(pool_y, c, 0.0).r, textureLod(pool_u, c, 0.0).r, textureLod(pool_v, c, 0.0).r);
}

bool covered(vec2 s) {
    return all(greaterThanEqual(s, edge)) && all(lessThanEqual(s, 1.0 - edge));
}

// 到画面边缘的距离, 以羽化带宽度为单位: >=1 表示已在羽化带之外(深处)
float edge_dist(vec2 s) {
    return min(min(s.x, 1.0 - s.x) / feather.x, min(s.y, 1.0 - s.y) / feather.y);
}

// 某一帧(输出 uv → 该帧 uv 的矩阵 m)在当前输出位置的权重: 帧外为 0, 深处为 1, 边缘内侧 feather 宽度里渐变。
// 只有「这一侧的帧边缘落在输出画面之内」(即该侧确有缺口)才渐变; 边缘在输出画面之外时那里不会出现接缝, 权重保持 1,
// 免得无缺口的一侧最外圈像素被无谓地换成别的帧。缺口宽度从 0 增到 2 个纹素之间权重连续过渡。
float cover_weight(mat3 m, vec2 s) {
    if (!covered(s)) return 0.0;
    float sl = (m * vec3(0.0, uv.y, 1.0)).x;     // 输出画面的左/右/上/下边缘落在该帧坐标的位置
    float sr = (m * vec3(1.0, uv.y, 1.0)).x;
    float st = (m * vec3(uv.x, 0.0, 1.0)).y;
    float sb = (m * vec3(uv.x, 1.0, 1.0)).y;
    vec2 gate = 8.0 * edge;                      // 2 个纹素
    float w = 1.0;
    w *= 1.0 - smoothstep(0.0, gate.x, -sl)      * (1.0 - smoothstep(0.0, feather.x, s.x));
    w *= 1.0 - smoothstep(0.0, gate.x, sr - 1.0) * (1.0 - smoothstep(0.0, feather.x, 1.0 - s.x));
    w *= 1.0 - smoothstep(0.0, gate.y, -st)      * (1.0 - smoothstep(0.0, feather.y, s.y));
    w *= 1.0 - smoothstep(0.0, gate.y, sb - 1.0) * (1.0 - smoothstep(0.0, feather.y, 1.0 - s.y));
    return w;
}

void main() {
    bool right = uv.x >= split;
    mat3 m = right ? m_right : m_left;
    vec2 s = (m * vec3(uv, 1.0)).xy;
    bool fill = right && nb_count > 0;
    float wc = covered(s) ? ((fill && edge_dist(s) < 1.0) ? cover_weight(m, s) : 1.0) : 0.0;
    vec3 yuv;
    if (wc >= 1.0) {
        yuv = sample_cur(s);                                    // 绝大多数像素: 直接取当前帧
    } else {
        // 去抖把画面移开后边缘露出了缺口。级联羽化: 底色是延伸边缘像素, 从最远的相邻帧到最近的相邻帧逐层叠加,
        // 最后叠当前帧, 每层的权重是该帧自己的覆盖权重(边缘处为 0, 往里渐变到 1)。
        // 这样每一处帧与帧的交界都是渐变, 不会出现一条硬接缝。
        yuv = sample_cur(clamp(s, 0.0, 1.0));
        if (fill) {
            for (int k = nb_count - 1; k >= 0; k--) {
                vec2 sk = (nb_mat[k] * vec3(uv, 1.0)).xy;
                float wk = cover_weight(nb_mat[k], sk);
                if (wk > 0.0) yuv = mix(yuv, sample_pool(sk, nb_layer[k]), wk);
            }
        }
        if (wc > 0.0) yuv = mix(yuv, sample_cur(s), wc);
    }
    vec3 rgb = yuv_mat * ((yuv * tex_scale - yuv_off) * yuv_scale);
    // 量化到 8 位前加 ±0.5 LSB 的静态抖动, 避免暗部渐变出现色带
    float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    rgb += (n - 0.5) / 255.0;
    frag = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
"""

FRAG_OSD = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(tex, uv); }
"""


def default_window_size(info: "VideoInfo"):
    """屏幕可用区域内取最大的、与视频同宽高比的窗口, 不超过视频原始分辨率。"""
    dw, dh = pygame.display.get_desktop_sizes()[0]
    avail_w, avail_h = int(dw * 0.92), int(dh * 0.88)
    if sys.platform == "win32":     # 用去掉任务栏的工作区, 再扣掉窗口边框和标题栏
        try:
            from ctypes import wintypes
            r = wintypes.RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0)    # SPI_GETWORKAREA
            avail_w, avail_h = r.right - r.left - 16, r.bottom - r.top - 72
        except Exception:
            pass
    w = max(320, min(avail_w, int(avail_h * info.aspect), info.width))
    return w, round(w / info.aspect)


def choose_lod(scale: float) -> float:
    """缩小显示时的 mip 层级(scale = 每个输出像素对应的源像素数)。

    依据: 在真实 4K 帧上与「面积平均」参照比较, lod=log2(scale) 在 1.25~3 倍缩小时偏软(锐度只有参照的 0.4~0.8);
    lod=0.5·log2(scale) 在 1.25~4 倍内锐度 0.96~1.10、PSNR 最高。放大/接近 1:1 时用双三次(lod=0)。
    """
    if scale <= 1.15:
        return 0.0
    return 0.5 * math.log2(scale) if scale <= 4 else math.log2(scale) - 1.0


def fit_rect(win_w: int, win_h: int, aspect: float):
    """在窗口内按显示宽高比居中取最大矩形, 返回 GL 坐标 (x, y, w, h), 原点在左下。"""
    if win_w / win_h > aspect:
        h, w = win_h, round(win_h * aspect)
    else:
        w, h = win_w, round(win_w / aspect)
    return (win_w - w) // 2, (win_h - h) // 2, w, h


def gl_mat(m: np.ndarray) -> bytes:
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()      # GLSL 按列主序


def log_environment():
    try:
        libs = ", ".join(f"{k}={'.'.join(map(str, v))}" for k, v in av.library_versions.items())
    except Exception:
        libs = "?"
    log.info("系统: %s | Python %s | %d 逻辑核 | frozen=%s", platform.platform(), sys.version.split()[0],
             os.cpu_count() or 0, getattr(sys, "frozen", False))
    log.info("库: av %s [%s] | OpenCV %s | numpy %s | pygame %s (SDL %s) | moderngl %s", av.__version__, libs,
             cv2.__version__, np.__version__, pygame.version.ver, ".".join(map(str, pygame.get_sdl_version())),
             moderngl.__version__)
    log.info("opencv 线程数 %d | stabcore 位于 %s", cv2.getNumThreads(), getattr(stabcore, "__file__", "?"))


def log_video(info: "VideoInfo"):
    log.info("视频: %s", info.path)
    log.info("  %dx%d  %.3f fps  %.2fs  %d-bit  %s-range  colorspace=%d  显示宽高比 %.4f  音频=%s", info.width, info.height,
             info.fps, info.duration, info.bits, "full" if info.full_range else "limited", info.colorspace, info.aspect,
             "有" if info.has_audio else "无")


def log_gl(ctx: moderngl.Context, vsync: str, size):
    info = ctx.info
    log.info("OpenGL: %s | %s | %s", info.get("GL_VENDOR"), info.get("GL_RENDERER"), info.get("GL_VERSION"))
    log.info("GL 上限: 最大纹理 %s, 视口 %s | SDL 视频驱动 %s | 窗口 %s | vsync %s", info.get("GL_MAX_TEXTURE_SIZE"),
             info.get("GL_MAX_VIEWPORT_DIMS"), pygame.display.get_driver(), size, vsync)
    log.info("显示器分辨率: %s", pygame.display.get_desktop_sizes())


class Renderer:
    def __init__(self, ctx: moderngl.Context, info: VideoInfo):
        self.ctx, self.info = ctx, info
        w, h = info.width, info.height
        dtype = "f1" if info.bits == 8 else "nu2"
        self.tex = [ctx.texture((w, h), 1, dtype=dtype),
                    ctx.texture(((w + 1) // 2, (h + 1) // 2), 1, dtype=dtype),
                    ctx.texture(((w + 1) // 2, (h + 1) // 2), 1, dtype=dtype)]
        for t in self.tex:
            t.repeat_x = t.repeat_y = False
        self.mips = None            # 当前纹理的 mip 链状态: None=未建, True=已建
        self._filter_mips = None
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_VIDEO)
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4"))
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "pos")])
        mat, off, scale = yuv_params(info.colorspace, info.full_range, info.bits, info.height)
        self.prog["yuv_mat"].write(gl_mat(mat))
        self.prog["yuv_off"].value = off
        self.prog["yuv_scale"].value = scale
        self.prog["tex_scale"].value = 1.0 if info.bits == 8 else 65535 / ((1 << info.bits) - 1)
        for i, name in enumerate(("tex_y", "tex_u", "tex_v", "pool_y", "pool_u", "pool_v")):
            self.prog[name].value = i
        self.prog["edge"].value = (0.25 / w, 0.25 / h)
        self.prog["feather"].value = (FEATHER_PX / w, FEATHER_PX / h)
        self._init_pool()

    def _init_pool(self):
        """相邻帧纹理池: 三个纹理数组(Y/U/V), 每层存一帧; 帧 idx 放在第 idx % 层数 层。全幅稳定时用它补画面边缘的缺口。"""
        w, h = self.info.width, self.info.height
        cw, ch = (w + 1) // 2, (h + 1) // 2
        dtype = "f1" if self.info.bits == 8 else "nu2"
        bpp = 1 if self.info.bits == 8 else 2
        try:
            self.pool = [self.ctx.texture_array((w, h, POOL_LAYERS), 1, dtype=dtype),
                         self.ctx.texture_array((cw, ch, POOL_LAYERS), 1, dtype=dtype),
                         self.ctx.texture_array((cw, ch, POOL_LAYERS), 1, dtype=dtype)]
            self.pool_ok = True
            log.info("全幅稳定: 相邻帧纹理池 %d 层, 显存约 %.0f MB", POOL_LAYERS,
                     (w * h + 2 * cw * ch) * POOL_LAYERS * bpp / 1e6)
        except Exception as e:
            log.warning("相邻帧纹理池分配失败(显存不足?), 缺口只能用边缘延伸: %s", e)
            self.pool = [self.ctx.texture_array((1, 1, 1), 1, dtype=dtype) for _ in range(3)]      # 占位, 采样器必须有纹理
            self.pool_ok = False
        for t in self.pool:
            t.repeat_x = t.repeat_y = False
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.pool_idx = [-1] * POOL_LAYERS

    def pool_has(self, idx: int) -> bool:
        return self.pool_ok and self.pool_idx[idx % POOL_LAYERS] == idx

    def pool_put(self, idx: int, planes):
        if self.pool_ok:
            layer = idx % POOL_LAYERS
            for t, p in zip(self.pool, planes):
                t.write(p, viewport=(0, 0, layer, p.shape[1], p.shape[0], 1))
            self.pool_idx[layer] = idx

    def pool_reset(self):
        self.pool_idx = [-1] * POOL_LAYERS

    def upload(self, planes):
        for t, p in zip(self.tex, planes):
            t.write(p)
        self.mips = None

    def warmup(self):
        """用空帧在小的离屏缓冲里画一次(含 mip 路径), 让驱动提前编译着色器、分配显存, 避免开始播放的第一帧卡一下。"""
        fbo = self.ctx.simple_framebuffer((64, 36))
        fbo.use()
        for t in self.tex:
            t.write(bytes(t.size[0] * t.size[1] * (1 if t.dtype == "f1" else 2)))
        ident = np.eye(3)
        for lod in (0.0, 1.0):
            self.mips = None
            self.draw((0, 0, 64, 36), ident, ident, 0.0, lod, 2, nb=[(0, ident)])
        self.ctx.finish()
        fbo.release()
        self.mips = None

    def _set_filters(self, use_mips: bool):
        if self._filter_mips != use_mips:
            f = moderngl.LINEAR_MIPMAP_LINEAR if use_mips else moderngl.LINEAR
            for t in self.tex:
                t.filter = (f, moderngl.LINEAR)
            self._filter_mips = use_mips

    def draw(self, rect, m_left, m_right, split: float, lod: float, quality: int, nb=()):
        """nb: [(纹理池层号, 输出 uv → 该相邻帧 uv 的 3×3 矩阵)], 最多 6 个, 仅去抖一侧用来补缺口。"""
        if lod > 0 and not self.mips:
            for t in self.tex:
                t.build_mipmaps()
            self.mips = True
        self._set_filters(lod > 0)
        for i, t in enumerate((*self.tex, *self.pool)):
            t.use(location=i)
        p = self.prog
        p["m_left"].write(gl_mat(m_left))
        p["m_right"].write(gl_mat(m_right))
        nb = tuple(nb)[:len(FILL_OFFSETS)]
        p["nb_count"].value = len(nb)
        if nb:
            mats, layers = np.zeros((len(FILL_OFFSETS), 9), "f4"), np.zeros(len(FILL_OFFSETS), "i4")
            for k, (layer, m) in enumerate(nb):
                mats[k], layers[k] = np.ascontiguousarray(m.T, "f4").ravel(), layer
            p["nb_mat"].write(mats.tobytes())
            p["nb_layer"].write(layers.tobytes())
        p["split"].value = split
        p["lod"].value = lod
        p["quality"].value = quality
        self.ctx.viewport = rect
        self.ctx.disable(moderngl.BLEND)
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def release(self):
        for o in (*self.tex, *self.pool, self.vao, self.vbo, self.prog):
            o.release()
        self.tex = []


class Overlay:
    """pygame 绘制的 RGBA 小图(文字、进度条) → 纹理 → 带 alpha 叠加。"""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_OSD)
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4"))
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "pos")])
        self.prog["tex"].value = 0
        self.tex, self.size = None, (0, 0)

    def set_surface(self, surf: pygame.Surface):
        size = surf.get_size()
        if self.tex is None or size != self.size:
            if self.tex is not None:
                self.tex.release()
            self.tex, self.size = self.ctx.texture(size, 4), size
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex.write(pygame.image.tostring(surf, "RGBA", False))

    def draw(self, x: int, y_top: int, win_h: int):
        w, h = self.size
        self.ctx.viewport = (x, win_h - y_top - h, w, h)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.tex.use(location=0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


# ───────────────────────────── 播放器 ─────────────────────────────

def fmt_time(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


HELP_LINES = (
    "Space   play / pause",
    "Left/Right   seek -5s / +5s   (Shift: 30s)",
    "S   stabilization on / off      C   split compare",
    "[ / ]   stabilization preset weaker / stronger (smoothing + correction range)",
    "- / =   correction range smaller / larger (fine tune)",
    "X   fill picture-edge gaps from neighbouring frames on / off",
    "1   native 1:1 pixels (shows the center of the frame)   G   render quality",
    "Up/Down   volume     M   mute     , / .   A/V offset",
    "D   next audio output device (if you hear nothing)",
    "F / F11 / Alt+Enter / double-click   fullscreen (Esc leaves it)",
    "P   screenshot   I   stats   H   help",
    "Q / Esc   quit      drop a file to open it",
)


class Player:
    def __init__(self, info: VideoInfo, args):
        self.info, self.args = info, args
        self.preset_i = min(max(args.preset, 1), len(PRESETS)) - 1
        self.sigma_sec, self.zoom = PRESETS[self.preset_i][1:]
        if args.sigma is not None:
            self.sigma_sec = args.sigma
        if args.crop is not None:
            self.zoom = args.crop
        self.zoom = min(max(self.zoom, CROP_MIN), CROP_MAX)
        self.cur_stab: stabcore.Stabilizer | None = None
        self.plan_buffer()
        self.analysis_width = args.analysis_width
        self.initial_buffer = True            # 首次开播(或打开新文件)缓冲到目标深度; seek 后只缓冲到够前瞻
        self.buffer_progress = 0.0
        self.clock_off = self.clock_t = None        # 音频时钟相对系统时钟的平滑偏移
        self.audio_stalled = False                  # 音频设备停摆: 之后改用系统时钟, 直到换设备或打开新文件
        self.mode = args.mode
        self.quality_i = QUALITIES.index(args.quality)
        self.stats_on, self.help_on = args.stats, False
        self.av_offset = args.av_offset / 1000
        self.quit = self.fullscreen = self.paused = False
        self.buffering = True
        self.ended = False
        self.dirty = True
        self.t_frozen = self.base_t = self.base_wall = self.seek_target = 0.0
        self.pipe: VideoPipeline | None = None
        self.cur: FrameItem | None = None
        self.toast, self.toast_until = "", 0.0
        self.mouse_t = self.last_click = 0.0
        self.cursor_hidden = False
        self.osd_live = False
        self.need_upload = False
        self.rect = (0, 0, 1, 1)
        self.native = False                 # 1:1 原生像素模式
        self.fill = not args.no_fill        # 全幅稳定: 用相邻帧补缺口
        self.pool_owner = None              # 纹理池里的帧属于哪条流水线(seek 后 idx 重新从 0 数, 必须清池)
        self.cur_pipe = None
        self.margin_zoom = None             # 稳定器当前使用的校正范围倍数
        self.sigma_dirty = True
        self.limited = 0.0                  # 软限幅触发比例(滑动平均)
        self.sample_gpu = False
        self.lod = 0.0
        self.st = self._new_stats()
        self.stat_lines: tuple = ()
        self.seek_wall = 0.0
        self._init_window()
        self.renderer = Renderer(self.ctx, info)
        self.renderer.warmup()
        self.overlay_top, self.overlay_bar = Overlay(self.ctx), Overlay(self.ctx)
        self.font = pygame.font.Font(None, 22)
        self.audio = AudioOut(info.path, args.audio_device) if info.has_audio and not args.no_audio else None
        if self.audio and not self.audio.ok:
            self.audio = None
        if self.audio:
            self.audio.muted = args.mute
        self.play_started = None
        log.info("参数: 预设=%s sigma=%.2fs 校正范围倍数=%.2f mode=%s quality=%s 音频=%s",
                 PRESETS[self.preset_i][0], self.sigma_sec, self.zoom, self.mode, QUALITIES[self.quality_i],
                 "开" if self.audio else "关")

    def plan_buffer(self):
        """帧缓冲 = 10 秒内容, 只受内存安全上限约束(给系统留余量, 其余都可以用)。"""
        info, args = self.info, self.args
        self.buffer_frames, note = stabres.buffer_frames(info.width, info.height, info.bits, info.fps, args.prebuffer, args.buffer_gb)
        log.info("资源: %d 逻辑核(解码与 OpenCV 用满, 解码/分析线程降一档优先级) | 帧缓冲 %s", stabres.usable_cores(), note)

    # ---- 窗口 ----
    def _init_window(self):
        pygame.init()
        pygame.display.set_caption(os.path.basename(self.info.path))
        for k, v in ((pygame.GL_CONTEXT_MAJOR_VERSION, 3), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                     (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)):
            pygame.display.gl_set_attribute(k, v)
        size = self.args.size or default_window_size(self.info)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        try:
            pygame.display.set_mode(size, flags, vsync=1)
            vsync = "请求成功"
        except pygame.error as e:
            log.warning("垂直同步不可用, 退回无 vsync: %s", e)
            pygame.display.set_mode(size, flags)
            vsync = "不可用"
        # SDL 默认开着文字输入(输入法): 中文输入法下字母键被输入法吃掉, 收不到按键事件, F/S/C 等快捷键全部失效。
        # 播放器只要按键、不要文字, 关掉后输入法不再介入窗口。
        pygame.key.stop_text_input()
        self.ctx = moderngl.create_context()
        log_gl(self.ctx, vsync, size)

    @staticmethod
    def _new_stats() -> dict:
        return dict(t0=time.perf_counter(), cpu0=time.process_time(), flips=0, new=0, dropped=0,
                    late_sum=0.0, late_n=0, late_max=0.0, up_sum=0.0, up_n=0, up_max=0.0,
                    draw_sum=0.0, draw_n=0, draw_max=0.0, q_min=10 ** 9, loop_max=0.0, last_loop=None, pe=[], nb_sum=0, nb_n=0)

    # ---- 时间与播放控制 ----
    def apply_params(self):
        """强度/校正范围变了: 通知流水线(前瞻帧数和帧缓冲容量随之变化)和正在显示的那一帧所属的稳定器。"""
        self.sigma_dirty = True
        self.dirty = True
        log.info("强度: 预设=%s sigma=%.2fs 校正范围倍数=%.2f(±%d px)", PRESETS[self.preset_i][0], self.sigma_sec, self.zoom, self.margin_px())

    def sync_stabilizer(self, zoom_eff: float):
        """把平滑强度和当前布局决定的校正范围倍数同步给稳定器(它们变了才同步, 同步会清空稳定器的缓存)。"""
        if not self.sigma_dirty and self.margin_zoom is not None and abs(zoom_eff - self.margin_zoom) < 1e-3:
            return
        self.margin_zoom, self.sigma_dirty = zoom_eff, False
        self.pipe.configure(self.sigma_sec, zoom_eff)
        if self.cur_stab is not None and self.cur_stab is not self.pipe.stab:
            self.cur_stab.configure(self.sigma_sec, zoom_eff)
        log.info("稳定器: sigma=%.2fs 校正范围倍数 %.3f 前瞻 %d 帧, 帧缓冲 %d 帧", self.sigma_sec, zoom_eff,
                 self.pipe.stab.lookahead(), self.pipe.capacity)

    def media_time(self) -> float:
        if self.paused or self.buffering or self.ended:
            return self.t_frozen
        now = time.perf_counter()
        if not self.audio or self.audio_stalled:
            return self.base_t + now - self.base_wall
        if self.audio.stalled():
            # 时钟靠设备回调推进, 设备不出声就永远不动, 画面会冻住。从冻结处接着走系统时钟, 画面照常播放。
            self.audio_stalled = True
            self.base_t, self.base_wall = self.audio.position() - self.av_offset, now
            log.warning("音频设备 1 秒以上没有取走数据, 时钟退回系统时钟(画面照常播放, 声音无法同步; 可按 D 换设备)")
            return self.base_t
        # 音频位置按回调块(约 21ms)阶梯更新且随调度抖动, 直接拿来选帧会让帧的显示时刻忽早忽晚, 看起来像卡顿/抖动。
        # 所以只取它相对系统时钟的偏移, 做 τ=0.5s 的低通; 时间本身走系统时钟。偏移突变(seek、换设备)时直接重置。
        sample = self.audio.position() - self.av_offset - now
        if self.clock_off is None or abs(sample - self.clock_off) > 0.1:
            self.clock_off = sample
        else:
            self.clock_off += (1 - math.exp(-(now - self.clock_t) / 0.5)) * (sample - self.clock_off)
        self.clock_t = now
        return now + self.clock_off

    def seek(self, t: float, initial: bool = False):
        info = self.info
        self.initial_buffer = initial
        t = min(max(t, 0.0), max(info.duration - 1.0 / info.fps, 0.0))
        if self.pipe:
            self.pipe.stop(join=False)      # 不等旧线程退出, 否则 seek 时主线程会卡住
        log.info("seek: %.2fs -> %.2fs", self.t_frozen if self.pipe is None else self.media_time(), t)
        self.pipe = VideoPipeline(info, t, self.analysis_width, self.sigma_sec, self.margin_zoom or self.zoom, self.buffer_frames,
                                  self.args.decode_threads)
        self.pipe.start()
        self.clock_off = None
        self.buffering, self.ended = True, False
        self.seek_target = self.t_frozen = t
        self.seek_wall = time.perf_counter()
        if self.audio:
            self.audio.pause()
            self.audio.start(t)
        self.dirty = True

    def _begin_play(self):
        self.buffering = False
        if self.play_started is None:
            self.play_started = time.perf_counter()
        self.clock_off = None
        self.base_t, self.base_wall = self.seek_target, time.perf_counter()
        have = len(self.pipe.queue)
        log.info("缓冲完成: 等待 %.0f ms, 已缓冲 %d 帧 ≈ %.1f 秒 (前瞻需求 %d 帧)", (self.base_wall - self.seek_wall) * 1000,
                 have, have / self.info.fps, self.pipe.stab.lookahead())
        if self.audio and not self.paused:
            self.audio.play()

    def toggle_pause(self):
        if self.ended:
            self.seek(0.0)
            self.paused = False
            return
        if self.paused:
            self.paused = False
            if not self.buffering:
                self.clock_off = None
                self.base_t, self.base_wall = self.t_frozen, time.perf_counter()
                if self.audio:
                    self.audio.play()
        else:
            self.t_frozen = self.media_time()
            self.paused = True
            if self.audio:
                self.audio.pause()
        log.info("%s @ %.2fs", "暂停" if self.paused else "继续播放", self.t_frozen)
        self.say("Paused" if self.paused else "Playing")

    def update(self) -> bool:
        """推进播放状态: 缓冲就绪检测、按时间戳取帧、片尾检测。取到新帧返回 True。"""
        p = self.pipe
        if p.error is not None and not p.queue:
            self.say(f"Decode error: {p.error}", 8)
            return False
        if self.buffering:
            have = len(p.queue)
            target = min(self.buffer_frames if self.initial_buffer else p.stab.lookahead() + 1, p.capacity)
            self.buffer_progress = min(have / target, 1.0)
            waited = time.perf_counter() - self.seek_wall
            full = have >= target or p.traj.eof or (waited >= self.args.prebuffer_wait and self.initial_buffer)
            if not (p.prefilled() and full):
                return False
            self._begin_play()
        t = self.media_time()
        tol = 0.5 / self.info.fps if self.paused else 0.002
        item, dropped = p.take(t + tol)
        if item is not None:
            self.cur, self.cur_stab, self.cur_pipe, self.need_upload = item, p.stab, p, True
            s = self.st
            s["new"] += 1
            s["dropped"] += dropped
            if not self.paused:
                late = (t - item.pts) * 1000        # 取到的帧比时钟晚了多少: 理想 0~一帧时长, 持续偏大说明跟不上
                s["late_sum"] += late
                s["late_n"] += 1
                s["late_max"] = max(s["late_max"], late)
            if dropped:
                log.warning("丢帧 %d (t=%.2fs, 队列 %d)", dropped, t, len(p.queue))
        if (not self.paused and not self.ended and self.cur is not None and p.drained()
                and t >= self.cur.pts + 1.0 / self.info.fps):
            self.ended = True
            self.t_frozen = self.cur.pts + 1.0 / self.info.fps
            if self.audio:
                self.audio.pause()
            log.info("播放结束 @ %.2fs", self.t_frozen)
            self.say("End - press Space to replay", 5)
        return item is not None

    # ---- 绘制 ----
    def say(self, text: str, secs: float = 2.0):
        self.toast, self.toast_until = text, time.perf_counter() + secs
        self.dirty = True

    def margin_px(self) -> int:
        """允许的最大校正量(单边, 源像素)。"""
        return round(self.info.width * (1 - 1 / self.zoom) / 2)

    def layout(self, w: int, h: int):
        """返回 (视频在窗口里的矩形, 该矩形覆盖的源像素范围, 稳定器限制校正量用的等效余量倍数)。
        画面永远是完整的(不裁切、不放大), 按比例缩放到输出大小: 4K 输出 1:1, 2560 宽输出 2/3, 1920 宽输出 1/2。"""
        info = self.info
        if self.native:
            # 1:1 原生像素: 每个窗口像素对应一个源像素, 只显示画面正中央放得下的部分(窗口比视频大就是整幅)。
            vw, vh = min(w, info.width), min(h, info.height)
            return ((w - vw) // 2, (h - vh) // 2, vw, vh), (vw, vh), max(self.zoom, min(info.width / vw, info.height / vh))
        return fit_rect(w, h, info.aspect), (info.width, info.height), self.zoom

    def fill_sync(self):
        """把用得到的相邻帧提前传进 GPU 的纹理池: 当前帧之后 FILL_RADIUS 帧内、还没在池里的都传(稳态下每帧只新增一帧)。"""
        r = self.renderer
        if not (self.fill and r.pool_ok):
            return
        if self.pool_owner is not self.cur_pipe:
            r.pool_reset()
            self.pool_owner = self.cur_pipe
        for it in self.cur_pipe.peek_upto(self.cur.idx + FILL_RADIUS):
            if not r.pool_has(it.idx):
                r.pool_put(it.idx, it.planes)
        if not r.pool_has(self.cur.idx):
            r.pool_put(self.cur.idx, self.cur.planes)

    def view_params(self, vsize):
        """当前帧的 (左矩阵, 右矩阵, 分屏位置, 相邻帧列表)。vsize: 输出矩形覆盖的源像素范围。"""
        info, r = self.info, self.renderer
        ident = stabcore.view_matrix(1 + 0j, 0j, info.width, info.height, view=vsize)
        if self.mode == "orig":
            return ident, ident, 0.0, ()
        z, t = self.cur_stab.correction(self.cur.idx)
        self.limited = 0.98 * self.limited + 0.02 * self.cur_stab.limit
        stab = stabcore.view_matrix(z, t, info.width, info.height, view=vsize)
        nb = []
        if self.fill and r.pool_ok:
            for d, a, b in self.cur_stab.neighbor_maps(self.cur.idx, z, t, FILL_OFFSETS):
                if r.pool_has(self.cur.idx + d):
                    nb.append(((self.cur.idx + d) % POOL_LAYERS, stabcore.forward_matrix(a, b, info.width, info.height, vsize)))
        self.st["nb_sum"] += len(nb)
        self.st["nb_n"] += 1
        return (ident, stab, 0.5, nb) if self.mode == "split" else (stab, stab, 0.0, nb)

    def draw(self):
        w, h = pygame.display.get_window_size()
        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.rect, vsize, zoom_eff = self.layout(w, h)
        self.sync_stabilizer(zoom_eff)
        if self.cur is not None:
            s = self.st
            if self.need_upload:
                t0 = time.perf_counter()
                self.renderer.upload(self.cur.planes)
                ms = (time.perf_counter() - t0) * 1000
                s["up_sum"] += ms
                s["up_n"] += 1
                s["up_max"] = max(s["up_max"], ms)
                self.need_upload = False
            self.fill_sync()
            left, right, split, nb = self.view_params(vsize)
            lod = choose_lod(vsize[0] / self.rect[2])
            if self.sample_gpu:             # 每秒抽样一次: 用 finish() 把 GPU 执行时间也算进来
                ctx.finish()
            t0 = time.perf_counter()
            self.renderer.draw(self.rect, left, right, split, lod, 2 - self.quality_i, nb)
            if self.sample_gpu:
                ctx.finish()
                s["gpu_ms"] = (time.perf_counter() - t0) * 1000
                self.sample_gpu = False
            ms = (time.perf_counter() - t0) * 1000
            s["draw_sum"] += ms
            s["draw_n"] += 1
            s["draw_max"] = max(s["draw_max"], ms)
            self.lod = lod
        self.draw_osd(w, h)

    def _text_surface(self, lines, width=None):
        imgs = [self.font.render(s, True, (255, 255, 255)) for s in lines]
        pad = 8
        surf = pygame.Surface((width or max(i.get_width() for i in imgs) + 2 * pad,
                               sum(i.get_height() + 2 for i in imgs) + 2 * pad), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 150))
        y = pad
        for i in imgs:
            surf.blit(i, (pad, y))
            y += i.get_height() + 2
        return surf

    def draw_osd(self, w: int, h: int):
        now = time.perf_counter()
        lines = []
        if now < self.toast_until:
            lines.append(self.toast)
        if self.help_on:
            lines += HELP_LINES
        if self.stats_on:
            lines += self.stats_lines()
        self.osd_live = now < self.toast_until or self.stats_on or now - self.mouse_t < 2.5
        if lines:
            self.overlay_top.set_surface(self._text_surface(lines))
            self.overlay_top.draw(12, 12, h)
        if self.paused or self.ended or self.buffering or now - self.mouse_t < 2.5:
            bar_h = 44
            surf = pygame.Surface((w, bar_h), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 140))
            frac = min(max(self.media_time() / self.info.duration, 0.0), 1.0) if self.info.duration else 0.0
            pygame.draw.rect(surf, (90, 90, 90), (20, 28, w - 40, 6))
            pygame.draw.rect(surf, (255, 255, 255), (20, 28, (w - 40) * frac, 6))
            label = f"{fmt_time(self.media_time())} / {fmt_time(self.info.duration)}"
            tags = [("STAB" if self.mode == "stab" else "ORIG" if self.mode == "orig" else "SPLIT"),
                    f"{PRESETS[self.preset_i][0]} (smooth {self.sigma_sec:.2f}s)",
                    "1:1" if self.native else f"range ±{self.margin_px()}px", "fill" if self.fill else "no-fill",
                    f"out {self.rect[2]}x{self.rect[3]}"]
            if self.buffering:
                tags.insert(0, f"buffering {self.buffer_progress * 100:.0f}%")
            surf.blit(self.font.render(label + "    " + "   ".join(tags), True, (255, 255, 255)), (20, 5))
            self.overlay_bar.set_surface(surf)
            self.overlay_bar.draw(0, h - bar_h, h)

    def stats_lines(self):
        return self.stat_lines or ("collecting...",)

    def tick_stats(self, now: float):
        """每秒汇总一次: 写日志, 同时刷新屏幕统计(按 I 显示)。"""
        s, p = self.st, self.pipe
        el = now - s["t0"]
        if el < 1.0:
            return
        avg = lambda total, n: total / n if n else 0.0
        frames, fails = p.n_frames, p.n_fail
        if s.get("pipe") is p:
            frames, fails = frames - s["frames0"], fails - s["fail0"]
        u = self.audio.underruns if self.audio else 0
        a_under = u - s.get("under0", 0)
        mem_total, mem_avail = stabres.memory_info()
        late_avg = avg(s["late_sum"], s["late_n"])
        pe = np.array(s["pe"]) if s["pe"] else np.zeros(1)
        jit_std, jit_max = float(pe.std()), float(np.abs(pe - pe.mean()).max())
        line1 = (f"display {s['flips'] / el:5.1f} fps   video {s['new'] / el:5.1f} fps   dropped {s['dropped']}   "
                 f"late avg/max {late_avg:3.0f}/{s['late_max']:3.0f} ms   judder std/max {jit_std:3.1f}/{jit_max:3.0f} ms")
        line2 = (f"analysis {p.ms_analysis:4.1f} ms   upload {avg(s['up_sum'], s['up_n']):4.1f} ms   "
                 f"draw {avg(s['draw_sum'], s['draw_n']):4.1f} ms   gpu {s.get('gpu_ms', 0.0):4.1f} ms   "
                 f"queue min {s['q_min'] if s['q_min'] < 10 ** 9 else 0}/{p.capacity}")
        line3 = (f"limiter {self.limited * 100:3.0f}%   estimate-fail {fails}/{frames}   quality {QUALITIES[self.quality_i]}   "
                 f"A/V offset {self.av_offset * 1000:+.0f} ms")
        self.stat_lines = (line1, line2, line3)
        w, h = pygame.display.get_window_size()
        log.info("stats t=%.1f | disp=%.1f vid=%.1f drop=%d late=%.0f/%.0f judder=%.1f/%.0f | up=%.1f(max %.1f) draw=%.1f(max %.1f) gpu=%.1f "
                 "ana=%.1f q_min=%d/%d loop_max=%.0f | est_fail=%d/%d lim=%.0f%% cpu=%.1f | aud_under=%d buf=%d pause=%d | "
                 "win=%dx%d view=%dx%d lod=%.2f | aw=%d q=%s free=%.1fG nb=%.1f",
                 self.t_frozen if self.paused or self.buffering else self.media_time(), s["flips"] / el, s["new"] / el,
                 s["dropped"], late_avg, s["late_max"], jit_std, jit_max, avg(s["up_sum"], s["up_n"]), s["up_max"],
                 avg(s["draw_sum"], s["draw_n"]), s["draw_max"], s.get("gpu_ms", 0.0), p.ms_analysis,
                 s["q_min"] if s["q_min"] < 10 ** 9 else 0, p.capacity, s["loop_max"] * 1000, fails, frames,
                 self.limited * 100, (time.process_time() - s["cpu0"]) / el, a_under, self.buffering, self.paused, w, h, self.rect[2], self.rect[3], self.lod,
                 self.analysis_width, QUALITIES[self.quality_i], (mem_avail or 0) / 1e9, avg(s["nb_sum"], s["nb_n"]))
        gpu_keep = s.get("gpu_ms", 0.0)
        self.st = self._new_stats()
        self.st.update(pipe=p, frames0=p.n_frames, fail0=p.n_fail, under0=u, gpu_ms=gpu_keep)
        self.sample_gpu = True

    def screenshot(self, path: str):
        x, y, w, h = self.rect
        data = self.ctx.screen.read(viewport=(x, y, w, h), components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1]
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.say(f"Saved {os.path.basename(path)}")

    # ---- 输入 ----
    def on_key(self, key: int, mod: int):
        shift = bool(mod & pygame.KMOD_SHIFT)
        step = 30.0 if shift else 5.0
        log.debug("按键 %s  (模式 %s, sigma %.2f, 范围倍数 %.2f)", pygame.key.name(key), self.mode, self.sigma_sec, self.zoom)
        if key in (pygame.K_q,) or (key == pygame.K_ESCAPE and not self.fullscreen):
            self.quit = True
        elif key == pygame.K_ESCAPE:
            self.toggle_fullscreen()
        elif key == pygame.K_SPACE:
            self.toggle_pause()
        elif key == pygame.K_RIGHT:
            self.seek(self.media_time() + step)
        elif key == pygame.K_LEFT:
            self.seek(self.media_time() - step)
        elif key == pygame.K_s:
            self.mode = "orig" if self.mode == "stab" else "stab"
            self.say("Stabilization OFF (original)" if self.mode == "orig" else "Stabilization ON")
        elif key == pygame.K_c:
            self.mode = "stab" if self.mode == "split" else "split"
            self.say("Split view: left original | right stabilized" if self.mode == "split" else "Stabilization ON")
        elif key in (pygame.K_RIGHTBRACKET, pygame.K_LEFTBRACKET):
            self.preset_i = min(max(self.preset_i + (1 if key == pygame.K_RIGHTBRACKET else -1), 0), len(PRESETS) - 1)
            name, self.sigma_sec, self.zoom = PRESETS[self.preset_i]
            self.apply_params()
            self.say(f"Stabilization {self.preset_i + 1}/{len(PRESETS)} {name}: smooth {self.sigma_sec:.2f}s, range +-{self.margin_px()}px")
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS):
            up = key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS)
            self.zoom = round(min(max(self.zoom + (CROP_STEP if up else -CROP_STEP), CROP_MIN), CROP_MAX), 2)
            self.apply_params()
            self.say(f"Correction range +-{self.margin_px()}px (source pixels)")
        elif key in (pygame.K_1, pygame.K_KP1):
            self.native = not self.native
            log.info("显示模式: %s", "1:1 原生像素" if self.native else "适应窗口")
            self.say("1:1 native pixels (center crop)" if self.native else "Fit to window")
        elif key == pygame.K_x:
            self.fill = not self.fill
            log.info("相邻帧补缺口: %s", "开" if self.fill else "关")
            self.say("Gap fill from neighbouring frames: ON" if self.fill else "Gap fill: OFF (edge stretch only)")
        elif key == pygame.K_g:
            self.quality_i = (self.quality_i + 1) % len(QUALITIES)
            self.say(f"Render quality: {QUALITIES[self.quality_i]}")
        elif key in (pygame.K_f, pygame.K_F11) or (key in (pygame.K_RETURN, pygame.K_KP_ENTER) and mod & pygame.KMOD_ALT):
            self.toggle_fullscreen()
        elif key == pygame.K_i:
            self.stats_on = not self.stats_on
        elif key == pygame.K_h:
            self.help_on = not self.help_on
        elif key == pygame.K_p:
            self.screenshot(f"stabplayer_{time.strftime('%H%M%S')}.png")
        elif key == pygame.K_m and self.audio:
            self.audio.muted = not self.audio.muted
            self.say("Muted" if self.audio.muted else "Unmuted")
        elif key == pygame.K_d and self.audio:
            i = self.audio.switch_device()
            self.clock_off, self.audio_stalled = None, False        # 换了设备, 重新信任音频时钟
            self.say(f"Audio device {i + 1}/{len(self.audio.names)}")
        elif key in (pygame.K_UP, pygame.K_DOWN) and self.audio:
            self.audio.volume = min(max(self.audio.volume + (0.1 if key == pygame.K_UP else -0.1), 0.0), 1.0)
            self.say(f"Volume {self.audio.volume * 100:.0f}%")
        elif key in (pygame.K_COMMA, pygame.K_PERIOD):
            self.av_offset += (0.01 if key == pygame.K_PERIOD else -0.01)
            self.clock_off = None
            self.say(f"A/V offset {self.av_offset * 1000:+.0f} ms  (positive = video later)")
        self.dirty = True

    def toggle_fullscreen(self):
        pygame.display.toggle_fullscreen()
        self.fullscreen = not self.fullscreen
        log.info("全屏 %s", "开" if self.fullscreen else "关")

    def _bar_seek(self, x: int):
        w = pygame.display.get_window_size()[0]
        if self.info.duration:
            self.seek(min(max((x - 20) / max(w - 40, 1), 0.0), 1.0) * self.info.duration)

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.quit = True
            elif ev.type == pygame.KEYDOWN:
                self.on_key(ev.key, ev.mod)
            elif ev.type == pygame.MOUSEMOTION:
                self.mouse_t = time.perf_counter()
                # 播放中下一帧本来就会重绘(带上进度条), 额外的重绘会在垂直同步下挤占翻页节拍、让画面节奏不匀; 只有静止时才需要立刻刷新
                self.dirty = self.dirty or self.paused or self.ended or self.buffering
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                now = time.perf_counter()
                self.mouse_t = now
                if ev.pos[1] >= pygame.display.get_window_size()[1] - 44:
                    self._bar_seek(ev.pos[0])
                elif now - self.last_click < 0.4:
                    self.toggle_fullscreen()
                    now = 0.0
                self.last_click = now
            elif ev.type in (pygame.TEXTINPUT, pygame.TEXTEDITING):
                log.debug("收到文字输入事件(输入法仍在介入窗口): %r", getattr(ev, "text", ""))
            elif ev.type == pygame.DROPFILE:
                self.open_file(ev.file)
            elif ev.type == pygame.VIDEORESIZE:
                log.info("窗口大小 -> %dx%d", ev.w, ev.h)
                self.dirty = True
            elif ev.type in (pygame.WINDOWEXPOSED, pygame.WINDOWSHOWN, pygame.WINDOWRESTORED):
                self.dirty = True

    def open_file(self, path: str):
        log.info("打开文件: %s", path)
        try:
            info = probe(path)
        except Exception as e:
            log.exception("无法打开 %s", path)
            self.say(f"Cannot open: {e}", 4)
            return
        log_video(info)
        if self.pipe:
            self.pipe.stop()
        if self.audio:
            self.audio.close()
        self.renderer.release()
        self.cur = self.cur_stab = None
        self.need_upload = False
        self.info = info
        self.renderer = Renderer(self.ctx, info)
        self.renderer.warmup()
        self.plan_buffer()
        self.audio = AudioOut(path, self.args.audio_device) if info.has_audio and not self.args.no_audio else None
        if self.audio and not self.audio.ok:
            self.audio = None
        self.clock_off, self.audio_stalled = None, False
        pygame.display.set_caption(os.path.basename(path))
        self.paused = False
        self.seek(0.0, initial=True)

    def run(self, hook=None):
        self.seek(self.args.start, initial=True)
        if self.args.fullscreen:
            self.toggle_fullscreen()
        last_draw = last_arm = 0.0
        try:
            while not self.quit:
                now = time.perf_counter()
                s = self.st
                if s["last_loop"] is not None:
                    s["loop_max"] = max(s["loop_max"], now - s["last_loop"])        # 主循环单次间隔: 大值=主线程被卡住
                s["last_loop"] = now
                s["q_min"] = min(s["q_min"], len(self.pipe.queue))
                if _crash_file and now - last_arm > 1.0:    # 看门狗: 主循环卡死 10 秒就把所有线程栈写进日志
                    faulthandler.dump_traceback_later(10, file=_crash_file)
                    last_arm = now
                self.tick_stats(now)
                if self.args.screenshot and self.play_started and now - self.play_started > self.args.screenshot_at:
                    self.draw()
                    self.screenshot(self.args.screenshot)
                    log.info("已保存截图 %s (输出 %dx%d)", self.args.screenshot, self.rect[2], self.rect[3])
                    self.args.screenshot = None
                if self.args.quit_after and self.play_started and now - self.play_started > self.args.quit_after:
                    log.info("到达 --quit-after, 退出")
                    self.quit = True
                self.handle_events()
                hide = self.fullscreen and not self.paused and now - self.mouse_t > 2.5
                if hide != self.cursor_hidden:
                    pygame.mouse.set_visible(not hide)
                    self.cursor_hidden = hide
                new_frame = self.update()
                now = time.perf_counter()
                # 只在有新帧、界面事件、或叠加层需要刷新(至多 10Hz)时才绘制; 其余时间让出 CPU/GIL 给解码线程
                if new_frame or self.dirty or ((self.osd_live or self.buffering) and now - last_draw > 0.1):
                    self.draw()
                    pygame.display.flip()
                    self.st["flips"] += 1
                    if new_frame and not self.paused:     # 显示呈现误差: 翻页完成时刻相对该帧时间戳; 其波动(std)就是观感上的节奏不匀
                        self.st["pe"].append((self.media_time() - self.cur.pts) * 1000)
                    self.dirty, last_draw = False, now
                else:
                    pygame.time.wait(2)
                if hook and hook(self):
                    break
        finally:
            faulthandler.cancel_dump_traceback_later()
            if self.pipe:
                self.pipe.stop()
            if self.audio:
                self.audio.close()
            pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="实时去抖视频播放器(4K 友好, 画质优先)")
    ap.add_argument("video", nargs="?", help="视频文件")
    ap.add_argument("--preset", type=int, default=DEFAULT_PRESET,
                    help="去抖强度预设 1~%d (%s), 默认 %d; 越强平滑越狠、校正范围越大" % (
                        len(PRESETS), " / ".join(n for n, _, _ in PRESETS), DEFAULT_PRESET))
    ap.add_argument("--sigma", type=float, help="覆盖预设里的平滑 σ(秒)")
    ap.add_argument("--prebuffer", type=float, default=10.0, help="开播前缓冲的内容时长(秒), 受 --buffer-gb 限制 (默认 10)")
    ap.add_argument("--prebuffer-wait", type=float, default=10.0, help="开播前最多等待多少秒(默认 10), 超时只要前瞻够用就开播")
    ap.add_argument("--buffer-gb", type=float, default=None, help="帧缓冲的内存预算 GB; 默认 = 可用内存减去给系统留的余量(至少 2GB 或总内存的 25%%), 缓冲 10 秒内容为止")
    ap.add_argument("--crop", type=float, help=f"覆盖预设里的校正范围倍数 ({CROP_MIN}~{CROP_MAX}); 范围 = 画面宽 × (1-1/倍数)/2。画面不会被裁切")
    ap.add_argument("--no-fill", action="store_true", help="不用相邻帧补边缘缺口(缺口只做边缘延伸)")
    ap.add_argument("--mode", choices=("stab", "orig", "split"), default="stab", help="起始显示模式: 去抖 / 原片 / 分屏对比")
    ap.add_argument("--quality", choices=QUALITIES, default="high", help="渲染质量, 默认最高; 显卡吃力时可降档(运行中按 G 也行)")
    ap.add_argument("--analysis-width", type=int, default=960, help="运动分析用的缩小图宽度(只影响估计精度和 CPU 占用, 不影响画面), 默认 960")
    ap.add_argument("--start", type=float, default=0.0, help="从第几秒开始")
    ap.add_argument("--size", type=lambda s: tuple(int(x) for x in s.lower().split("x")), help="初始窗口大小, 如 1600x900")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--mute", action="store_true", help="启动即静音(音频设备照常运行, 时钟不受影响)")
    ap.add_argument("--quit-after", type=float, help="开始播放后 N 秒自动退出(自动化测试用)")
    ap.add_argument("--screenshot", help="开始播放后 --screenshot-at 秒时把当前画面存成 PNG(自动化测试用)")
    ap.add_argument("--screenshot-at", type=float, default=3.0)
    ap.add_argument("--audio-device", help="音频输出设备名(--list-audio 查看)")
    ap.add_argument("--list-audio", action="store_true", help="列出音频输出设备后退出")
    ap.add_argument("--av-offset", type=float, default=0.0, help="音画同步微调(毫秒), 正值=画面晚于声音")
    ap.add_argument("--cv-threads", type=int, default=0, help="OpenCV 线程数(0=用满所有核)")
    ap.add_argument("--decode-threads", type=int, default=0, help="视频解码线程数(0=用满所有核)")
    ap.add_argument("--stats", action="store_true", help="启动时就叠加显示帧率/丢帧/耗时(日志里始终有逐秒统计)")
    ap.add_argument("--selftest", action="store_true", help="不开窗口, 测试本机的解码/GL 上传/绘制/音频性能并写入日志后退出")
    return ap


def ask_video() -> str:
    """没有命令行参数(双击 exe)时弹出系统文件选择框。"""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="选择要播放的视频", filetypes=[
            ("视频文件", "*.webm *.mkv *.mp4 *.mov *.avi *.m4v *.ts"), ("所有文件", "*.*")])
        root.destroy()
        return path
    except Exception:
        log.exception("文件选择框不可用")
        return ""


def selftest(path: str):
    """在隐藏窗口里量: 解码+分析吞吐、GL 上传带宽、1080p/4K 输出的绘制耗时、音频设备。结果全写进日志,
    用来在目标机器(NVIDIA / AMD / Intel 核显)上定位性能瓶颈, 不依赖肉眼观察。"""
    log.info("SELFTEST 开始")
    info = probe(path)
    log_video(info)
    W, H = info.width, info.height
    clock = time.perf_counter

    _, note = stabres.buffer_frames(W, H, info.bits, info.fps, 10.0)
    log.info("SELFTEST 资源: %d 逻辑核 | 帧缓冲 %s", stabres.usable_cores(), note)
    n, t_plane, t_ana, first = 0, 0.0, 0.0, None
    est = stabcore.MotionEstimator(W, H, 960, info.bits)
    with av.open(path) as c:
        vs = next(s for s in c.streams if s.type == "video")
        vs.thread_type = "AUTO"
        t_start = clock()
        for packet in c.demux(vs):
            try:
                frames = packet.decode()
            except av.error.FFmpegError:
                continue
            for f in frames:
                a = clock()
                planes = frame_planes(f, info.bits)
                b = clock()
                est.push(planes[0])
                t_plane, t_ana, n = t_plane + b - a, t_ana + clock() - b, n + 1
                first = first or planes
            if n >= 90:
                break
        wall = clock() - t_start
    prod_fps = n / wall
    log.info("SELFTEST 生产侧(解码+取平面+运动分析, 流水线并行): %d 帧 %.0f ms => %.1f fps; 其中取平面 %.1f ms/帧, 运动分析 %.1f ms/帧",
             n, wall * 1000, prod_fps, t_plane / n * 1000, t_ana / n * 1000)

    pygame.init()
    for k, v in ((pygame.GL_CONTEXT_MAJOR_VERSION, 3), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                 (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)):
        pygame.display.gl_set_attribute(k, v)
    pygame.display.set_mode((640, 360), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    ctx = moderngl.create_context()
    log_gl(ctx, "n/a(隐藏窗口)", (640, 360))
    r = Renderer(ctx, info)
    ups = []
    for _ in range(14):
        a = clock()
        r.upload(first)
        ctx.finish()
        ups.append(clock() - a)
    ups = sorted(ups[3:])
    mb = sum(p.nbytes for p in first) / 1e6
    up_ms = ups[len(ups) // 2] * 1000
    log.info("SELFTEST GL 上传 %.1f MB/帧: 中位 %.1f ms (%.0f MB/s), 最差 %.1f ms", mb, up_ms, mb / (up_ms / 1000), ups[-1] * 1000)

    main_ms, mips_needed = {}, {}
    for ow, oh in ((1920, 1080), (3840, 2160)):
        fbo = ctx.simple_framebuffer((ow, oh))
        fbo.use()
        scale = W / ow
        lod = choose_lod(scale)
        m = stabcore.view_matrix(complex(0.9998, 0.001), complex(3, 2), W, H, view=(W, H))
        r.upload(first)
        for t in r.tex:                     # 预热: 首次建 mip 链要分配显存, 不计入
            t.build_mipmaps()
        ctx.finish()
        a = clock()
        for _ in range(8):
            for t in r.tex:
                t.build_mipmaps()
            ctx.finish()
        mip_ms = (clock() - a) / 8 * 1000
        mips_needed[oh] = mip_ms if lod > 0 else 0.0
        for q, name in ((2, "high"), (0, "fast")):
            ts = []
            for _ in range(12):
                a = clock()
                r.draw((0, 0, ow, oh), m, m, 0.0, lod, q)
                ctx.finish()
                ts.append(clock() - a)
            ts = sorted(ts[2:])
            main_ms[(oh, name)] = ts[len(ts) // 2] * 1000
            log.info("SELFTEST 绘制 输出 %dx%d quality=%s: 中位 %.1f ms, 最差 %.1f ms (lod=%.2f)", ow, oh, name,
                     main_ms[(oh, name)], ts[-1] * 1000, lod)
        log.info("SELFTEST 生成 mipmap(输出 %dx%d 时每帧需要, 仅 lod>0 时): %.1f ms", ow, oh, mip_ms)
        fbo.release()

    if info.has_audio:
        a = AudioOut(path)
        if a.ok:
            a.start(0.0)
            time.sleep(0.5)
            a.play()
            time.sleep(0.6)
            log.info("SELFTEST 音频: 0.6s 后位置 %.3fs, 缓冲欠载 %d 次 (设备 %r)", a.position(), a.underruns, a.name)
            a.close()
    else:
        log.info("SELFTEST 文件没有音频流, 跳过音频测试")

    need = 1000 / info.fps
    for oh, label in ((1080, "1080p 窗口"), (2160, "4K 全屏")):
        render_ms = up_ms + main_ms[(oh, "high")] + mips_needed[oh]
        verdict = "够用" if render_ms < need * 0.7 and prod_fps > info.fps * 1.3 else "吃紧/可能掉帧"
        log.info("SELFTEST 结论[%s, quality=high]: 渲染侧 上传+绘制+mip ≈ %.1f ms/帧 (预算 %.1f ms), 生产侧 %.1f fps (需要 %.1f) => %s",
                 label, render_ms, need, prod_fps, info.fps, verdict)
    log.info("SELFTEST 结束")


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    setup_logging()
    # OpenCV 的线程池会空转: 默认 16 线程时稳态要占约 4 个核, 限制到 4 个核降到约 2.4 个核, 分析只慢 8%
    if args.cv_threads:
        cv2.setNumThreads(args.cv_threads)
    log_environment()
    log.info("命令行: %s", sys.argv)
    if args.list_audio:
        pygame.init()
        names = list_audio_devices()
        log.info("音频输出设备: %s", names)
        print("\n".join(names) or "(无音频输出设备)")
        return
    enable_dpi_awareness()
    try:
        path = args.video or ask_video()
        if not path:
            log.info("没有选择文件, 退出")
            return
        args.video = path
        if args.selftest:
            selftest(path)
            return
        info = probe(path)
        log_video(info)
        Player(info, args).run()
        log.info("正常退出")
    except SystemExit:
        raise
    except Exception as e:
        fatal("播放器异常退出", e)


if __name__ == "__main__":
    main()
