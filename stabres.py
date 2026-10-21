"""stabres: 资源相关的小工具(只依赖标准库)。

原则: 默认把资源用满、画质开到最高; 只守住一条底线 —— 不让内存耗尽(那会拖死整个系统)。
CPU / GPU 打满最多是慢一点, 系统不会死, 所以不做限制; 解码/分析线程降一档优先级, 让界面、渲染、音频先拿到 CPU。
"""
from __future__ import annotations

import ctypes
import os
import sys

MIN_BUFFER_FRAMES = 45
RESERVE_MIN = 2e9           # 给系统留的内存底线: 至少 2GB
RESERVE_FRACTION = 0.25     # 或总内存的 25%, 取大者


def usable_cores() -> int:
    """本进程实际能用的逻辑核数。Linux 上 os.cpu_count() 不尊重 CPU 亲和性(容器、taskset), 要用 sched_getaffinity。"""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 4


def memory_info() -> tuple[int | None, int | None]:
    """(物理内存总量, 当前可用) 字节; 取不到返回 None。"""
    try:
        if sys.platform == "win32":
            class MemStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32), ("total", ctypes.c_uint64),
                            ("avail", ctypes.c_uint64), ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64),
                            ("total_virt", ctypes.c_uint64), ("avail_virt", ctypes.c_uint64), ("avail_ext", ctypes.c_uint64)]
            st = MemStatus()
            st.length = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.total), int(st.avail)
        elif sys.platform.startswith("linux"):
            vals = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    if k in ("MemTotal", "MemAvailable"):
                        vals[k] = int(v.split()[0]) * 1024
            if len(vals) == 2:
                return vals["MemTotal"], vals["MemAvailable"]
        elif sys.platform == "darwin":
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), None
    except Exception:
        pass
    return None, None


def lower_thread_priority():
    """把当前线程降到「低于正常」: 解码/分析线程把 CPU 让给界面、渲染、音频线程, 机器再忙也不会卡住界面。"""
    if sys.platform == "win32":
        try:
            k = ctypes.windll.kernel32
            k.GetCurrentThread.restype = ctypes.c_void_p
            k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            k.SetThreadPriority(k.GetCurrentThread(), -1)       # THREAD_PRIORITY_BELOW_NORMAL
        except Exception:
            pass


def buffer_frames(width: int, height: int, bits: int, fps: float, prebuffer_s: float, buffer_gb: float | None = None):
    """帧缓冲帧数 = 目标时长对应的帧数, 受内存安全上限约束: 可用内存减去给系统留的余量, 其余都可以用。
    返回 (帧数, 说明)。buffer_gb 为用户指定时直接按它算。"""
    frame_bytes = width * height * 3 // 2 * (1 if bits == 8 else 2)
    total, avail = memory_info()
    if buffer_gb is not None:
        budget, why = buffer_gb * 1e9, f"用户指定 {buffer_gb:g} GB"
    elif avail is None:
        budget, why = 2e9, "取不到可用内存, 保守按 2 GB"
    else:
        reserve = max(RESERVE_MIN, RESERVE_FRACTION * (total or 0))
        budget = max(avail - reserve, 0)
        why = f"可用 {avail / 1e9:.1f} GB 减去给系统留的 {reserve / 1e9:.1f} GB"
    by_time = round(prebuffer_s * fps)
    by_mem = int(budget / frame_bytes)
    frames = max(min(by_time, by_mem), MIN_BUFFER_FRAMES)
    return frames, (f"目标 {prebuffer_s:g} 秒({by_time} 帧), 内存可用于缓冲 {budget / 1e9:.1f} GB[{why}](每帧 {frame_bytes / 1e6:.1f} MB, "
                    f"最多 {by_mem} 帧) => {frames} 帧 ≈ {frames / fps:.1f} 秒, 约占 {frames * frame_bytes / 1e9:.2f} GB")
