#!/usr/bin/env python3
"""stabcore: 去抖算法核心(只依赖 numpy / OpenCV, 与解码、渲染、界面无关)。

  MotionEstimator  降采样灰度图 → 网格特征点 → LK 光流 → 覆盖度加权 RANSAC → 相邻帧相似变换
  Trajectory       累积相机路径; 带前瞻的局部二次回归平滑 → 每帧校正量
  Stabilizer       强/弱平滑在校正范围内插值, 限制最大校正量且不放回原始抖动
  view_matrix      校正量 + 输出范围 → 「输出 uv → 源纹理 uv」矩阵, 交给 GPU 着色器

坐标约定: 以画面中心为原点、像素为单位、y 向下; 相似变换 x' = z·x + t 用复数表示。
"""
from __future__ import annotations

import functools
import math
import threading

import cv2
import numpy as np

IDENTITY = (complex(1, 0), complex(0, 0))


def _fit_similarity(a: np.ndarray, b: np.ndarray, w: np.ndarray):
    """加权最小二乘拟合 b ≈ z·a + t (复数相似变换, 闭式解)。"""
    ws = w.sum()
    ma, mb = (w * a).sum() / ws, (w * b).sum() / ws
    da, db = a - ma, b - mb
    z = (w * np.conj(da) * db).sum() / (w * np.abs(da) ** 2).sum()
    return z, mb - z * ma


class MotionEstimator:
    """逐帧估计相邻两帧的全局运动 (prev → cur), 在降采样的灰度图上做。

    要点是让估计对前景有抗性: 全局运动应当被整幅画面支撑, 而不是被某个大刚体(舞者、车)支撑。
      · 选点: 每个网格单元取最强角点, 阈值用低的绝对下限, 不与全图最大值比
        (高对比前景会把相对阈值抬高, 把低纹理背景全部筛掉);
      · 打分: RANSAC 按粗网格区域给内点加权(每区域总权重 ∝ √点数), 被更多区域支撑的运动胜出。
    """

    BLOCK = 3
    FLOOR = 2e-5            # 单元最强角点的最低响应 (8 位灰度上的 cornerMinEigenVal), 更弱的交给 LK 反向检验
    REGIONS = (6, 4)
    HYPOTHESES = 250
    SCORE_POINTS = 300
    THRESH = 1.0            # 内点阈值 (分析图像素)
    MIN_TRACKED = 16
    MIN_INLIERS = 12
    MIN_REGIONS = 3
    MIN_RATIO = 0.15

    def __init__(self, width: int, height: int, analysis_width: int = 960, bits: int = 8):
        self.shift = bits - 8
        aw = min(analysis_width, width)
        ah = max(8, round(height * aw / width))
        self.size = (aw, ah)
        self.scale = width / aw
        self.center = complex((aw - 1) / 2, (ah - 1) / 2)
        self.hsize = (aw // 2, ah // 2)         # 特征点只是 LK 的起点, 在半分辨率上检测即可(cornerMinEigenVal 是最大单项且单线程)
        self.cell = max(6, aw // 68)            # 半分辨率下的网格单元边长
        self.ny, self.nx = self.hsize[1] // self.cell, self.hsize[0] // self.cell
        self.rng = np.random.default_rng(0)
        self.prev = None
        self.prev_pts = None
        self.last_inliers = self.last_tracked = 0
        self.last_p0 = self.last_p1 = self.last_mask = None     # 调试/可视化用

    def _features(self, img: np.ndarray) -> np.ndarray:
        eig = cv2.cornerMinEigenVal(cv2.resize(img, self.hsize, interpolation=cv2.INTER_AREA), self.BLOCK)
        c = self.cell
        blocks = (eig[: self.ny * c, : self.nx * c]
                  .reshape(self.ny, c, self.nx, c).transpose(0, 2, 1, 3).reshape(self.ny, self.nx, -1))
        arg = blocks.argmax(-1)
        val = np.take_along_axis(blocks, arg[..., None], -1)[..., 0]
        ys, xs = np.nonzero(val > self.FLOOR)
        a = arg[ys, xs]
        pts = (np.stack([xs * c + a % c, ys * c + a // c], 1) * 2 + 0.5).astype(np.float32)    # 半分辨率像素中心 → 分析图坐标
        return pts.reshape(-1, 1, 2)

    def _track(self, prev, img, pts):
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, pts, None, winSize=(21, 21), maxLevel=3, criteria=crit)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(img, prev, nxt, None, winSize=(21, 21), maxLevel=3, criteria=crit)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.abs(back - pts).reshape(-1, 2).max(1) < 0.5)
        return pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)

    def _fit(self, p0: np.ndarray, p1: np.ndarray):
        aw, ah = self.size
        a = (p0[:, 0] + 1j * p0[:, 1]) - self.center
        b = (p1[:, 0] + 1j * p1[:, 1]) - self.center
        rx, ry = self.REGIONS
        region = (np.minimum((p0[:, 1] / ah * ry).astype(int), ry - 1) * rx
                  + np.minimum((p0[:, 0] / aw * rx).astype(int), rx - 1))
        w = 1 / np.sqrt(np.bincount(region, minlength=rx * ry)[region])

        i = self.rng.integers(0, len(a), self.HYPOTHESES)
        j = self.rng.integers(0, len(a), self.HYPOTHESES)
        d = a[i] - a[j]
        keep = np.abs(d) > aw / 24          # 两点太近, 缩放/旋转病态
        i, j, d = i[keep], j[keep], d[keep]
        if not len(i):
            return None
        z = (b[i] - b[j]) / d
        t = b[i] - z * a[i]
        # 给假设打分: 在至多 SCORE_POINTS 个点上、用 float32 实数运算, 临时数组留在缓存里
        sel = self.rng.choice(len(a), self.SCORE_POINTS, replace=False) if len(a) > self.SCORE_POINTS else slice(None)
        ar, ai, br, bi, ws = (v.astype(np.float32) for v in (a[sel].real, a[sel].imag, b[sel].real, b[sel].imag, w[sel]))
        zr, zi, tr, ti = (v.astype(np.float32)[:, None] for v in (z.real, z.imag, t.real, t.imag))
        dx = br - (zr * ar - zi * ai + tr)
        dy = bi - (zr * ai + zi * ar + ti)
        best = ((dx * dx + dy * dy < self.THRESH ** 2).astype(np.float32) @ ws).argmax()
        z, t = z[best], t[best]
        for _ in range(3):
            mask = np.abs(b - (z * a + t)) < self.THRESH
            if mask.sum() < 3:
                return None
            z, t = _fit_similarity(a[mask], b[mask], w[mask])
        mask = np.abs(b - (z * a + t)) < self.THRESH
        self.last_inliers = int(mask.sum())
        self.last_p0, self.last_p1, self.last_mask = p0, p1, mask
        if (self.last_inliers < self.MIN_INLIERS or self.last_inliers < self.MIN_RATIO * len(a)
                or len(np.unique(region[mask])) < self.MIN_REGIONS):
            return None
        return z, t

    def push(self, luma: np.ndarray):
        """喂入一帧亮度平面。返回 (z, t, ok): 以画面中心为原点、原分辨率像素为单位的 prev→cur 相似变换。"""
        img = cv2.resize(luma, self.size, interpolation=cv2.INTER_AREA)
        if self.shift:
            img = (img >> self.shift).astype(np.uint8)
        prev, pts = self.prev, self.prev_pts
        self.prev = img
        self.prev_pts = self._features(img)
        self.last_inliers = self.last_tracked = 0
        self.last_p0 = self.last_p1 = self.last_mask = None
        if prev is None:
            return (*IDENTITY, True)
        if pts is None or len(pts) < self.MIN_TRACKED:
            return (*IDENTITY, False)
        p0, p1 = self._track(prev, img, pts)
        self.last_tracked = len(p0)
        if len(p0) < self.MIN_TRACKED:
            return (*IDENTITY, False)
        fit = self._fit(p0, p1)
        if fit is None:
            return (*IDENTITY, False)
        z, t = fit
        return z, t * self.scale, True      # 已是画面中心原点, 只需换算到原分辨率


class Trajectory:
    """相机路径: 逐帧累积的相似变换, 参数化为 (tx, ty, θ, ln s)。

    G_i 把第 0 帧坐标映射到第 i 帧坐标; 追加单步运动 M_i 时 G_i = M_i ∘ G_{i-1}。
    生产者线程追加, 渲染线程读取。
    """

    def __init__(self):
        self.p = np.zeros((1024, 4))
        self.n = 0
        self.eof = False
        self._t = complex(0, 0)
        self._lock = threading.Lock()

    def append(self, z: complex, t: complex):
        with self._lock:
            if self.n:
                self._t = z * self._t + t
                prev = self.p[self.n - 1]
                row = (self._t.real, self._t.imag, prev[2] + math.atan2(z.imag, z.real), prev[3] + math.log(abs(z)))
            else:
                row = (0.0, 0.0, 0.0, 0.0)
            if self.n == len(self.p):
                self.p = np.concatenate([self.p, np.zeros_like(self.p)])
            self.p[self.n] = row
            self.n += 1

    def pose(self, i: int):
        """G_i 的 (z, t): 第 0 帧坐标 → 第 i 帧坐标的相似变换(中心系, 复数)。"""
        with self._lock:
            tx, ty, th, ls = self.p[i]
        return complex(math.exp(ls) * math.cos(th), math.exp(ls) * math.sin(th)), complex(tx, ty)

    def snapshot(self):
        with self._lock:
            return self.p, self.n, self.eof

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def kernel(sigma: float):
        """平滑核: 高斯加权的局部二次回归在窗口中心的估计值(Savitzky–Golay 型)。

        相比纯高斯, 通带更平、截止更陡: 匀速、匀加速的有意运动(摇镜、变焦)不被磨平, 只去掉高频抖动。
        对称窗口下核为 w·(α + β·k²), 由 Σh=1、Σh·k²=0 解出 α、β。
        """
        sigma = max(sigma, 1.0)
        radius = math.ceil(3 * sigma)
        k = np.arange(-radius, radius + 1, dtype=float)
        w = np.exp(-0.5 * (k / sigma) ** 2)
        m0, m2, m4 = w.sum(), (w * k ** 2).sum(), (w * k ** 4).sum()
        alpha = 1 / (m0 - m2 * m2 / m4)
        return radius, w * (alpha - alpha * m2 / m4 * k ** 2)

    def ready(self, i: int, lookahead: int) -> bool:
        """第 i 帧往后 lookahead 帧是否都已分析完(或已到片尾)。"""
        _, n, eof = self.snapshot()
        return i < n and (eof or i + lookahead <= n - 1)

    def correction(self, i: int, sigma: float):
        """第 i 帧的校正变换 W = H ∘ G^{-1} (H 为平滑后的路径), 返回 (z, t)。

        窗口越界处用「奇反射」补齐: 以端点为中心做点对称, 能保住线性趋势(匀速摇镜), 且端点处校正量为 0。
        """
        p, n, _ = self.snapshot()
        last = n - 1
        radius, w = self.kernel(sigma)
        idx = np.arange(i - radius, i + radius + 1)
        src = idx.copy()
        anchor = np.zeros(len(idx), int)
        neg, pos = idx < 0, idx > last
        src[neg] = np.minimum(-idx[neg], last)
        src[pos] = np.maximum(2 * last - idx[pos], 0)
        anchor[pos] = last
        vals = p[src]
        flip = neg | pos
        vals[flip] = 2 * p[anchor[flip]] - vals[flip]
        h = w @ vals
        g = p[i]
        z_h = complex(math.exp(h[3]) * math.cos(h[2]), math.exp(h[3]) * math.sin(h[2]))
        z_g = complex(math.exp(g[3]) * math.cos(g[2]), math.exp(g[3]) * math.sin(g[2]))
        z_w = z_h / z_g
        return z_w, complex(h[0], h[1]) - z_w * complex(g[0], g[1])


def _scale_correction(z: complex, t: complex, a: float):
    """把校正量(平移、转角、对数缩放)整体乘以 a。"""
    mag = math.exp(a * math.log(abs(z)))
    ang = a * math.atan2(z.imag, z.real)
    return complex(mag * math.cos(ang), mag * math.sin(ang)), t * a


def overshoot(z: complex, t: complex, hw: float, hh: float, zoom: float) -> float:
    """校正范围对应的内窗口(画面中心 1/zoom 大小)四角经校正反推回源帧后, 偏离名义位置的最大比例(相对校正范围, >1 即超出范围)。"""
    mx, my = hw * (1 - 1 / zoom), hh * (1 - 1 / zoom)
    worst = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = complex(sx * hw / zoom, sy * hh / zoom)
            d = (c - t) / z - c
            worst = max(worst, abs(d.real) / mx, abs(d.imag) / my)
    return worst


def forward_matrix(a: complex, b: complex, width: int, height: int, view) -> np.ndarray:
    """3×3 矩阵: 输出 uv([0,1]², 左上原点) → 某一帧的纹理 uv。

    输出位置 p = (uv - 0.5)·view (以画面中心为原点, 按源像素计, view=(vw, vh) 为输出覆盖的源像素范围),
    该帧中心系坐标 q = a·p + b (复数相似变换)。
    """
    vw, vh = view
    rot = np.array([[a.real, -a.imag], [a.imag, a.real]])
    d = np.array([width, height], float)
    lin = rot * np.array([vw, vh])[None, :] / d[:, None]                 # D⁻¹ R V
    off = np.array([b.real, b.imag]) / d + 0.5 - lin @ [0.5, 0.5]
    m = np.eye(3)
    m[:2, :2], m[:2, 2] = lin, off
    return m


def view_matrix(z: complex, t: complex, width: int, height: int, zoom: float = 1.0, view=None) -> np.ndarray:
    """当前帧的 输出 uv → 源纹理 uv。校正 W(z, t) 把源(中心系)映射到稳定帧, 反过来 p_src = (p - t)/z。
    view 默认 W/zoom × H/zoom(按 zoom 裁切放大); 全幅输出时 view=(W, H)。"""
    return forward_matrix(1 / z, -t / z, width, height, view if view is not None else (width / zoom, height / zoom))


def _lerp_correction(a, b, beta: float):
    """校正量在参数空间(平移、转角、对数缩放)里从 a 线性插值到 b。"""
    (za, ta), (zb, tb) = a, b
    ang = math.atan2(za.imag, za.real) * (1 - beta) + math.atan2(zb.imag, zb.real) * beta
    mag = math.exp(math.log(abs(za)) * (1 - beta) + math.log(abs(zb)) * beta)
    return complex(mag * math.cos(ang), mag * math.sin(ang)), ta * (1 - beta) + tb * beta


class Stabilizer:
    """把一条相机路径变成逐帧校正量 (z, t)。

    强平滑(σ)与弱平滑(σ_weak)各给出一条校正量, 最终校正量在两者之间按增益 β 插值, β 由校正范围决定:
    强平滑放得下就用强平滑; 放不下就向弱平滑靠拢(它仍是平滑路径); 弱平滑也放不下才向 0 收缩(增益 γ)。
    直接向 0 收缩会把原始抖动按比例混回画面, 强平滑 + 小校正范围时高频抖动几乎压不掉。

    增益必须在时间上平滑: 逐帧求出的增益会随原始抖动起伏, 乘上大幅的慢速校正量后凭空产生新的高频抖动。
    做法是先取 ±m 帧内的最小值(保证不比逐帧值更大, 余量约束仍然成立), 再做 σ=m/3 的高斯平滑。
    代价是额外 2m 帧前瞻(只涉及轨迹, 不增加帧缓冲)。
    """

    CAP = 0.95              # 余量利用率上限(相对校正范围)
    WEAK_SEC = 0.15         # 弱平滑 σ
    GAIN_SEC = 0.33         # 增益平滑窗口 m
    MAX_LOOKAHEAD = 100     # 默认前瞻帧数上限; 帧缓冲随它线性增长(4K 每帧 12.4MB), 超出时压低 σ。播放器按内存预算传入

    def __init__(self, traj: Trajectory, width: int, height: int, fps: float, max_lookahead: int | None = None):
        self.traj, self.width, self.height, self.fps = traj, width, height, fps
        self.max_lookahead = max_lookahead or self.MAX_LOOKAHEAD
        self.m = max(int(round(self.GAIN_SEC * fps)), 2)
        k = np.exp(-0.5 * (np.arange(-self.m, self.m + 1) / (self.m / 3)) ** 2)
        self._gauss = k / k.sum()
        self.limit = 0.0    # 最近一帧被限制的程度: 0=强平滑完全生效, 1=退到零校正
        self.configure(0.2, 1.05)

    def configure(self, sigma_sec: float, zoom: float):
        self.sigma = min(max(sigma_sec * self.fps, 1.0), (self.max_lookahead - 2 * self.m) / 3)
        self.weak = min(self.WEAK_SEC * self.fps, self.sigma)
        self.zoom = zoom
        self.radius = Trajectory.kernel(self.sigma)[0]
        self._raw: dict[int, tuple] = {}

    def neighbor_maps(self, i: int, z_w: complex, t_w: complex, offsets):
        """全幅输出(不裁切)时, 用相邻帧补画面边缘的缺口: 返回 [(偏移 d, a, b)], 满足 该帧坐标 q = a·p + b。

        输出像素 p 对应世界点 X = H_i^{-1}(p), 其中 H_i = W_i ∘ G_i 是平滑后的虚拟相机; 它在第 j 帧里的位置是 G_j(X)。
        """
        zg, tg = self.traj.pose(i)
        zh, th = z_w * zg, z_w * tg + t_w
        n = self.traj.snapshot()[1]
        out = []
        for d in offsets:
            j = i + d
            if 0 <= j < n:
                zj, tj = self.traj.pose(j)
                a = zj / zh
                out.append((d, a, tj - a * th))
        return out

    def lookahead(self) -> int:
        """显示第 i 帧需要轨迹已分析到 i + lookahead()。"""
        return self.radius + 2 * self.m

    def _raw_gain(self, j: int):
        hit = self._raw.get(j)
        if hit is not None:
            return hit
        hw, hh = self.width / 2, self.height / 2
        strong, weak = self.traj.correction(j, self.sigma), self.traj.correction(j, self.weak)
        rs, rw = overshoot(*strong, hw, hh, self.zoom), overshoot(*weak, hw, hh, self.zoom)
        beta = 1.0 if rs <= self.CAP else 0.0 if rw >= self.CAP else (self.CAP - rw) / (rs - rw)
        gamma = 1.0 if rw <= self.CAP else self.CAP / rw
        val = (strong, weak, beta, gamma)
        if self.traj.ready(j, self.radius):        # 窗口不完整时的结果以后会变, 不缓存
            self._raw[j] = val
        return val

    def _smooth(self, values: np.ndarray) -> float:
        m = self.m
        floor = np.lib.stride_tricks.sliding_window_view(values, 2 * m + 1).min(axis=1)
        return float(floor @ self._gauss)

    def correction(self, i: int):
        last = self.traj.snapshot()[1] - 1
        raw = [self._raw_gain(min(max(j, 0), last)) for j in range(i - 2 * self.m, i + 2 * self.m + 1)]
        beta = self._smooth(np.array([r[2] for r in raw]))
        gamma = self._smooth(np.array([r[3] for r in raw]))
        strong, weak = raw[2 * self.m][:2]
        z, t = _scale_correction(*_lerp_correction(weak, strong, beta), gamma)
        r = overshoot(z, t, self.width / 2, self.height / 2, self.zoom)
        if r > 1:                                   # 旋转/缩放非线性, 兜底
            z, t = _scale_correction(z, t, 1 / r)
        self.limit = 1 - beta * gamma
        return z, t
