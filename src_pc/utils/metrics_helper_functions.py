"""
metrics_helper_functions.py

System monitoring + trajectory evaluation (ATE, RPE) for the PC project.
Depends only on stdlib, numpy, psutil, and pynvml (optional, for GPU stats).

What's in here:
  SystemMonitor       background thread sampling CPU/RAM/GPU every N sec
  summarize_system    squash a list of samples -> avg/max dict
  load_tum_trajectory read a TUM file -> Trajectory
  compute_ate         Absolute Trajectory Error (Umeyama alignment + scale)
  compute_rpe         Relative Pose Error (translational + rotational)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

# GPU monitoring (optional) -----------------------------------------------------------------

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False


# Data structures --------------------------------------------------------------------------

@dataclass
class SystemSample:
    t:               float
    cpu_percent:     float
    ram_percent:     float
    ram_used_gb:     float = 0.0
    gpu_percent:     Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_power_w:     Optional[float] = None


@dataclass
class Trajectory:
    t:    np.ndarray   # (N,)   timestamps in seconds
    xyz:  np.ndarray   # (N,3)  positions
    quat: np.ndarray   # (N,4)  [qx, qy, qz, qw]


@dataclass
class ATEResult:
    rmse:       float
    mean:       float
    median:     float
    max:        float
    num_frames: int
    scale:      float                        # Umeyama scale factor (1.0 = no drift)
    errors:     Optional[np.ndarray] = None  # per-frame errors (N,)
    idx_ref:    Optional[np.ndarray] = None  # matched indices into GT
    idx_query:  Optional[np.ndarray] = None  # matched indices into prediction


@dataclass
class RPEResult:
    trans_rmse:    float
    trans_mean:    float
    trans_median:  float
    rot_rmse_deg:  float
    rot_mean_deg:  float
    rot_median_deg: float
    num_pairs:     int
    delta_t:       float
    trans_errors:  Optional[np.ndarray] = None  # (N,)
    rot_errors_deg: Optional[np.ndarray] = None  # (N,)
    t0:            Optional[np.ndarray] = None   # (N,) start timestamps
    t1:            Optional[np.ndarray] = None   # (N,) end timestamps


# System monitor ---------------------------------------------------------------------------

class SystemMonitor:
    """Background thread polling CPU/RAM/GPU every `interval` seconds.
    Usage: start() ... stop() -> summarize() / .samples."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        self.interval  = interval
        self.gpu_index = gpu_index
        self.samples: List[SystemSample] = []
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=self.interval * 3)

    def _sample(self) -> SystemSample:
        t   = time.time()
        cpu = psutil.cpu_percent(interval=None)
        vm  = psutil.virtual_memory()
        ram = vm.percent
        ram_gb = vm.used / 1024 / 1024 / 1024

        gpu_pct = gpu_mem_used = gpu_mem_total = gpu_power = None
        if _HAS_GPU:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpu_pct       = float(u.gpu)
                gpu_mem_used  = m.used  / 1024 / 1024
                gpu_mem_total = m.total / 1024 / 1024
                try:
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    pass
            except Exception:
                pass

        return SystemSample(
            t=t, cpu_percent=cpu, ram_percent=ram, ram_used_gb=ram_gb,
            gpu_percent=gpu_pct, gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total, gpu_power_w=gpu_power,
        )

    def sample(self) -> SystemSample:
        """Grab one sample now (per-frame path, no thread). cpu_percent() is
        non-blocking - call it once beforehand to prime it, or the first
        reading comes back as 0."""
        s = self._sample()
        self.samples.append(s)
        return s

    def _run(self):
        while not self._stop_event.is_set():
            self.samples.append(self._sample())
            self._stop_event.wait(self.interval)

    def summarize(self) -> Dict[str, float]:
        """avg/max dict."""
        return summarize_system_metrics(self.samples)


def summarize_system_metrics(samples: List[SystemSample]) -> Dict[str, float]:
    """Squash a list of SystemSamples down to avg/max per metric."""
    if not samples:
        return {}

    cpu = np.array([s.cpu_percent for s in samples], dtype=float)
    ram = np.array([s.ram_percent for s in samples], dtype=float)
    gpu = np.array(
        [s.gpu_percent for s in samples if s.gpu_percent is not None], dtype=float
    )

    out: Dict[str, float] = {
        "cpu_avg": float(cpu.mean()), "cpu_max": float(cpu.max()),
        "ram_avg": float(ram.mean()), "ram_max": float(ram.max()),
    }
    if gpu.size > 0:
        out["gpu_avg"] = float(gpu.mean())
        out["gpu_max"] = float(gpu.max())
    return out


# Trajectory I/O ---------------------------------------------------------------------------

def load_tum_trajectory(path: Path) -> Trajectory:
    """Load TUM-format trajectory: ts x y z qx qy qz qw"""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append(list(map(float, parts[:8])))
    if not rows:
        raise RuntimeError(f"No valid TUM poses in {path}")
    arr = np.array(rows, dtype=float)
    return Trajectory(t=arr[:, 0], xyz=arr[:, 1:4], quat=arr[:, 4:8])


# Timestamp association --------------------------------------------------------------------

def _associate(
    t_ref: np.ndarray,
    t_query: np.ndarray,
    max_dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match each ref pose to the nearest-in-time query pose. Returns (idx_ref, idx_query)."""
    idx_ref, idx_q = [], []
    j = 0
    for i in range(len(t_ref)):
        while j + 1 < len(t_query) and t_query[j + 1] <= t_ref[i]:
            j += 1
        best_k  = j
        best_dt = abs(t_query[j] - t_ref[i])
        if j + 1 < len(t_query):
            dt1 = abs(t_query[j + 1] - t_ref[i])
            if dt1 < best_dt:
                best_k, best_dt = j + 1, dt1
        if best_dt <= max_dt:
            idx_ref.append(i)
            idx_q.append(best_k)
    return np.array(idx_ref, dtype=int), np.array(idx_q, dtype=int)


# ATE --------------------------------------------------------------------------------------

def _umeyama(
    xyz_ref: np.ndarray,
    xyz_q:   np.ndarray,
    with_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment - finds s, R, t so that xyz_ref ≈ s * R * xyz_q + t."""
    n      = xyz_ref.shape[0]
    mu_r   = xyz_ref.mean(0);  mu_q = xyz_q.mean(0)
    Y      = xyz_ref - mu_r;   X    = xyz_q  - mu_q
    Sigma  = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float((D @ S.diagonal()) / ((X ** 2).sum() / n)) if with_scale else 1.0
    t = mu_r - s * R @ mu_q
    return R, t, s


def compute_ate(
    traj_ref:   Trajectory,
    traj_query: Trajectory,
    max_dt:     float = 0.05,
    with_scale: bool  = True,
) -> ATEResult:
    """
    Absolute Trajectory Error, after aligning the two trajectories with Umeyama.
    with_scale=True also solves for a scale factor (matters a lot for monocular).
    Stereo should come out at scale ≈ 1.0 - if it's way off, that's scale drift.
    """
    ir, iq = _associate(traj_ref.t, traj_query.t, max_dt)
    if len(ir) == 0:
        raise RuntimeError("ATE: no matching timestamps between GT and prediction.")

    xyz_r = traj_ref.xyz[ir]
    xyz_q = traj_query.xyz[iq]

    R, t, s = _umeyama(xyz_r, xyz_q, with_scale)
    aligned  = (s * (R @ xyz_q.T)).T + t
    errors   = np.linalg.norm(xyz_r - aligned, axis=1)

    return ATEResult(
        rmse=float(np.sqrt((errors ** 2).mean())),
        mean=float(errors.mean()),
        median=float(np.median(errors)),
        max=float(errors.max()),
        num_frames=int(len(errors)),
        scale=float(s),
        errors=errors,
        idx_ref=ir,
        idx_query=iq,
    )


def compute_ate_from_files(
    gt_path:    Path,
    pred_path:  Path,
    max_dt:     float = 0.05,
    with_scale: bool  = True,
) -> ATEResult:
    """Convenience wrapper - load the two TUM files, then compute ATE."""
    return compute_ate(
        load_tum_trajectory(gt_path),
        load_tum_trajectory(pred_path),
        max_dt=max_dt,
        with_scale=with_scale,
    )


# RPE --------------------------------------------------------------------------------------

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1-2*y*y-2*z*z,  2*x*y-2*z*w,    2*x*z+2*y*w  ],
        [2*x*y+2*z*w,    1-2*x*x-2*z*z,  2*y*z-2*x*w  ],
        [2*x*z-2*y*w,    2*y*z+2*x*w,    1-2*x*x-2*y*y],
    ], dtype=float)


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _relative_pose(
    xyz: np.ndarray, quat: np.ndarray, i: int, j: int
) -> Tuple[np.ndarray, np.ndarray]:
    Ri = _quat_to_R(quat[i]);  Rj = _quat_to_R(quat[j])
    return Ri.T @ Rj, Ri.T @ (xyz[j] - xyz[i])


def compute_rpe(
    traj_ref:   Trajectory,
    traj_query: Trajectory,
    delta_t:    float = 1.0,
    max_dt:     float = 0.05,
) -> RPEResult:
    """
    Relative Pose Error over a fixed delta_t window (captures local drift,
    unlike ATE's single global alignment).
    trans error = ||t_ref_rel - t_query_rel||; rot error = angle(R_ref_rel @ R_query_rel.T), degrees.
    """
    ir, iq = _associate(traj_ref.t, traj_query.t, max_dt)
    if len(ir) < 2:
        raise RuntimeError("RPE: not enough matched timestamps.")

    t_assoc = traj_ref.t[ir]
    trans_errs, rot_errs, t0s, t1s = [], [], [], []

    for k in range(len(t_assoc)):
        t_target = t_assoc[k] + delta_t
        j = int(np.searchsorted(t_assoc, t_target))
        best, best_dt = None, None
        for c in ([j, j-1] if j > 0 else [j]):
            if 0 <= c < len(t_assoc):
                dt = abs(t_assoc[c] - t_target)
                if best_dt is None or dt < best_dt:
                    best, best_dt = c, dt

        if best is None or best == k:
            continue
        if best_dt > max(delta_t * 0.5, max_dt):
            continue

        R_r, t_r = _relative_pose(traj_ref.xyz,   traj_ref.quat,   ir[k], ir[best])
        R_q, t_q = _relative_pose(traj_query.xyz, traj_query.quat, iq[k], iq[best])

        trans_errs.append(float(np.linalg.norm(t_r - t_q)))
        rot_errs.append(_angle_deg(R_r @ R_q.T))
        t0s.append(traj_ref.t[ir[k]])
        t1s.append(traj_ref.t[ir[best]])

    if not trans_errs:
        raise RuntimeError("RPE: no valid pairs found.")

    ta = np.array(trans_errs); ra = np.array(rot_errs)
    return RPEResult(
        trans_rmse=float(np.sqrt((ta**2).mean())),
        trans_mean=float(ta.mean()),
        trans_median=float(np.median(ta)),
        rot_rmse_deg=float(np.sqrt((ra**2).mean())),
        rot_mean_deg=float(ra.mean()),
        rot_median_deg=float(np.median(ra)),
        num_pairs=int(len(ta)),
        delta_t=float(delta_t),
        trans_errors=ta,
        rot_errors_deg=ra,
        t0=np.array(t0s),
        t1=np.array(t1s),
    )