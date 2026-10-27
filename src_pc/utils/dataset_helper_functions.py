"""
src_pc/utils/dataset_helper_functions.py

different helper functions for processing datasets (KITTI-360, TUM, etc.)
"""

import numpy as np
from datetime import datetime
from pathlib import Path


# Shared math helpers ----------------------------------------------------------------------

def load_frame_indices(image_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in image_dir.glob("*.png"))


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [qw, qx, qy, qz]."""
    q     = np.empty(4)
    trace = np.trace(R)
    if trace > 0:
        s    = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i    = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s    = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[i + 1] = 0.25 * s
        q[0]     = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


# KITTI-360 processing ---------------------------------------------------------------------

def load_timestamps_list(ts_file: Path) -> list[float]:
    ts_list = []; first_dt = None
    if not ts_file.exists():
        return ts_list
    with open(ts_file) as f:
        for line in f:
            clean = line.strip()[:26]
            if not clean:
                continue
            try:
                dt = datetime.strptime(clean, "%Y-%m-%d %H:%M:%S.%f")
                if first_dt is None:
                    first_dt = dt
                ts_list.append((dt - first_dt).total_seconds())
            except ValueError:
                continue
    return ts_list


def _read_cam_to_pose(calib_file: Path, cam: str = "image_00") -> np.ndarray:
    """KITTI-360 calib_cam_to_pose.txt -> 4x4 T_cam_to_vehicle."""
    T = np.eye(4)
    with open(calib_file) as f:
        for line in f:
            if line.startswith(cam + ":"):
                vals = np.array(line.split(":")[1].split(), dtype=float)
                T[:3, :4] = vals.reshape(3, 4)
                return T
    raise RuntimeError(f"{cam} not found in {calib_file}")


def _generate_tum_trajectory(pose_lines, timestamps, T_cam_to_pose=None):
    traj = []
    for i, line in enumerate(pose_lines):
        if i >= len(timestamps):
            break
        vals = list(map(float, line.split()[1:]))
        T = np.eye(4)
        T[:3, :4] = np.array(vals).reshape(3, 4)   # T_vehicle_to_world
        if T_cam_to_pose is not None:
            T = T @ T_cam_to_pose                  # -> T_camera_to_world
        R, t = T[:3, :3], T[:3, 3]
        q = _rotation_to_quaternion(R)
        traj.append(f"{timestamps[i]:.6f} {t[0]} {t[1]} {t[2]} "
                    f"{q[1]} {q[2]} {q[3]} {q[0]}")
    return traj


def process_sequence(
    test_dir:         Path,
    drive_dir:        Path,
    processed_images: Path,
    processed_poses:  Path,
    raw_poses_root:   Path,
    target_width:     int | None = None,
    grayscale:        bool = False,
) -> dict | None:
    """
    Turn one raw KITTI-360 sequence into the processed layout pySLAM expects.
    target_width : resize frames to this width (keeps the aspect ratio)
    grayscale    : if True, convert to gray and save to image_0/image_1 (is_color=False)
    """
    import cv2

    test_id = test_dir.name.replace("_", "")
    drive   = drive_dir.name
    print(f"\n>>> [KITTI-360] Processing: {test_id}  ({drive})")

    raw_img_dir    = drive_dir / "image_00" / "data_rect"
    all_raw_frames = load_frame_indices(raw_img_dir)
    if not all_raw_frames:
        print(f"  [!] Skipping — no images in {raw_img_dir}")
        return None

    seq_root  = processed_images / test_id / drive
    left_dir  = "image_0" if grayscale else "image_2"
    right_dir = "image_1" if grayscale else "image_3"
    copied_frames = []

    resize_to = None
    if target_width is not None:
        sample = next(raw_img_dir.glob("*.png"), None)
        if sample:
            img0 = cv2.imread(str(sample))
            if img0 is not None:
                oh, ow = img0.shape[:2]
                nh = round(oh * target_width / ow)
                resize_to = (target_width, nh)
                print(f"  - Resizing {ow}×{oh} → {target_width}×{nh}")

    for raw_cam, dst_cam in (("image_00", left_dir), ("image_01", right_dir)):
        src = drive_dir / raw_cam / "data_rect"
        dst = seq_root / dst_cam
        if not src.exists():
            print(f"  [!] Warning: {raw_cam} missing.")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        print(f"  - Copying {raw_cam} → {dst_cam} ...")
        src_imgs = sorted(img for img in src.glob("*.png") if "Identifier" not in img.name)
        for seq_idx, img in enumerate(src_imgs):
            frame = cv2.imread(str(img))
            if resize_to:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(dst / f"{seq_idx:06d}.png"), frame)
            if raw_cam == "image_00":
                copied_frames.append(int(img.stem))

    copied_frames.sort()
    if not copied_frames:
        print("  [!] No frames copied — skipping.")
        return None

    raw_ts_file    = drive_dir / "image_00" / "timestamps.txt"
    all_timestamps = load_timestamps_list(raw_ts_file)

    if not all_timestamps:
        print("  [!] timestamps.txt not found — using 10 Hz fallback.")
        seq_timestamps = [i * 0.1 for i in range(len(copied_frames))]
    else:
        frame_positions = {f: i for i, f in enumerate(all_raw_frames)}
        seq_timestamps  = [
            all_timestamps[frame_positions[f]]
            for f in copied_frames
            if f in frame_positions and frame_positions[f] < len(all_timestamps)
        ]

    with open(seq_root / "times.txt", "w") as f:
        f.write("\n".join(f"{ts:.6f}" for ts in seq_timestamps))
    print(f"  - times.txt  ({len(seq_timestamps)} entries)")

    pose_src   = raw_poses_root / drive / "poses.txt"
    pose_lines = []
    if pose_src.exists():
        frame_set = set(copied_frames)
        with open(pose_src) as f:
            for line in f:
                parts = line.split()
                if parts and int(parts[0]) in frame_set:
                    pose_lines.append(line.strip())
        pose_lines.sort(key=lambda ln: int(ln.split()[0]))
    else:
        print(f"  [!] poses not found at {pose_src}")

    print(f"  - Frames: {len(copied_frames)}  Poses: {len(pose_lines)}")

    poses_dir  = processed_poses / test_id
    poses_dir.mkdir(parents=True, exist_ok=True)
    kitti_file = poses_dir / f"{drive}.txt"

    with open(kitti_file, "w") as f:
        for line in pose_lines:
            f.write(" ".join(line.split()[1:]) + "\n")

    tum_file = poses_dir / f"{drive}_tum.txt"
    with open(tum_file, "w") as f:
        f.write("\n".join(_generate_tum_trajectory(pose_lines, seq_timestamps)))

    print(f"  - KITTI poses → poses/{test_id}/{drive}.txt")
    print(f"  - TUM poses   → poses/{test_id}/{drive}_tum.txt")

    return {
        "test":        test_id,
        "drive":       drive,
        "frames":      len(copied_frames),
        "min":         min(copied_frames),
        "max":         max(copied_frames),
        "kitti_poses": kitti_file,
        "gt_tum":      tum_file,
        "seq_root":    seq_root,
        "source":      "kitti360",
        "is_color":    not grayscale,
        "resize_to":   resize_to,
    }