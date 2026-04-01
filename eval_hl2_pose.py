#!/usr/bin/env python3
from pathlib import Path
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT_FILE = REPO_ROOT / "data/gt/poses_gt_comparable_from_eval.txt"
DEFAULT_VO_FILE = REPO_ROOT / "data/result/poses/poses_KITTI.txt"


def load_kitti_poses(path: Path):
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = np.fromstring(line, sep=" ", dtype=np.float32)
        if values.size != 12:
            raise ValueError(f"{path} has a line with {values.size} values, expected 12")
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :4] = values.reshape(3, 4)
        poses.append(transform)
    return poses


def to_tcw_poses(poses, pose_format: str):
    pose_format = pose_format.lower()
    if pose_format not in {"tcw", "twc"}:
        raise ValueError(f"Unsupported pose format: {pose_format}")
    if pose_format == "tcw":
        return poses
    return [np.linalg.inv(pose) for pose in poses]


def trajectory_stats(poses):
    if len(poses) < 2:
        return 0.0, 0.0
    path_length = 0.0
    for i in range(1, len(poses)):
        path_length += np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3])
    end_displacement = np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])
    return path_length, end_displacement


def compute_pose_errors(gt_poses, vo_poses):
    frame_count = min(len(gt_poses), len(vo_poses))
    if frame_count == 0:
        raise ValueError("No pose to compare")

    total_rot_error_rad = 0.0
    total_trans_error_m = 0.0
    per_frame_errors = []

    # Preserve the original error logic.
    for frame_idx in range(frame_count):
        pose_gt = gt_poses[frame_idx]
        pose_vo = vo_poses[frame_idx]
        pose_error = np.linalg.inv(pose_vo) @ pose_gt

        rot_error_rad = np.arccos(np.clip((np.trace(pose_error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        trans_error_m = np.linalg.norm(pose_error[:3, 3])
        total_rot_error_rad += rot_error_rad
        total_trans_error_m += trans_error_m
        per_frame_errors.append((frame_idx, np.rad2deg(rot_error_rad), trans_error_m))

    avg_rot_error_deg = np.rad2deg(total_rot_error_rad) / frame_count
    avg_trans_error_m = total_trans_error_m / frame_count
    return frame_count, avg_rot_error_deg, avg_trans_error_m, per_frame_errors


def evaluate_once(gt_poses, vo_poses):
    frame_count, avg_rot_error_deg, avg_trans_error_m, per_frame_errors = compute_pose_errors(gt_poses, vo_poses)
    gt_len, gt_end = trajectory_stats(gt_poses[:frame_count])
    vo_len, vo_end = trajectory_stats(vo_poses[:frame_count])
    return {
        "frame_count": frame_count,
        "avg_rot_error_deg": avg_rot_error_deg,
        "avg_trans_error_m": avg_trans_error_m,
        "per_frame_errors": per_frame_errors,
        "gt_len": gt_len,
        "gt_end": gt_end,
        "vo_len": vo_len,
        "vo_end": vo_end,
    }


def save_per_frame_errors(output_dir: Path, per_frame_errors, avg_rot_error_deg: float, avg_trans_error_m: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_frame_pose_error.txt"
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("frame,rot_error_deg,trans_error_m\n")
        for frame_idx, rot_err_deg, trans_err_m in per_frame_errors:
            file_obj.write(f"{frame_idx},{rot_err_deg:.8f},{trans_err_m:.8f}\n")
        file_obj.write(f"avg,{avg_rot_error_deg:.8f},{avg_trans_error_m:.8f}\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate camera pose error with KITTI-format trajectories.")
    parser.add_argument("--gt-file", type=Path, default=DEFAULT_GT_FILE)
    parser.add_argument("--vo-file", type=Path, default=DEFAULT_VO_FILE)
    parser.add_argument("--gt-format", choices=["Tcw", "Twc"], default="Tcw", help="Pose convention used in GT file.")
    parser.add_argument("--vo-format", choices=["Tcw", "Twc"], default="Tcw", help="Pose convention used in VO file.")
    parser.add_argument(
        "--auto-vo-inverse",
        action="store_true",
        help="Try both VO and inverse(VO), then keep the one with lower average translation error.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Use first N frames only. 0 means use all frames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save per-frame error file. Default: same folder as --vo-file.",
    )
    args = parser.parse_args()

    if not args.gt_file.is_file():
        raise FileNotFoundError(f"GT file not found: {args.gt_file}")
    if not args.vo_file.is_file():
        raise FileNotFoundError(f"VO file not found: {args.vo_file}")

    gt_poses_raw = load_kitti_poses(args.gt_file)
    vo_poses_raw = load_kitti_poses(args.vo_file)
    gt_poses = to_tcw_poses(gt_poses_raw, args.gt_format)
    vo_poses = to_tcw_poses(vo_poses_raw, args.vo_format)

    if args.max_frames > 0:
        gt_poses = gt_poses[: args.max_frames]
        vo_poses = vo_poses[: args.max_frames]

    if len(gt_poses) != len(vo_poses):
        print(
            f"warning: pose length mismatch, gt={len(gt_poses)}, vo={len(vo_poses)}. "
            "Comparing only overlapping prefix."
        )

    best_eval = evaluate_once(gt_poses, vo_poses)
    used_inverse = False
    if args.auto_vo_inverse:
        inv_eval = evaluate_once(gt_poses, [np.linalg.inv(pose) for pose in vo_poses])
        if inv_eval["avg_trans_error_m"] < best_eval["avg_trans_error_m"]:
            best_eval = inv_eval
            used_inverse = True

    print(f"frames compared: {best_eval['frame_count']}")
    print(f"gt trajectory: path_len={best_eval['gt_len']:.6f} m, end_disp={best_eval['gt_end']:.6f} m")
    print(f"vo trajectory: path_len={best_eval['vo_len']:.6f} m, end_disp={best_eval['vo_end']:.6f} m")
    if best_eval["gt_len"] > 0 and (
        best_eval["vo_len"] / best_eval["gt_len"] > 2.0
        or best_eval["gt_len"] / max(best_eval["vo_len"], 1e-9) > 2.0
    ):
        print("warning: GT/VO trajectory scales differ a lot, check sequence/file matching first.")
    if args.auto_vo_inverse:
        print(f"auto_vo_inverse selected: {'inverse(VO)' if used_inverse else 'VO'}")
    print("average rotation error (degrees)")
    print(best_eval["avg_rot_error_deg"])
    print("average translation error (meters)")
    print(best_eval["avg_trans_error_m"])

    output_dir = args.output_dir if args.output_dir is not None else args.vo_file.parent
    output_path = save_per_frame_errors(
        output_dir,
        best_eval["per_frame_errors"],
        best_eval["avg_rot_error_deg"],
        best_eval["avg_trans_error_m"],
    )
    print(f"Per-frame errors saved to: {output_path}")


if __name__ == "__main__":
    main()
