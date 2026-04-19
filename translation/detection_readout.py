"""Object-detection readout for simulator activity maps.

The readout layer consumes dense activity frames ``[T, H, W]`` from
post-processing and emits object-level bounding boxes. It is deliberately
independent from SpikeNet HDF5 parsing so it can also be used with a Python
simulator that returns tensors directly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops_table
from skimage.morphology import binary_closing, disk, remove_small_objects


@dataclass
class BBoxDetection:
    window_index: int
    t_start: int
    t_end: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    area: int
    label: int = 1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def integrate_activity_windows(frames: np.ndarray, *, window: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Sum activity over sliding time windows.

    Args:
        frames: Dense simulator activity with shape ``[T, H, W]``.
        window: Number of frames per integration window.
        stride: Window stride in frames.
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"frames must be [T,H,W], got {arr.shape}")
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")

    windows: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    for start in range(0, max(arr.shape[0] - window + 1, 1), stride):
        end = min(start + window, arr.shape[0])
        if end <= start:
            continue
        windows.append(arr[start:end].sum(axis=0))
        ranges.append((start, end))
        if end == arr.shape[0]:
            break

    if not windows:
        return np.zeros((0,) + arr.shape[1:], dtype=np.float32), []
    return np.stack(windows, axis=0), ranges


def activity_threshold(activity: np.ndarray, *, threshold: float | None, quantile: float | None) -> float:
    """Choose an activity threshold from an absolute value or non-zero quantile."""
    if threshold is not None:
        return float(threshold)
    positive = activity[activity > 0]
    if positive.size == 0:
        return float("inf")
    q = 0.90 if quantile is None else float(quantile)
    if not 0 <= q <= 1:
        raise ValueError("threshold quantile must be in [0, 1]")
    return float(np.quantile(positive, q))


def bbox_iou(a: BBoxDetection, b: BBoxDetection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms_detections(detections: list[BBoxDetection], *, iou_threshold: float) -> list[BBoxDetection]:
    """Apply per-window non-maximum suppression."""
    if iou_threshold <= 0:
        return detections
    kept: list[BBoxDetection] = []
    for window in sorted({d.window_index for d in detections}):
        current = [d for d in detections if d.window_index == window]
        current.sort(key=lambda d: d.score, reverse=True)
        while current:
            best = current.pop(0)
            kept.append(best)
            current = [d for d in current if bbox_iou(best, d) <= iou_threshold]
    kept.sort(key=lambda d: (d.t_start, d.y1, d.x1))
    return kept


def detect_bboxes_from_windows(
    activity_windows: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    threshold: float | None = 1.0,
    threshold_quantile: float | None = None,
    min_area: int = 4,
    closing_radius: int = 0,
    nms_iou: float = 0.0,
) -> list[BBoxDetection]:
    """Extract connected-component boxes from integrated activity windows."""
    detections: list[BBoxDetection] = []
    for idx, activity in enumerate(np.asarray(activity_windows, dtype=np.float32)):
        thr = activity_threshold(activity, threshold=threshold, quantile=threshold_quantile)
        mask = activity >= thr
        if closing_radius > 0:
            mask = binary_closing(mask, footprint=disk(closing_radius))
        if min_area > 1:
            mask = remove_small_objects(mask.astype(bool), min_size=min_area)

        labels = label(mask, connectivity=2, background=0)
        props = regionprops_table(labels, intensity_image=activity, properties=("bbox", "area", "mean_intensity", "max_intensity"))
        count = len(props.get("bbox-0", []))
        t_start, t_end = ranges[idx]
        for item in range(count):
            area = int(props["area"][item])
            if area < min_area:
                continue
            y1 = int(props["bbox-0"][item])
            x1 = int(props["bbox-1"][item])
            y2 = int(props["bbox-2"][item])
            x2 = int(props["bbox-3"][item])
            mean_i = float(props["mean_intensity"][item])
            max_i = float(props["max_intensity"][item])
            score = mean_i * area + max_i
            detections.append(
                BBoxDetection(
                    window_index=idx,
                    t_start=int(t_start),
                    t_end=int(t_end),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    area=area,
                )
            )
    return nms_detections(detections, iou_threshold=nms_iou)


def readout_bboxes(
    frames: np.ndarray,
    *,
    window: int = 5,
    stride: int = 5,
    threshold: float | None = 1.0,
    threshold_quantile: float | None = None,
    min_area: int = 4,
    closing_radius: int = 0,
    nms_iou: float = 0.0,
) -> tuple[np.ndarray, list[BBoxDetection]]:
    """Run the full detection readout from ``[T,H,W]`` activity to bboxes."""
    activity_windows, ranges = integrate_activity_windows(frames, window=window, stride=stride)
    detections = detect_bboxes_from_windows(
        activity_windows,
        ranges,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        min_area=min_area,
        closing_radius=closing_radius,
        nms_iou=nms_iou,
    )
    return activity_windows, detections


def scale_detections(
    detections: list[BBoxDetection],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[BBoxDetection]:
    """Map boxes from simulator coordinates back to source-image coordinates."""
    scaled: list[BBoxDetection] = []
    for det in detections:
        scaled.append(
            BBoxDetection(
                window_index=det.window_index,
                t_start=det.t_start,
                t_end=det.t_end,
                x1=int(round(det.x1 * scale_x + offset_x)),
                y1=int(round(det.y1 * scale_y + offset_y)),
                x2=int(round(det.x2 * scale_x + offset_x)),
                y2=int(round(det.y2 * scale_y + offset_y)),
                score=det.score,
                area=det.area,
                label=det.label,
            )
        )
    return scaled


def save_detections_json(detections: list[BBoxDetection], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(d) for d in detections], indent=2), encoding="utf-8")
    return output_path


def save_mot_txt(detections: list[BBoxDetection], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, det in enumerate(detections, start=1):
            frame_id = det.t_start + 1
            f.write(
                f"{frame_id},{idx},{det.x1},{det.y1},{det.width},{det.height},"
                f"{det.score:.6f},-1,-1,-1\n"
            )
    return output_path


def evaluate_iou_matches(
    detections: list[BBoxDetection],
    ground_truth: list[BBoxDetection],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Greedy per-window detection evaluation for debugging."""
    tp = 0
    fp = 0
    fn = 0
    matched_ious: list[float] = []
    windows = sorted({d.window_index for d in detections} | {g.window_index for g in ground_truth})
    for window in windows:
        dets = [d for d in detections if d.window_index == window]
        gts = [g for g in ground_truth if g.window_index == window]
        used_gt: set[int] = set()
        for det in sorted(dets, key=lambda d: d.score, reverse=True):
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts):
                if idx in used_gt:
                    continue
                iou = bbox_iou(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                used_gt.add(best_idx)
                matched_ious.append(best_iou)
            else:
                fp += 1
        fn += len(gts) - len(used_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def _load_frames(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        raise ValueError("Only .npy activity frame input is supported by this CLI")
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got {arr.shape}")
    return arr


def _main() -> None:
    parser = argparse.ArgumentParser(description="Read object-detection bboxes from activity frames.")
    parser.add_argument("frames_npy")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-quantile", type=float, default=0.90)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--closing-radius", type=int, default=0)
    parser.add_argument("--nms-iou", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-mot")
    args = parser.parse_args()

    frames = _load_frames(Path(args.frames_npy))
    activity, detections = readout_bboxes(
        frames,
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        threshold_quantile=args.threshold_quantile,
        min_area=args.min_area,
        closing_radius=args.closing_radius,
        nms_iou=args.nms_iou,
    )
    save_detections_json(detections, args.output_json)
    if args.output_mot:
        save_mot_txt(detections, args.output_mot)
    print(json.dumps({"activity_shape": list(activity.shape), "num_detections": len(detections)}, indent=2))


if __name__ == "__main__":
    _main()

