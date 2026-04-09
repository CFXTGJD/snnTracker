# -*- coding: utf-8 -*-
# @Time : 2022/7/15 14:24
# @Author : Yajing Zheng
# @File : tracking_mot.py
import os, pathlib, csv, re
import pandas as pd
import motmetrics as mm

def _coerce_single_path(x):
    """把 list/tuple/PathLike/str 统一为单个字符串路径。"""
    # PathLike -> str
    if isinstance(x, os.PathLike) or isinstance(x, pathlib.Path):
        return os.fspath(x)
    # str 直接返回
    if isinstance(x, str):
        return x
    # list/tuple：取第一个「存在的」路径；都不存在就取第一个元素转成 str
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("gt_file is an empty list/tuple")
        for cand in x:
            if isinstance(cand, (str, os.PathLike, pathlib.Path)):
                p = os.fspath(cand)
                if os.path.exists(p):
                    return p
        return os.fspath(x[0])
    raise TypeError(f"gt_file must be str/PathLike or list/tuple thereof, got: {type(x)}")

def normalize_gt_to_csv(src_path, dst_path, expected_cols=10):
    # 读二进制并规范化：去 BOM、统一换行、去控制符
    raw = open(src_path, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    txt = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n').decode('utf-8', errors='ignore')
    txt = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', txt)

    rows = []
    for line in txt.split('\n'):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        # 先按逗号 CSV 解析（尊重引号）
        r = next(csv.reader([s], delimiter=',', quotechar='"', skipinitialspace=True))
        # 如果列数异常，回退为空白分隔
        if len(r) > expected_cols + 10 or len(r) == 1:
            r_ws = re.split(r'\s+', s.replace(',', ' '))
            if 3 <= len(r_ws) <= max(expected_cols, 12) + 10:
                r = r_ws
        # 裁剪/填充列数
        if len(r) < expected_cols:
            r = r + ['-1'] * (expected_cols - len(r))
        elif len(r) > expected_cols:
            r = r[:expected_cols]
        rows.append(r)

    with open(dst_path, 'w', encoding='utf-8', newline='') as g:
        csv.writer(g).writerows(rows)

def robust_load_gt(gt_file, fmt="mot15-2D"):
    path = _coerce_single_path(gt_file)  # << 关键：把 list 等转成单一路径
    # 最小修复：mot15-2D 对应常见 10 列 MOT 文本；若补到 12 列会导致列语义错位
    # （FrameId/Id 被当成普通列，X/Y/W/H整体左移），从而 IoU 基本为 0。
    expected_cols = 10
    cleaned = os.path.splitext(path)[0] + f".clean_{expected_cols}.csv"

    normalize_gt_to_csv(path, cleaned, expected_cols=expected_cols)

    # C 引擎 + 明确逗号分隔，避免 regex 分隔导致引号失效
    _ = pd.read_csv(cleaned, header=None, engine='c')  # 触发一次严格解析，若有问题可直接报错定位
    gt = mm.io.loadtxt(cleaned, fmt=fmt, min_confidence=0.5, sep=",", engine="c", skipinitialspace=True)
    return gt



class TrackingMetrics:

    def __init__(self, res_filepath, **dataDict):
        self.gt_file = dataDict.get('labeled_data_dir')
        print(f'gt_file: {self.gt_file}')
        # self.gt = mm.io.loadtxt(self.gt_file, fmt="mot15-2D", min_confidence=0.5)
        self.gt = robust_load_gt(self.gt_file, fmt="mot15-2D")  # 或 "mot15-2D"

        model_res = mm.io.loadtxt(res_filepath, fmt="mot15-2D")

        # 最小修复：统一 MultiIndex 名称，避免 pandas union/join 报错
        if isinstance(self.gt.index, pd.MultiIndex) and len(self.gt.index.names) >= 2:
            self.gt.index = self.gt.index.set_names(["FrameId", "Id"])
        if isinstance(model_res.index, pd.MultiIndex) and len(model_res.index.names) >= 2:
            model_res.index = model_res.index.set_names(["FrameId", "Id"])

        # ---- Diagnostics for metric abnormalities ----
        gt_frame_counts = self.gt.groupby(level="FrameId").size()
        dt_frame_counts = model_res.groupby(level="FrameId").size()
        common_frames = sorted(set(gt_frame_counts.index) & set(dt_frame_counts.index))

        print("[Diag] GT rows:", len(self.gt), "DT rows:", len(model_res))
        print(
            "[Diag] GT det/frame mean,min,max = "
            f"{gt_frame_counts.mean():.2f}, {gt_frame_counts.min()}, {gt_frame_counts.max()}"
        )
        print(
            "[Diag] DT det/frame mean,min,max = "
            f"{dt_frame_counts.mean():.2f}, {dt_frame_counts.min()}, {dt_frame_counts.max()}"
        )
        print("[Diag] overlap frame count:", len(common_frames))

        dt_dup_count = int(model_res.reset_index().duplicated(subset=["FrameId", "Id"]).sum())
        gt_dup_count = int(self.gt.reset_index().duplicated(subset=["FrameId", "Id"]).sum())
        print(f"[Diag] duplicate (FrameId,Id): GT={gt_dup_count}, DT={dt_dup_count}")

        # Minimal guard fix: deduplicate DT by (FrameId, Id) before evaluation.
        # Keep the highest-confidence row for each key.
        if dt_dup_count > 0:
            model_res = (
                model_res
                .reset_index()
                .sort_values(["FrameId", "Id", "Confidence"], ascending=[True, True, False])
                .drop_duplicates(subset=["FrameId", "Id"], keep="first")
                .set_index(["FrameId", "Id"])
                .sort_index()
            )
            print(f"[Diag] DT rows after dedup: {len(model_res)}")

        # quick geometric sanity check: max IoU in first overlap frame
        def _max_iou_for_frame(gt_f: pd.DataFrame, dt_f: pd.DataFrame) -> float:
            if gt_f.empty or dt_f.empty:
                return 0.0
            g = gt_f[["X", "Y", "Width", "Height"]].to_numpy(dtype=float)
            d = dt_f[["X", "Y", "Width", "Height"]].to_numpy(dtype=float)

            best = 0.0
            for gx, gy, gw, gh in g:
                g_x2, g_y2 = gx + gw, gy + gh
                for dx, dy, dw, dh in d:
                    d_x2, d_y2 = dx + dw, dy + dh
                    inter_w = max(0.0, min(g_x2, d_x2) - max(gx, dx))
                    inter_h = max(0.0, min(g_y2, d_y2) - max(gy, dy))
                    inter = inter_w * inter_h
                    union = gw * gh + dw * dh - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best:
                        best = iou
            return best

        if common_frames:
            f0 = common_frames[0]
            gt0 = self.gt.xs(f0, level="FrameId")
            dt0 = model_res.xs(f0, level="FrameId")
            print(f"[Diag] first overlap frame={f0}, GT count={len(gt0)}, DT count={len(dt0)}")
            print(f"[Diag] first overlap frame max IoU={_max_iou_for_frame(gt0, dt0):.4f}")

        # 根据GT和自己的结果，生成accumulator，distth是距离阈值
        self.acc = mm.utils.compare_to_groundtruth(self.gt, model_res, 'iou', distth=0.6)
        self.mh = mm.metrics.create()

        # 打印单个accumulator
        # mh模块中有内置的显示格式

    def get_results(self):
        summary = self.mh.compute_many([self.acc, self.acc.events.loc[0:1]],
                                       metrics=mm.metrics.motchallenge_metrics,
                                       names=['full', 'part'])

        strsummary = mm.io.render_summary(
            summary,
            formatters=self.mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        print(strsummary)
