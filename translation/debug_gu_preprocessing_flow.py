"""Debug GU-style preprocessing with configurable lattice size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spkData.export_hdf5_events import write_event_hdf5  # noqa: E402
from translation.debug_preprocessing_flow import (  # noqa: E402
    Report,
    check_population_match,
    h5_tree,
    inspect_event_file,
    inspect_in_file,
    save_event_preview,
    write_json_report,
)
from translation.spikenet_preprocessing import GU2018Config, build_gu_2018_input  # noqa: E402


def make_synthetic_event_file(path: Path, *, frames: int, height: int, width: int) -> Path:
    data = np.zeros((frames, height, width), dtype=np.uint8)
    box_h = max(2, height // 6)
    box_w = max(2, width // 6)
    for t in range(frames):
        y0 = int(round(t * max(height - box_h - 1, 1) / max(frames - 1, 1)))
        x0 = int(round(t * max(width - box_w - 1, 1) / max(frames - 1, 1)))
        data[t, y0 : y0 + box_h, x0 : x0 + box_w] = 1
    write_event_hdf5(path, data, dt=1, start_time=0, threshold=0, pol_value=1)
    return path


def inspect_gu_metadata(in_h5: Path) -> dict:
    with h5py.File(in_h5, "r") as h5f:
        n = np.asarray(h5f["/config/Net/INIT001/N"][()], dtype=int)
        syn_counts = {}
        n_syns = int(h5f["/config/syns/n_syns"][()])
        for idx in range(n_syns):
            base = f"/config/syns/syn{idx}/INIT006"
            syn_counts[f"syn{idx}"] = {
                "type": int(h5f[f"{base}/type"][()]),
                "pop_pre": int(h5f[f"{base}/i_pre"][()]),
                "pop_post": int(h5f[f"{base}/j_post"][()]),
                "count": int(len(h5f[f"{base}/I"])),
                "k_min": float(np.min(h5f[f"{base}/K"][()])) if len(h5f[f"{base}/K"]) else None,
                "k_max": float(np.max(h5f[f"{base}/K"][()])) if len(h5f[f"{base}/K"]) else None,
            }
    return {"N": n.tolist(), "synapses": syn_counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-e", type=Path)
    parser.add_argument("--event-i", type=Path)
    parser.add_argument("--make-synthetic", action="store_true")
    parser.add_argument("--lattice-shape", nargs=2, type=int, default=(15, 15), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--n-i", type=int)
    parser.add_argument("--step-tot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--connection-device", default="auto")
    parser.add_argument("--post-chunk-size", type=int, default=512)
    parser.add_argument("--p-scale", type=float, default=1.0, help="Scale GU P_mat for lightweight debug runs.")
    parser.add_argument("--disable-inverse-pool", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("translation/debug_outputs/gu_preprocessing"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    height, width = int(args.lattice_shape[0]), int(args.lattice_shape[1])
    event_e = args.event_e
    if args.make_synthetic:
        event_e = args.output_dir / f"synthetic_gu_{height}x{width}_events.h5"
        make_synthetic_event_file(event_e, frames=12, height=height, width=width)
        report.pass_("synthetic event file", str(event_e))
    if event_e is None:
        report.fail("event input", "provide --event-e or use --make-synthetic")
        report.print()
        return 1

    event_i = args.event_i
    event_summary = inspect_event_file(event_e, report)
    if event_summary is not None:
        save_event_preview(event_e, args.output_dir / f"{event_e.stem}_preview.png")
        check_population_match(event_summary, (height, width), report)

    p_base = np.asarray([[0.16, 0.2], [0.2, 0.4]], dtype=float) * float(args.p_scale)
    cfg = GU2018Config(
        lattice_shape=(height, width),
        n_i=args.n_i,
        step_tot=args.step_tot,
        p_mat=((float(p_base[0, 0]), float(p_base[0, 1])), (float(p_base[1, 0]), float(p_base[1, 1]))),
        use_inverse_pool=not args.disable_inverse_pool,
        post_chunk_size=args.post_chunk_size,
        connection_device=args.connection_device,
    )
    in_h5 = args.output_dir / f"gu_{height}x{width}_debug_in.h5"
    build_gu_2018_input(in_h5, event_e, event_i, config=cfg, loop_num=1, seed=args.seed)
    report.pass_("GU preprocessing build", str(in_h5))
    in_summary = inspect_in_file(in_h5, event_e, event_i or event_e, report)
    tree_path = args.output_dir / f"gu_{height}x{width}_debug_in_tree.txt"
    tree_path.write_text("\n".join(h5_tree(in_h5)), encoding="utf-8")
    report.pass_("in.h5 tree dump", str(tree_path))

    payload = {
        "event": event_summary,
        "in_h5": in_summary,
        "gu": inspect_gu_metadata(in_h5),
        "checks": [check.__dict__ for check in report.checks],
    }
    report_path = args.output_dir / f"gu_{height}x{width}_debug_report.json"
    write_json_report(report_path, payload)
    report.pass_("json report", str(report_path))
    report.print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
