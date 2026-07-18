from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

# Compatibility shim: imgaug used by pyiqa 0.1.7 still references np.sctypes,
# which was removed in NumPy 2.x. Define the old mapping before pyiqa is lazily imported.
import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }

import torch
from torchvision import transforms
from tqdm import tqdm

from guidediisr_metrics.metrics import (
    THREE_MTI_FULL_METRICS,
    THREE_MTI_STYLE,
    THREE_MTI_TRAIN_METRICS,
    ThreeMTIFullMetrics,
    compute_three_mti_fid,
    summarize_metric_rows,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="GuidedIISR image-folder evaluation with the 3MTI-style metric protocol.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", choices=["train", "full"], default="full")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--paired-resize-512", action="store_true")
    parser.add_argument("--noref-no-resize", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.max_samples is not None:
        pred_files = pred_files[: args.max_samples]
    if not pred_files:
        raise RuntimeError(f"No prediction images found under {pred_dir}")

    pairs = []
    missing = []
    for pred_path in pred_files:
        ref_path = ref_dir / pred_path.name
        if ref_path.exists():
            pairs.append((pred_path, ref_path))
        else:
            missing.append(pred_path.name)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} reference images, first examples: {missing[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_names = list(THREE_MTI_TRAIN_METRICS if args.metrics == "train" else THREE_MTI_FULL_METRICS)
    include_fid = "fid" in metric_names and not args.skip_fid
    tensor_metric_names = [m for m in metric_names if m != "fid"]
    metric_suite = ThreeMTIFullMetrics(
        device,
        metrics=tensor_metric_names,
        paired_resize_to=(512, 512) if args.paired_resize_512 else None,
        noref_resize_to=None if args.noref_no_resize else (512, 512),
    )

    rows = []
    with torch.no_grad():
        for pred_path, ref_path in tqdm(pairs, desc=f"guidediisr-{args.metrics}"):
            pred = load_rgb_tensor(pred_path).unsqueeze(0).to(device)
            ref = load_rgb_tensor(ref_path).unsqueeze(0).to(device)
            row = {"name": pred_path.name}
            row.update(metric_suite.evaluate_image(pred, ref))
            rows.append(row)

    summary = summarize_metric_rows(rows)
    if include_fid:
        summary["fid"] = compute_three_mti_fid(pred_dir, ref_dir, device=device)
    summary.update(
        {
            "num_images": len(rows),
            "pred_dir": str(pred_dir),
            "ref_dir": str(ref_dir),
            "metric_style": THREE_MTI_STYLE,
            "paired_resize_to": [512, 512] if args.paired_resize_512 else None,
            "noref_resize_to": None if args.noref_no_resize else [512, 512],
        }
    )

    with (out_dir / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name"] + tensor_metric_names)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {out_dir}")


if __name__ == "__main__":
    main()
