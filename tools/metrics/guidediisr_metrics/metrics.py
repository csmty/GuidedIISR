from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

THREE_MTI_PAIRED_METRICS: Tuple[str, ...] = ("psnr", "ssim", "lpips", "dists")
THREE_MTI_NOREF_METRICS: Tuple[str, ...] = ("clipiqa", "musiq", "niqe", "maniqa")
THREE_MTI_FULL_METRICS: Tuple[str, ...] = (
    "psnr",
    "ssim",
    "lpips",
    "dists",
    "clipiqa",
    "musiq",
    "niqe",
    "maniqa",
    "fid",
)
THREE_MTI_TRAIN_METRICS: Tuple[str, ...] = ("psnr", "ssim")

METRIC_DIRECTIONS: Mapping[str, str] = {
    "psnr": "up",
    "ssim": "up",
    "lpips": "down",
    "dists": "down",
    "clipiqa": "up",
    "musiq": "up",
    "niqe": "down",
    "maniqa": "up",
    "fid": "down",
}

THREE_MTI_STYLE = (
    "3MTI-style pyiqa metrics: PSNR/SSIM use test_y_channel=True and color_space='ycbcr'; "
    "LPIPS/DISTS are paired pyiqa metrics; CLIPIQA/MUSIQ/NIQE/MANIQA are no-reference "
    "pyiqa metrics on bilinear-resized 512x512 predictions by default; FID uses pyiqa fid "
    "between the evaluated prediction/reference image directories."
)


def _ensure_torch_home() -> None:
    """Prefer existing project/3MTI metric caches before pyiqa tries to download weights."""
    if os.environ.get("TORCH_HOME"):
        return
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "model_cache" / "torch",
        project_root.parent / "3MTI" / "model_cache" / "torch",
        Path("/data/matengyu/cch/3MTI/model_cache/torch"),
    ]
    hf_candidates = [
        project_root / "model_cache" / "huggingface",
        project_root.parent / "3MTI" / "model_cache" / "huggingface",
        Path("/data/matengyu/cch/3MTI/model_cache/huggingface"),
    ]
    hf_home = next((p for p in hf_candidates if p.exists()), hf_candidates[0])
    for cand in candidates:
        if (cand / "hub" / "pyiqa").exists() or (cand / "hub" / "checkpoints").exists():
            os.environ["TORCH_HOME"] = str(cand)
            if not os.environ.get("HF_HOME"):
                os.environ["HF_HOME"] = str(hf_home)
            return
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(hf_home)


def _import_pyiqa():
    _ensure_torch_home()
    try:
        import pyiqa  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("3MTI-style metrics require pyiqa. Install/import pyiqa in this environment.") from exc
    return pyiqa


def _autocast_context(device: torch.device):
    # Keep IQA metrics in fp32.  With pyiqa==0.1.7, SSIM under CUDA
    # autocast/fp16 can return NaN even for finite images.
    return nullcontext()


def _normalize_metric_names(metrics: Iterable[str]) -> Tuple[str, ...]:
    names = tuple(str(m).lower() for m in metrics)
    unknown = [m for m in names if m not in THREE_MTI_FULL_METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Supported: {THREE_MTI_FULL_METRICS}")
    return names


def _resize_if_needed(x: torch.Tensor, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    if size is None:
        return x
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _metric_to_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().flatten().float().mean().cpu().item())
    return float(value)


def create_three_mti_metric(name: str, device: torch.device):
    """Create one pyiqa metric with the same metric definitions used by 3MTI."""
    name = name.lower()
    pyiqa = _import_pyiqa()
    try:
        if name == "psnr":
            metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr")
        elif name == "ssim":
            metric = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr")
        elif name in {"lpips", "dists", "clipiqa", "musiq", "niqe", "maniqa", "fid"}:
            metric = pyiqa.create_metric(name)
        else:
            raise ValueError(name)
    except Exception as exc:  # pragma: no cover - depends on external cached metric weights
        raise RuntimeError(
            f"Failed to create 3MTI-style metric '{name}'. "
            f"Check cached pyiqa/HF weights. TORCH_HOME={os.environ.get('TORCH_HOME')} "
            f"HF_HOME={os.environ.get('HF_HOME')}"
        ) from exc
    metric = metric.to(device)
    if hasattr(metric, "eval"):
        metric.eval()
    return metric


class ThreeMTITrainMetrics:
    """Fast train-time 3MTI-style metrics: PSNR and SSIM only."""

    def __init__(
        self,
        device: torch.device,
        metrics: Sequence[str] = THREE_MTI_TRAIN_METRICS,
        paired_resize_to: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.device = device
        self.metric_names = _normalize_metric_names(metrics)
        disallowed = [m for m in self.metric_names if m not in {"psnr", "ssim"}]
        if disallowed:
            raise ValueError(f"Train metrics should only use PSNR/SSIM, got {disallowed}")
        self.paired_resize_to = paired_resize_to
        self.metrics = {name: create_three_mti_metric(name, device) for name in self.metric_names}

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred = _resize_if_needed(pred.clamp(0, 1), self.paired_resize_to)
        target = _resize_if_needed(target.clamp(0, 1), self.paired_resize_to)
        out: Dict[str, float] = {}
        with _autocast_context(self.device):
            for name, metric in self.metrics.items():
                out[name] = _metric_to_float(metric(pred, target))
        return out


class ThreeMTIFullMetrics:
    """Full 3MTI-style evaluation suite.

    Per-image metrics are computed from tensors. FID is directory-level and should be
    computed after the evaluated prediction/reference image directories have been saved.
    """

    def __init__(
        self,
        device: torch.device,
        metrics: Sequence[str] = THREE_MTI_FULL_METRICS,
        paired_resize_to: Optional[Tuple[int, int]] = None,
        noref_resize_to: Optional[Tuple[int, int]] = (512, 512),
    ) -> None:
        self.device = device
        self.metric_names = _normalize_metric_names(metrics)
        self.paired_resize_to = paired_resize_to
        self.noref_resize_to = noref_resize_to
        self.tensor_metric_names = tuple(m for m in self.metric_names if m != "fid")
        self.paired_names = tuple(m for m in self.tensor_metric_names if m in THREE_MTI_PAIRED_METRICS)
        self.noref_names = tuple(m for m in self.tensor_metric_names if m in THREE_MTI_NOREF_METRICS)
        self.metrics = {name: create_three_mti_metric(name, device) for name in self.tensor_metric_names}

    @torch.no_grad()
    def evaluate_image(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target is not None and target.ndim == 3:
            target = target.unsqueeze(0)
        pred = pred.clamp(0, 1)
        target = None if target is None else target.clamp(0, 1)

        values: Dict[str, float] = {}
        with _autocast_context(self.device):
            if self.paired_names:
                if target is None:
                    raise ValueError(f"Paired metrics {self.paired_names} require target images")
                pred_pair = _resize_if_needed(pred, self.paired_resize_to)
                target_pair = _resize_if_needed(target, self.paired_resize_to)
                for name in self.paired_names:
                    values[name] = _metric_to_float(self.metrics[name](pred_pair, target_pair))

            if self.noref_names:
                pred_noref = _resize_if_needed(pred, self.noref_resize_to)
                for name in self.noref_names:
                    values[name] = _metric_to_float(self.metrics[name](pred_noref))
        return values

    @torch.no_grad()
    def evaluate_batch(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor],
        names: Sequence[str],
    ) -> List[Dict[str, float | str]]:
        rows: List[Dict[str, float | str]] = []
        for i, name in enumerate(names):
            row: Dict[str, float | str] = {"name": str(name)}
            target_i = None if target is None else target[i : i + 1]
            row.update(self.evaluate_image(pred[i : i + 1], target_i))
            rows.append(row)
        return rows


def summarize_metric_rows(rows: Sequence[Mapping[str, float | str]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if key == "name" or value is None:
                continue
            if isinstance(value, str):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / max(1, counts[key]) for key in sums}


def compute_three_mti_fid(pred_dir: str | Path, ref_dir: str | Path, device: Optional[torch.device] = None) -> float:
    pyiqa = _import_pyiqa()
    metric = pyiqa.create_metric("fid")
    if device is not None:
        metric = metric.to(device)
    value = metric(str(pred_dir), str(ref_dir))
    return _metric_to_float(value)


# Legacy lightweight metrics kept for older stage1 scripts and quick debugging.
def _crop_border(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def _to_metric_channel(x: torch.Tensor, mode: str = "gray") -> torch.Tensor:
    mode = mode.lower()
    if mode in {"rgb", "all"}:
        return x
    if mode in {"gray", "thermal", "y"}:
        if x.shape[1] == 1:
            return x
        return x.mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown metric channel mode: {mode}")


def calc_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    channel: str = "gray",
    eps: float = 1e-12,
) -> torch.Tensor:
    pred = _to_metric_channel(_crop_border(pred.clamp(0, 1), crop_border), channel)
    target = _to_metric_channel(_crop_border(target.clamp(0, 1), crop_border), channel)
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse + eps)


def _ssim_per_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    dims = (-1, -2, -3)
    mu_x = x.mean(dim=dims)
    mu_y = y.mean(dim=dims)
    sigma_x = ((x - mu_x[:, None, None, None]) ** 2).mean(dim=dims)
    sigma_y = ((y - mu_y[:, None, None, None]) ** 2).mean(dim=dims)
    sigma_xy = ((x - mu_x[:, None, None, None]) * (y - mu_y[:, None, None, None])).mean(dim=dims)
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )


def calc_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    channel: str = "gray",
) -> torch.Tensor:
    pred = _to_metric_channel(_crop_border(pred.clamp(0, 1), crop_border), channel)
    target = _to_metric_channel(_crop_border(target.clamp(0, 1), crop_border), channel)
    return _ssim_per_batch(pred, target)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


class MetricTracker:
    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        self.meters.setdefault(key, AverageMeter()).update(value, n=n)

    def as_dict(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.meters.items()}
