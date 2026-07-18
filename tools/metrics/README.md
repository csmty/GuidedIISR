# GuidedIISR Metrics

This directory contains the evaluation utility used for GuidedIISR release results.

## Layout

```text
tools/metrics/evaluate.py
tools/metrics/guidediisr_metrics/metrics.py
tools/metrics/guidediisr_metrics/__init__.py
```

## Metric Protocol

The default paired training metrics are PSNR and SSIM with the current 3MTI-style definition:

```python
pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr")
pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr")
```

Predictions and references are paired by identical filenames. The script writes per-image results to `metrics_per_image.csv` and aggregate results to `summary.json`.

## Usage

```bash
cd tools/metrics
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python evaluate.py \
  --pred-dir /path/to/pred \
  --ref-dir /path/to/hr_thermal \
  --output-dir /path/to/metrics \
  --metrics train
```

Use `--metrics full` to also compute LPIPS, DISTS, CLIPIQA, MUSIQ, NIQE, MANIQA, and FID when the required metric weights are available.
