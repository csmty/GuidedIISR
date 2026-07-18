# GuidedIISR

GuidedIISR is a visible-reference guided infrared image super-resolution model. It super-resolves low-resolution thermal images with paired high-resolution visible reference images, using SD-Turbo priors, LoRA fine-tuning, latent residual refinement, and SV-CM-AICG reference attention.

## Release Checkpoint

The released checkpoint is trained at step `48501`.

Recommended release layout:

```text
checkpoints/GuidedIISR_model_48501.pkl
```

On the training server, `run_inference.sh` also falls back to:

```text
output_runs/run_20260717_154501_dwt_split_ref/checkpoints/model_48501.pkl
```

The selected checkpoint gives the following validation/test results on 1176 images with the current 3MTI-style PSNR/SSIM protocol:

```text
PSNR: 30.32744
SSIM: 0.86670
```

## Environment

Use Python 3.10 and the project environment used for training:

```bash
conda activate <your-conda-env>
pip install -r requirements.txt
```

If running offline, set `SD_PATH` to a local SD-Turbo directory.

## Data Layout

The config files expect paired thermal and visible-reference folders:

```text
train/lr_thermal
train/hr_thermal
train/hr_visible
val/lr_thermal
val/hr_thermal
val/hr_visible
```

Edit these paths in:

```text
configs/sr.yaml
configs/sr_test.yaml
```

Images are paired by filename stem.

## Inference

Put the release checkpoint at `checkpoints/GuidedIISR_model_48501.pkl`, or set `PRETRAINED_PATH` explicitly.

```bash
bash run_inference.sh
```

Useful overrides:

```bash
PRETRAINED_PATH=/path/to/GuidedIISR_model_48501.pkl \
OUTPUT_DIR=./output/inference/model_48501_seed42 \
SEED=42 \
bash run_inference.sh
```

Inference uses:

```text
seed = 42
align_method = nofix
thermal UNet timestep = 50
visible-reference UNet timestep = 1
```

The script writes generated images and `results.txt` to the output directory.

## Training

Training uses paired LR-HR thermal data and HR visible references.

```bash
bash run_training.sh
```

Current default training settings:

```text
loss = 1.0 * L2 + 5.0 * LPIPS + 1.0 * DWT
GAN loss: disabled
gradient_accumulation_steps: 2
seed: 42
validation samples: 20
```

Validation is run in eval mode and computes PSNR/SSIM using the same 3MTI-style paired metric definition used for final testing:

```python
pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr")
pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr")
```

Saved validation files include per-image `metrics.csv`; the PSNR/SSIM columns are `psnr_y` and `ssim_y`.

## Metrics

The release PSNR/SSIM protocol is Y-channel pyiqa without crop border and pairs images by identical filenames.

To reproduce the 3MTI-style PSNR/SSIM calculation with the provided metric script:

```bash
cd tools/metric_scripts_20260718/metric_scripts
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/eval_image_folder_3mti_style.py \
  --pred-dir /path/to/pred \
  --ref-dir /path/to/hr_thermal \
  --output-dir /path/to/metrics_3mti_train \
  --metrics train
```

## Notes

The visible-reference VAE encoder and reference UNet are frozen. The thermal branch trains LoRA parameters, VAE encoder LoRA, input convolution, latent residual refinement, and reference-attention injection parameters.
