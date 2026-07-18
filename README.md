# GuidedIISR

GuidedIISR is a visible-guided infrared image super-resolution method. Given a low-resolution infrared image and a paired high-resolution visible reference image, the model reconstructs a high-resolution infrared image.

The method uses a diffusion prior with LoRA tuning, latent-space residual refinement, and SV-CM-AICG reference attention for controlled visible-to-infrared feature injection.

## Released Model

The released checkpoint is `model_48501.pkl` from the run:

```text
output_runs/run_20260717_154501_dwt_split_ref/checkpoints/model_48501.pkl
```

For release packaging, place it at:

```text
checkpoints/GuidedIISR_model_48501.pkl
```

Current validation/test result on 1176 images with the 3MTI-style metric protocol:

```text
PSNR: 30.32744
SSIM: 0.86670
```

## Environment

```bash
conda activate <your-conda-env>
pip install -r requirements.txt
```

If the SD-Turbo files are not downloaded automatically, set `SD_PATH` to a local SD-Turbo directory.

## Data

The dataset should contain paired infrared and visible images. Files are paired by identical filename stem.

```text
train/lr_thermal
train/hr_thermal
train/hr_visible
val/lr_thermal
val/hr_thermal
val/hr_visible
```

Update dataset paths in:

```text
configs/sr.yaml
configs/sr_test.yaml
```

## Inference

Default inference uses the released checkpoint, fixed seed `42`, thermal timestep `50`, visible-reference timestep `1`, and `align_method=nofix`.

```bash
bash run_inference.sh
```

Common overrides:

```bash
PRETRAINED_PATH=/path/to/GuidedIISR_model_48501.pkl \
OUTPUT_DIR=./output/inference/model_48501_seed42 \
SEED=42 \
bash run_inference.sh
```

`OUTPUT_DIR` must be a new directory; the script will not overwrite existing results.

## Training

```bash
bash run_training.sh
```

Current training defaults:

```text
loss = 1.0 * L2 + 5.0 * LPIPS + 1.0 * DWT
GAN loss = disabled
gradient_accumulation_steps = 2
seed = 42
validation samples = 20
```

The visible-reference VAE encoder and reference UNet are frozen. The thermal branch trains LoRA parameters, VAE encoder LoRA, input convolution, latent residual refinement, and reference-attention injection parameters.

## Metrics

Training validation and final testing use the same 3MTI-style PSNR/SSIM definition:

```python
pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr")
pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr")
```

To evaluate saved predictions:

```bash
cd tools/metrics
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python evaluate.py \
  --pred-dir /path/to/pred \
  --ref-dir /path/to/hr_thermal \
  --output-dir /path/to/metrics \
  --metrics train
```

The metric script pairs predictions and ground truth by filename, then writes per-image metrics and a summary file.
