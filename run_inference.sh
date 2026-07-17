#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1

SD_PATH="${SD_PATH:-/data/matengyu/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2}"
PRETRAINED_PATH="${PRETRAINED_PATH:-$(ls -t ./output/checkpoints/model_*.pkl 2>/dev/null | head -n 1)}"
if [[ -z "${PRETRAINED_PATH}" ]]; then
    echo "No checkpoint found. Set PRETRAINED_PATH or train until output/checkpoints/model_*.pkl exists." >&2
    exit 1
fi

CKPT_NAME="$(basename "${PRETRAINED_PATH}" .pkl)"
OUTPUT_DIR="${OUTPUT_DIR:-./output/inference/${CKPT_NAME}}"
REF_PATH="${REF_PATH:-/data/matengyu/code/@submit/AAAI26/InfraredSR/datasets_prepared/3MTI_4Datasets_x8_native/val/hr_thermal}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/inference_s3diff.py \
    --sd_path="${SD_PATH}" \
    --pretrained_path="${PRETRAINED_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --ref_path="${REF_PATH}" \
    --align_method="nofix"
