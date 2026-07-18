#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

SEED="${SEED:-42}"
SD_PATH="${SD_PATH:-/data/matengyu/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2}"
RELEASE_CKPT="./checkpoints/GuidedIISR_model_48501.pkl"
SERVER_RELEASE_CKPT="./output_runs/run_20260717_154501_dwt_split_ref/checkpoints/model_48501.pkl"

if [[ -z "${PRETRAINED_PATH:-}" ]]; then
    if [[ -f "${RELEASE_CKPT}" ]]; then
        PRETRAINED_PATH="${RELEASE_CKPT}"
    elif [[ -f "${SERVER_RELEASE_CKPT}" ]]; then
        PRETRAINED_PATH="${SERVER_RELEASE_CKPT}"
    else
        echo "No GuidedIISR checkpoint found. Set PRETRAINED_PATH or place model_48501 at ${RELEASE_CKPT}." >&2
        exit 1
    fi
fi

CKPT_NAME="$(basename "${PRETRAINED_PATH}" .pkl)"
OUTPUT_DIR="${OUTPUT_DIR:-./output/inference/${CKPT_NAME}_seed${SEED}}"
REF_PATH="${REF_PATH:-/data/matengyu/code/@submit/AAAI26/InfraredSR/datasets_prepared/3MTI_4Datasets_x8_native/val/hr_thermal}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output directory already exists: ${OUTPUT_DIR}. Set OUTPUT_DIR to a new path to avoid overwriting results." >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

echo "GuidedIISR inference"
echo "  checkpoint: ${PRETRAINED_PATH}"
echo "  seed:       ${SEED}"
echo "  output:     ${OUTPUT_DIR}"

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port "${MAIN_PROCESS_PORT:-29300}" src/inference_guidediisr.py \
    --sd_path="${SD_PATH}" \
    --pretrained_path="${PRETRAINED_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --ref_path="${REF_PATH}" \
    --align_method="nofix" \
    --seed="${SEED}"
