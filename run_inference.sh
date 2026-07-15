export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/inference_s3diff.py \
    --output_dir="./output" \
    --ref_path="/data/matengyu/code/@submit/AAAI26/InfraredSR/datasets_prepared/3MTI_4Datasets_x8_native/val/hr_thermal" \
    --align_method="nofix"
