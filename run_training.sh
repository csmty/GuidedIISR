export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1
accelerate launch --num_processes=2 --gpu_ids="0,1" --main_process_port 29300 src/train_s3diff.py \
    --sd_path="/data/matengyu/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2" \
    --output_dir="./output" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq 25
