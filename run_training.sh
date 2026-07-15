accelerate launch --num_processes=2 --gpu_ids="0,1" --main_process_port 29300 src/train_s3diff.py \
    --output_dir="./output" \
    --resolution=512 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq 25
