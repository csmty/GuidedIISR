import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from s3diff import S3Diff
from my_utils.training_utils import parse_args_paired_training, ThermalPairDataset, paired_thermal_proc
from my_utils.dwt_loss import DWTLoss
from basicsr.metrics import calculate_psnr_pt, calculate_ssim_pt


def main(args):
    config = OmegaConf.load(args.base_config)

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_sr = S3Diff(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=args.pretrained_path,
        enable_so_lrrb=not args.disable_so_lrrb,
        so_lrrb_hidden_channels=args.so_lrrb_hidden_channels,
        so_lrrb_growth_channels=args.so_lrrb_growth_channels,
        so_lrrb_num_blocks=args.so_lrrb_num_blocks,
        so_lrrb_num_layers_per_block=args.so_lrrb_num_layers_per_block,
        so_lrrb_residual_scale=args.so_lrrb_residual_scale,
        so_lrrb_kappa=args.so_lrrb_kappa,
        enable_rgb_ref=not args.disable_rgb_ref,
        rgb_ref_num_summary_tokens=args.rgb_ref_num_summary_tokens,
    )
    net_sr.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
            net_sr.ref_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    net_dwt = DWTLoss().cuda()
    net_dwt.requires_grad_(False)

    layers_to_opt = []
    seen_params = set()

    def add_param(p):
        if id(p) not in seen_params:
            layers_to_opt.append(p)
            seen_params.add(id(p))

    for n, p in net_sr.unet.named_parameters():
        if "lora" in n and "ref_attn" not in n:
            assert p.requires_grad
            add_param(p)
    for p in net_sr.unet.conv_in.parameters():
        add_param(p)
    for n, p in net_sr.vae.named_parameters():
        if "lora" in n:
            assert p.requires_grad
            add_param(p)
    if net_sr.so_lrrb is not None:
        for p in net_sr.so_lrrb.parameters():
            add_param(p)
    for block in net_sr.rgb_ref_blocks:
        for p in block.ref_attn.parameters():
            add_param(p)

    dataset_train = ThermalPairDataset(config.train, sf=config.sf)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    dataset_val = ThermalPairDataset(config.validation, sf=config.sf)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    net_sr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_sr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)
    net_dwt = accelerator.prepare(net_dwt)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_dwt.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_sr):
                x_src, x_tgt, _, x_visible = paired_thermal_proc(batch, accelerator.device)
                prompt = [args.pos_prompt for _ in range(x_src.shape[0])]
                rgb_prompt = [args.rgb_ref_prompt for _ in range(x_src.shape[0])]

                x_tgt_pred = net_sr(x_src.detach(), prompt, visible_ref=x_visible, rgb_prompt=rgb_prompt)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips
                loss_dwt = net_dwt(x_tgt_pred.float(), x_tgt.detach().float()) * args.lambda_dwt
                loss = loss_l2 + loss_lpips + loss_dwt

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {
                        "loss": loss.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_dwt": loss_dwt.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_sr).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_dwt, l_psnr, l_ssim = [], [], [], [], []
                        metrics_rows = []
                        eval_dir = os.path.join(args.output_dir, "eval", f"step_{global_step:06d}")
                        net_sr_unwrapped = accelerator.unwrap_model(net_sr)
                        net_sr_unwrapped.set_eval()
                        if args.save_val:
                            os.makedirs(eval_dir, exist_ok=True)

                        for step_val, batch_val in enumerate(dl_val):
                            if step_val >= args.num_samples_eval:
                                break
                            x_src, x_tgt, _, x_visible = paired_thermal_proc(batch_val, accelerator.device)
                            with torch.no_grad():
                                prompt_val = [args.pos_prompt for _ in range(x_src.shape[0])]
                                rgb_prompt_val = [args.rgb_ref_prompt for _ in range(x_src.shape[0])]
                                x_tgt_pred = net_sr_unwrapped(x_src.detach(), prompt_val, visible_ref=x_visible, rgb_prompt=rgb_prompt_val)
                                l2_val = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean").item()
                                lpips_val = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean().item()
                                dwt_val = net_dwt(x_tgt_pred.float(), x_tgt.detach().float()).item()
                                pred_metric = (x_tgt_pred.float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
                                tgt_metric = (x_tgt.detach().float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
                                psnr_val = calculate_psnr_pt(pred_metric, tgt_metric, crop_border=0).mean().item()
                                ssim_val = calculate_ssim_pt(pred_metric, tgt_metric, crop_border=0).mean().item()

                            l_l2.append(l2_val)
                            l_lpips.append(lpips_val)
                            l_dwt.append(dwt_val)
                            l_psnr.append(psnr_val)
                            l_ssim.append(ssim_val)
                            metrics_rows.append((step_val, l2_val, lpips_val, dwt_val, psnr_val, ssim_val))

                            if args.save_val:
                                x_src_vis = x_src.cpu().detach() * 0.5 + 0.5
                                x_tgt_vis = x_tgt.cpu().detach() * 0.5 + 0.5
                                x_tgt_pred_vis = x_tgt_pred.cpu().detach() * 0.5 + 0.5
                                x_visible_vis = x_visible.cpu().detach() * 0.5 + 0.5
                                combined = torch.cat([x_src_vis, x_tgt_pred_vis, x_tgt_vis, x_visible_vis], dim=3).clamp(0, 1)
                                transforms.ToPILImage()(combined[0]).save(os.path.join(eval_dir, f"val_{step_val:03d}.png"))

                        logs["val/l2"] = np.mean(l_l2) if l_l2 else 0.0
                        logs["val/lpips"] = np.mean(l_lpips) if l_lpips else 0.0
                        logs["val/dwt"] = np.mean(l_dwt) if l_dwt else 0.0
                        logs["val/psnr"] = np.mean(l_psnr) if l_psnr else 0.0
                        logs["val/ssim"] = np.mean(l_ssim) if l_ssim else 0.0

                        if args.save_val:
                            metrics_path = os.path.join(eval_dir, "metrics.csv")
                            with open(metrics_path, "w", encoding="utf-8") as f:
                                f.write("index,l2,lpips,dwt,psnr,ssim\n")
                                for row in metrics_rows:
                                    f.write(f"{row[0]},{row[1]:.8f},{row[2]:.8f},{row[3]:.8f},{row[4]:.6f},{row[5]:.6f}\n")
                                f.write(f"mean,{logs['val/l2']:.8f},{logs['val/lpips']:.8f},{logs['val/dwt']:.8f},{logs['val/psnr']:.6f},{logs['val/ssim']:.6f}\n")
                        net_sr_unwrapped.set_train()
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
