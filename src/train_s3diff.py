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

    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(
            cv_type='dino',
            output_type='conv_multi_level',
            loss_type=args.gan_loss_type,
            device="cuda",
        )
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

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

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    net_sr, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_sr, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_sr, net_disc):
                x_src, x_tgt, _, x_visible = paired_thermal_proc(batch, accelerator.device)
                prompt = [args.pos_prompt for _ in range(x_src.shape[0])]

                x_tgt_pred = net_sr(x_src.detach(), prompt, visible_ref=x_visible)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                x_tgt_pred = net_sr(x_src.detach(), prompt, visible_ref=x_visible)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {
                        "lossG": lossG.detach().item(),
                        "lossD": lossD.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_sr).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        val_count = 0
                        for step_val, batch_val in enumerate(dl_val):
                            if step_val >= args.num_samples_eval:
                                break
                            x_src, x_tgt, _, x_visible = paired_thermal_proc(batch_val, accelerator.device)
                            with torch.no_grad():
                                prompt_val = [args.pos_prompt for _ in range(x_src.shape[0])]
                                x_tgt_pred = accelerator.unwrap_model(net_sr)(x_src.detach(), prompt_val, visible_ref=x_visible)
                                l_l2.append(F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean").item())
                                l_lpips.append(net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean().item())

                            if args.save_val and val_count < 5:
                                x_src_vis = x_src.cpu().detach() * 0.5 + 0.5
                                x_tgt_vis = x_tgt.cpu().detach() * 0.5 + 0.5
                                x_tgt_pred_vis = x_tgt_pred.cpu().detach() * 0.5 + 0.5
                                combined = torch.cat([x_src_vis, x_tgt_pred_vis, x_tgt_vis], dim=3)
                                transforms.ToPILImage()(combined[0]).save(os.path.join(args.output_dir, f"val_{step_val}.png"))
                                val_count += 1

                        logs["val/l2"] = np.mean(l_l2) if l_l2 else 0.0
                        logs["val/lpips"] = np.mean(l_lpips) if l_lpips else 0.0
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
