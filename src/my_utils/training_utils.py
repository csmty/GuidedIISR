import argparse
import io
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torch.utils import data as data


def parse_args_paired_training(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_dwt", default=1.0, type=float)
    parser.add_argument("--base_config", default="./configs/sr.yaml", type=str)

    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_val", default=True, action="store_false")
    parser.add_argument("--num_samples_eval", type=int, default=20)
    parser.add_argument("--viz_freq", type=int, default=100)

    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--neg_prob", default=0.05, type=float)
    parser.add_argument("--pos_prompt", type=str, default="high-resolution infrared thermal image, sharp thermal structures, clear object boundaries, fine texture details, low noise, natural thermal contrast, faithful spatial geometry")
    parser.add_argument("--neg_prompt", type=str, default=None)
    parser.add_argument("--rarg_prompt", type=str, default="high-resolution visible-light image, clear object boundaries, structural edges, scene layout, geometric contours, fine spatial details, natural perspective, possible slight viewpoint or alignment variation")
    parser.add_argument("--disable_solr", action="store_true")
    parser.add_argument("--solr_hidden_channels", type=int, default=64)
    parser.add_argument("--solr_growth_channels", type=int, default=32)
    parser.add_argument("--solr_num_blocks", type=int, default=3)
    parser.add_argument("--solr_num_layers_per_block", type=int, default=3)
    parser.add_argument("--solr_residual_scale", type=float, default=0.2)
    parser.add_argument("--solr_kappa", type=float, default=0.2)
    parser.add_argument("--disable_rarg", action="store_true")
    parser.add_argument("--rarg_num_summary_tokens", type=int, default=16)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(root):
    root = Path(root)
    return sorted(str(x) for x in root.rglob("*") if x.suffix.lower() in IMG_EXTENSIONS)


def _load_rgb(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def _to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _rgb_low_quality_augmentation(rgb):
    """Apply non-geometric quality degradation to the visible reference only."""
    if random.random() >= 0.4:
        return rgb

    selected = random.choice(("blur", "low_light", "overexpose", "noise", "jpeg"))
    if selected == "blur":
        radius = random.uniform(0.6, 2.0)
        kernel_size = max(3, int(math.ceil(radius * 6)) | 1)
        rgb = TF.gaussian_blur(rgb, kernel_size=[kernel_size, kernel_size], sigma=[radius, radius])
    elif selected == "low_light":
        rgb = rgb * random.uniform(0.35, 0.75)
    elif selected == "overexpose":
        factor = random.uniform(1.25, 1.8)
        rgb = rgb * factor + (factor - 1.0) * 0.15
    elif selected == "noise":
        rgb = rgb + torch.randn_like(rgb) * random.uniform(0.01, 0.05)
    elif selected == "jpeg":
        device, dtype = rgb.device, rgb.dtype
        buffer = io.BytesIO()
        TF.to_pil_image(rgb.detach().cpu().clamp(0.0, 1.0)).save(
            buffer,
            format="JPEG",
            quality=random.randint(25, 70),
        )
        buffer.seek(0)
        rgb = TF.to_tensor(Image.open(buffer).convert("RGB")).to(device=device, dtype=dtype)

    return rgb.clamp(0.0, 1.0)


def _augment_visible_reference(rgb, use_misalign=True):
    """Apply optional RGB-only geometric misalignment during training."""
    return _rgb_misalignment_augmentation(rgb) if use_misalign else rgb

def _rgb_misalignment_augmentation(rgb):
    """Apply 3MTI-style geometric misalignment to the visible reference only."""
    if random.random() >= 0.4:
        return rgb

    width, height = TF.get_image_size(rgb)
    translate = [
        round(random.uniform(-0.03, 0.03) * width),
        round(random.uniform(-0.03, 0.03) * height),
    ]
    scale = random.uniform(0.90, 1.10)
    angle = random.uniform(-5.0, 5.0)
    if translate != [0, 0] or abs(scale - 1.0) > 1e-6 or abs(angle) > 1e-6:
        rgb = TF.affine(
            rgb,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    max_dx = round(0.05 * width / 2)
    max_dy = round(0.05 * height / 2)
    if max_dx > 0 or max_dy > 0:
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [
            [random.randint(0, max_dx), random.randint(0, max_dy)],
            [width - 1 - random.randint(0, max_dx), random.randint(0, max_dy)],
            [width - 1 - random.randint(0, max_dx), height - 1 - random.randint(0, max_dy)],
            [random.randint(0, max_dx), height - 1 - random.randint(0, max_dy)],
        ]
        rgb = TF.perspective(
            rgb,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    return rgb


class ThermalPairDataset(data.Dataset):
    """Read paired LR thermal, HR thermal, and optional HR visible reference images."""

    def __init__(self, opt, sf=8):
        super().__init__()
        self.sf = int(sf)
        self.lr_root = Path(opt["lr_path"])
        self.hr_root = Path(opt["hr_path"])
        self.visible_root = Path(opt["visible_path"]) if "visible_path" in opt else None
        self.crop_size = int(opt.get("crop_size", 512))
        self.use_hflip = bool(opt.get("use_hflip", True))
        self.use_rot = bool(opt.get("use_rot", False))
        self.use_rgb_misalign = bool(opt.get("use_rgb_misalign", True))

        hr_by_name = {Path(p).name: p for p in _list_images(self.hr_root)}
        visible_by_name = {Path(p).name: p for p in _list_images(self.visible_root)} if self.visible_root is not None else {}
        self.pairs = []
        for lr_path in _list_images(self.lr_root):
            name = Path(lr_path).name
            if name in hr_by_name:
                self.pairs.append((lr_path, hr_by_name[name], visible_by_name.get(name)))
        if not self.pairs:
            raise ValueError(f"No paired thermal images found in {self.lr_root} and {self.hr_root}")

    def __getitem__(self, index):
        lr_path, hr_path, visible_path = self.pairs[index]
        img_lr = _load_rgb(lr_path)
        img_hr = _load_rgb(hr_path)
        img_visible = _load_rgb(visible_path) if visible_path is not None else None

        h_hr, w_hr = img_hr.shape[:2]
        crop_hr = min(self.crop_size, h_hr, w_hr)
        crop_hr = (crop_hr // self.sf) * self.sf
        crop_lr = crop_hr // self.sf

        if crop_hr > 0 and (h_hr > crop_hr or w_hr > crop_hr):
            h_lr, w_lr = img_lr.shape[:2]
            max_top_lr = min(h_lr - crop_lr, (h_hr - crop_hr) // self.sf)
            max_left_lr = min(w_lr - crop_lr, (w_hr - crop_hr) // self.sf)
            top_lr = random.randint(0, max(0, max_top_lr))
            left_lr = random.randint(0, max(0, max_left_lr))
            top_hr = top_lr * self.sf
            left_hr = left_lr * self.sf
            img_lr = img_lr[top_lr:top_lr + crop_lr, left_lr:left_lr + crop_lr, :]
            img_hr = img_hr[top_hr:top_hr + crop_hr, left_hr:left_hr + crop_hr, :]
            if img_visible is not None:
                img_visible = img_visible[top_hr:top_hr + crop_hr, left_hr:left_hr + crop_hr, :]

        if self.use_hflip and random.random() < 0.5:
            img_lr = np.flip(img_lr, axis=1).copy()
            img_hr = np.flip(img_hr, axis=1).copy()
            if img_visible is not None:
                img_visible = np.flip(img_visible, axis=1).copy()
        if self.use_rot and random.random() < 0.5:
            img_lr = np.flip(img_lr, axis=0).copy()
            img_hr = np.flip(img_hr, axis=0).copy()
            if img_visible is not None:
                img_visible = np.flip(img_visible, axis=0).copy()

        sample = {"lr": _to_tensor(img_lr), "gt": _to_tensor(img_hr), "lr_path": lr_path, "gt_path": hr_path}
        if img_visible is not None:
            visible_tensor = _augment_visible_reference(_to_tensor(img_visible), self.use_rgb_misalign)
            sample["visible"] = visible_tensor
            sample["visible_path"] = visible_path
        return sample

    def __len__(self):
        return len(self.pairs)


def paired_thermal_proc(batch, device):
    im_lr = batch["lr"].to(device, memory_format=torch.contiguous_format).float()
    im_gt = batch["gt"].to(device, memory_format=torch.contiguous_format).float()
    im_visible = batch.get("visible")
    if im_visible is not None:
        im_visible = im_visible.to(device, memory_format=torch.contiguous_format).float() * 2 - 1.0
    ori_lr = im_lr
    im_lr = F.interpolate(im_lr, size=im_gt.shape[-2:], mode="bicubic", align_corners=False)
    im_lr = torch.clamp(im_lr * 2 - 1.0, -1.0, 1.0)
    im_gt = im_gt * 2 - 1.0
    return im_lr.contiguous(), im_gt.contiguous(), ori_lr, im_visible
