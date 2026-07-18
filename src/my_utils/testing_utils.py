import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils import data as data


def parse_args_paired_testing(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default=None)
    parser.add_argument("--base_config", default="./configs/sr_test.yaml", type=str)
    parser.add_argument("--tracker_project_name", type=str, default="guidediisr")

    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)

    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--chop_size", type=int, default=128, choices=[512, 256, 128])
    parser.add_argument("--chop_stride", type=int, default=96)
    parser.add_argument("--padding_offset", type=int, default=32)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--align_method", type=str, default="wavelet")

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

    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class PlainDataset(data.Dataset):
    """Read LR thermal images and paired visible references for inference."""

    def __init__(self, opt):
        super().__init__()
        self.lr_paths = self._collect_image_paths(opt["lr_path"])
        self.visible_paths = None
        if "visible_path" in opt:
            visible_paths = self._collect_image_paths(opt["visible_path"])
            visible_by_stem = {}
            for visible_path in visible_paths:
                stem = Path(visible_path).stem
                if stem in visible_by_stem:
                    raise ValueError(f"Duplicate visible reference stem: {stem}")
                visible_by_stem[stem] = visible_path
            self.visible_paths = []
            for lr_path in self.lr_paths:
                stem = Path(lr_path).stem
                if stem not in visible_by_stem:
                    raise FileNotFoundError(f"Missing visible reference for {lr_path}")
                self.visible_paths.append(visible_by_stem[stem])

    @staticmethod
    def _collect_image_paths(root):
        roots = [root] if isinstance(root, str) else root
        paths = []
        for item in roots:
            base = Path(item)
            paths.extend(str(x) for x in base.iterdir() if x.suffix.lower() in IMG_EXTENSIONS)
        return sorted(paths)

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        sample = {"lr": transforms.ToTensor()(Image.open(lr_path).convert("RGB")), "lr_path": lr_path}
        if self.visible_paths is not None:
            visible_path = self.visible_paths[index]
            sample["visible"] = transforms.ToTensor()(Image.open(visible_path).convert("RGB"))
            sample["visible_path"] = visible_path
        return sample

    def __len__(self):
        return len(self.lr_paths)


def lr_proc(config, batch, device):
    im_lr = batch["lr"].to(device, memory_format=torch.contiguous_format).float()
    ori_lr = im_lr
    im_lr = F.interpolate(
        im_lr,
        size=(im_lr.size(-2) * config.sf, im_lr.size(-1) * config.sf),
        mode="bicubic",
        align_corners=False,
    )
    im_lr = torch.clamp(im_lr.contiguous() * 2 - 1.0, -1.0, 1.0)

    ori_h, ori_w = im_lr.size(-2), im_lr.size(-1)
    pad_h = math.ceil(ori_h / 64) * 64 - ori_h
    pad_w = math.ceil(ori_w / 64) * 64 - ori_w
    im_lr = F.pad(im_lr, pad=(0, pad_w, 0, pad_h), mode="reflect")

    return im_lr.to(device), ori_lr.to(device), (ori_h, ori_w)
