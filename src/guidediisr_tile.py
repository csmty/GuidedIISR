import sys
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

p = "src/"
sys.path.append(p)
from model import make_1step_sched
from models.latent_residual_refinement import SOLRRB
from models.unet_reference_fusion import install_unet_reference_fusion


def _upgrade_rgb_ref_state_dict(state_dict):
    upgraded = dict(state_dict)
    for key, value in state_dict.items():
        if key.startswith("to_k."):
            suffix = key[len("to_k."):]
            upgraded.setdefault(f"to_k_t.{suffix}", value)
            upgraded.setdefault(f"to_k_rgb.{suffix}", value)
            upgraded.pop(key, None)
        elif key.startswith("to_v."):
            suffix = key[len("to_v."):]
            upgraded.setdefault(f"to_v_rgb.{suffix}", value)
            upgraded.pop(key, None)
    return upgraded


class GuidedIISR(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, args=None, enable_so_lrrb=True, so_lrrb_hidden_channels=64, so_lrrb_growth_channels=32, so_lrrb_num_blocks=3, so_lrrb_num_layers_per_block=3, so_lrrb_residual_scale=0.2, so_lrrb_kappa=0.2, enable_rgb_ref=True, rgb_ref_num_summary_tokens=16, rgb_ref_timestep=1):
        super().__init__()
        self.args = args
        self.guidance_scale = 1.07
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        rgb_vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        ref_unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        rgb_ref_bank = []
        rgb_ref_blocks = []
        rgb_ref_block_paths = []

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            lora_rank_vae = sd["rank_vae"]
            lora_rank_unet = sd["rank_unet"]
            target_modules_vae = sd["vae_lora_target_modules"]
            target_modules_unet = sd["unet_lora_target_modules"]

            enable_rgb_ref = sd.get("enable_rgb_ref", enable_rgb_ref)
            rgb_ref_config = sd.get("rgb_ref_config", {})
            rgb_ref_num_summary_tokens = rgb_ref_config.get("num_summary_tokens", rgb_ref_num_summary_tokens)
            rgb_ref_timestep = rgb_ref_config.get("timestep", rgb_ref_timestep)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            vae_state = vae.state_dict()
            vae_state.update(sd["state_dict_vae"])
            vae.load_state_dict(vae_state)

            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)
            if enable_rgb_ref:
                rgb_ref_blocks, rgb_ref_block_paths = install_unet_reference_fusion(
                    unet, ref_unet, rgb_ref_bank, num_summary_tokens=rgb_ref_num_summary_tokens
                )
            unet_state = unet.state_dict()
            unet_state.update(sd["state_dict_unet"])
            unet.load_state_dict(unet_state)

            enable_so_lrrb = sd.get("enable_so_lrrb", enable_so_lrrb)
            so_lrrb_config = sd.get("so_lrrb_config", {})
            so_lrrb_hidden_channels = so_lrrb_config.get("hidden_channels", so_lrrb_hidden_channels)
            so_lrrb_growth_channels = so_lrrb_config.get("growth_channels", so_lrrb_growth_channels)
            so_lrrb_num_blocks = so_lrrb_config.get("num_blocks", so_lrrb_num_blocks)
            so_lrrb_num_layers_per_block = so_lrrb_config.get("num_layers_per_block", so_lrrb_num_layers_per_block)
            so_lrrb_residual_scale = so_lrrb_config.get("residual_scale", so_lrrb_residual_scale)
            so_lrrb_kappa = so_lrrb_config.get("kappa", so_lrrb_kappa)
        else:
            print("Initializing model with standard LoRA weights")
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet
        self.enable_so_lrrb = enable_so_lrrb
        self.so_lrrb_config = {
            "latent_channels": 4,
            "hidden_channels": so_lrrb_hidden_channels,
            "growth_channels": so_lrrb_growth_channels,
            "num_blocks": so_lrrb_num_blocks,
            "num_layers_per_block": so_lrrb_num_layers_per_block,
            "residual_scale": so_lrrb_residual_scale,
            "kappa": so_lrrb_kappa,
        }
        self.so_lrrb = SOLRRB(**self.so_lrrb_config) if enable_so_lrrb else None
        if pretrained_path is not None and self.so_lrrb is not None and "state_dict_so_lrrb" in sd:
            self.so_lrrb.load_state_dict(sd["state_dict_so_lrrb"])

        self.enable_rgb_ref = enable_rgb_ref
        self.rgb_ref_config = {"num_summary_tokens": rgb_ref_num_summary_tokens, "timestep": rgb_ref_timestep}
        self.rgb_ref_bank = rgb_ref_bank
        if enable_rgb_ref and not rgb_ref_blocks:
            rgb_ref_blocks, rgb_ref_block_paths = install_unet_reference_fusion(
                unet, ref_unet, self.rgb_ref_bank, num_summary_tokens=rgb_ref_num_summary_tokens
            )
        self.rgb_ref_blocks, self.rgb_ref_block_paths = rgb_ref_blocks, rgb_ref_block_paths
        if pretrained_path is not None and self.rgb_ref_blocks and "state_dict_rgb_ref_blocks" in sd:
            block_states = sd["state_dict_rgb_ref_blocks"]
            for path, block in zip(self.rgb_ref_block_paths, self.rgb_ref_blocks):
                if path in block_states:
                    block.ref_attn.load_state_dict(_upgrade_rgb_ref_state_dict(block_states[path]))


        unet.to("cuda")
        vae.to("cuda")
        rgb_vae.to("cuda")
        ref_unet.to("cuda")
        rgb_vae.eval()
        rgb_vae.requires_grad_(False)
        ref_unet.eval()
        ref_unet.requires_grad_(False)
        self.unet, self.vae, self.rgb_vae, self.ref_unet = unet, vae, rgb_vae, ref_unet
        self.timesteps = torch.tensor([50], device="cuda").long()
        self.rgb_ref_timesteps = torch.tensor([rgb_ref_timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        super().eval()
        self.unet.eval()
        self.vae.eval()
        self.rgb_vae.eval()
        self.ref_unet.eval()
        self.text_encoder.eval()
        self.rgb_vae.requires_grad_(False)
        if self.so_lrrb is not None:
            self.so_lrrb.eval()
            self.so_lrrb.requires_grad_(False)
        for block in self.rgb_ref_blocks:
            block.ref_attn.eval()
            block.ref_attn.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        super().train()
        self.unet.train()
        self.vae.train()
        self.text_encoder.eval()
        self.rgb_vae.eval()
        self.rgb_vae.requires_grad_(False)
        self.ref_unet.eval()
        self.ref_unet.requires_grad_(False)
        if self.so_lrrb is not None:
            self.so_lrrb.train()
            self.so_lrrb.requires_grad_(True)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        for block in self.rgb_ref_blocks:
            block.ref_attn.train()
            block.ref_attn.requires_grad_(True)
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    @torch.no_grad()
    def forward(self, c_t, pos_prompt, neg_prompt=None, visible_ref=None, rgb_prompt=None):
        pos_caption_enc = self._encode_prompt(pos_prompt)
        rgb_caption_enc = self._encode_prompt(rgb_prompt if rgb_prompt is not None else pos_prompt)
        lq_latent = self._encode_vae_latent(self.vae, c_t)
        self._write_visible_reference(visible_ref, rgb_caption_enc)
        pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
        if neg_prompt is not None:
            neg_caption_enc = self._encode_prompt(neg_prompt)
            neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
            model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            model_pred = pos_model_pred
        r2 = self.so_lrrb(lq_latent, model_pred) if self.so_lrrb is not None else torch.zeros_like(model_pred)
        x_denoised = lq_latent - (model_pred + r2)
        return self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def _encode_prompt(self, prompt):
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        return self.text_encoder(caption_tokens)[0]


    def _encode_vae_latent(self, vae, image):
        latent_dist = vae.encode(image).latent_dist
        latent = latent_dist.sample() if self.training else latent_dist.mode()
        return latent * vae.config.scaling_factor

    def _write_visible_reference(self, visible_ref, caption_enc):
        self.rgb_ref_bank.clear()
        if not self.rgb_ref_blocks or visible_ref is None:
            return
        with torch.no_grad():
            visible_latent = self._encode_vae_latent(self.rgb_vae, visible_ref)
            _ = self.ref_unet(visible_latent, self.rgb_ref_timesteps, encoder_hidden_states=caption_enc).sample

    def save_model(self, outf):
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if ("ref_attn" not in k and ("lora" in k or "conv_in" in k))},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k},
            "enable_so_lrrb": self.so_lrrb is not None,
            "so_lrrb_config": self.so_lrrb_config,
            "state_dict_so_lrrb": self.so_lrrb.state_dict() if self.so_lrrb is not None else None,
            "enable_rgb_ref": bool(self.rgb_ref_blocks),
            "rgb_ref_config": self.rgb_ref_config,
            "rgb_ref_block_paths": self.rgb_ref_block_paths,
            "state_dict_rgb_ref_blocks": {path: block.ref_attn.state_dict() for path, block in zip(self.rgb_ref_block_paths, self.rgb_ref_blocks)},
        }
        torch.save(sd, outf)

# Backward-compatible alias for checkpoints/scripts created before the GuidedIISR release.
