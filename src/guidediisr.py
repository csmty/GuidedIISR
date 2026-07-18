import sys
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

p = "src/"
sys.path.append(p)
from model import make_1step_sched
from models.solr import SOLR
from models.rarg_reference_fusion import install_rarg_reference_fusion


def _upgrade_rarg_state_dict(state_dict):
    """Upgrade legacy RARG attention projection keys when loading older checkpoints."""
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
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, enable_solr=True, solr_hidden_channels=64, solr_growth_channels=32, solr_num_blocks=3, solr_num_layers_per_block=3, solr_residual_scale=0.2, solr_kappa=0.2, enable_rarg=True, rarg_num_summary_tokens=16, rarg_timestep=1):
        super().__init__()
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

        rarg_bank = []
        rarg_blocks = []
        rarg_block_paths = []

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            lora_rank_vae = sd["rank_vae"]
            lora_rank_unet = sd["rank_unet"]
            target_modules_vae = sd["vae_lora_target_modules"]
            target_modules_unet = sd["unet_lora_target_modules"]

            enable_rarg = sd.get("enable_rarg", sd.get("enable_rgb_ref", enable_rarg))
            rarg_config = sd.get("rarg_config", sd.get("rgb_ref_config", {}))
            rarg_num_summary_tokens = rarg_config.get("num_summary_tokens", rarg_num_summary_tokens)
            rarg_timestep = rarg_config.get("timestep", rarg_timestep)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            vae_state = vae.state_dict()
            vae_state.update(sd["state_dict_vae"])
            vae.load_state_dict(vae_state)

            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)
            if enable_rarg:
                rarg_blocks, rarg_block_paths = install_rarg_reference_fusion(
                    unet, ref_unet, rarg_bank, num_summary_tokens=rarg_num_summary_tokens
                )
            unet_state = unet.state_dict()
            unet_state.update(sd["state_dict_unet"])
            unet.load_state_dict(unet_state)

            enable_solr = sd.get("enable_solr", sd.get("enable_so_lrrb", enable_solr))
            solr_config = sd.get("solr_config", sd.get("so_lrrb_config", {}))
            solr_hidden_channels = solr_config.get("hidden_channels", solr_hidden_channels)
            solr_growth_channels = solr_config.get("growth_channels", solr_growth_channels)
            solr_num_blocks = solr_config.get("num_blocks", solr_num_blocks)
            solr_num_layers_per_block = solr_config.get("num_layers_per_block", solr_num_layers_per_block)
            solr_residual_scale = solr_config.get("residual_scale", solr_residual_scale)
            solr_kappa = solr_config.get("kappa", solr_kappa)
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
        self.enable_solr = enable_solr
        self.solr_config = {
            "latent_channels": 4,
            "hidden_channels": solr_hidden_channels,
            "growth_channels": solr_growth_channels,
            "num_blocks": solr_num_blocks,
            "num_layers_per_block": solr_num_layers_per_block,
            "residual_scale": solr_residual_scale,
            "kappa": solr_kappa,
        }
        self.solr = SOLR(**self.solr_config) if enable_solr else None
        if pretrained_path is not None and self.solr is not None:
            solr_state = sd.get("state_dict_solr", sd.get("state_dict_so_lrrb"))
            if solr_state is not None:
                self.solr.load_state_dict(solr_state)

        self.enable_rarg = enable_rarg
        self.rarg_config = {"num_summary_tokens": rarg_num_summary_tokens, "timestep": rarg_timestep}
        self.rarg_bank = rarg_bank
        if enable_rarg and not rarg_blocks:
            rarg_blocks, rarg_block_paths = install_rarg_reference_fusion(
                unet, ref_unet, self.rarg_bank, num_summary_tokens=rarg_num_summary_tokens
            )
        self.rarg_blocks, self.rarg_block_paths = rarg_blocks, rarg_block_paths
        if pretrained_path is not None and self.rarg_blocks:
            block_states = sd.get("state_dict_rarg_blocks", sd.get("state_dict_rgb_ref_blocks", {}))
            for path, block in zip(self.rarg_block_paths, self.rarg_blocks):
                if path in block_states:
                    block.ref_attn.load_state_dict(_upgrade_rarg_state_dict(block_states[path]))


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
        self.rarg_timesteps = torch.tensor([rarg_timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        super().eval()
        self.unet.eval()
        self.vae.eval()
        self.rgb_vae.eval()
        self.ref_unet.eval()
        self.text_encoder.eval()
        self.rgb_vae.requires_grad_(False)
        if self.solr is not None:
            self.solr.eval()
            self.solr.requires_grad_(False)
        for block in self.rarg_blocks:
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
        if self.solr is not None:
            self.solr.train()
            self.solr.requires_grad_(True)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        for block in self.rarg_blocks:
            block.ref_attn.train()
            block.ref_attn.requires_grad_(True)
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    def forward(self, c_t, prompt, visible_ref=None, rgb_prompt=None):
        """Super-resolve thermal LR input with optional RARG visible-reference guidance."""
        caption_enc = self._encode_prompt(prompt)
        rgb_caption_enc = self._encode_prompt(rgb_prompt if rgb_prompt is not None else prompt)
        encoded_control = self._encode_vae_latent(self.vae, c_t)
        self._write_visible_reference(visible_ref, rgb_caption_enc)
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
        r2 = self.solr(encoded_control, model_pred) if self.solr is not None else torch.zeros_like(model_pred)
        x_denoised = encoded_control - (model_pred + r2)
        return self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def _encode_prompt(self, prompt):
        """Tokenize a thermal or visible-reference prompt and return CLIP text features."""
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        return self.text_encoder(caption_tokens)[0]

    def _encode_vae_latent(self, vae, image):
        """Encode an image into SD latent space; use deterministic mode in eval."""
        latent_dist = vae.encode(image).latent_dist
        latent = latent_dist.sample() if self.training else latent_dist.mode()
        return latent * vae.config.scaling_factor

    def _write_visible_reference(self, visible_ref, caption_enc):
        """Run the frozen visible branch once so RARG can cache per-block reference features."""
        self.rarg_bank.clear()
        if not self.rarg_blocks or visible_ref is None:
            return
        with torch.no_grad():
            visible_latent = self._encode_vae_latent(self.rgb_vae, visible_ref)
            _ = self.ref_unet(visible_latent, self.rarg_timesteps, encoder_hidden_states=caption_enc).sample

    def save_model(self, outf):
        """Save release checkpoints with SOLR/RARG keys only."""
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if ("ref_attn" not in k and ("lora" in k or "conv_in" in k))},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k},
            "enable_solr": self.solr is not None,
            "solr_config": self.solr_config,
            "state_dict_solr": self.solr.state_dict() if self.solr is not None else None,
            "enable_rarg": bool(self.rarg_blocks),
            "rarg_config": self.rarg_config,
            "rarg_block_paths": self.rarg_block_paths,
            "state_dict_rarg_blocks": {path: block.ref_attn.state_dict() for path, block in zip(self.rarg_block_paths, self.rarg_blocks)},
        }
        torch.save(sd, outf)
