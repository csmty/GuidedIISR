import sys
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

p = "src/"
sys.path.append(p)
from model import make_1step_sched
from models.latent_residual_refinement import SOLRRB


class S3Diff(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, enable_so_lrrb=True, so_lrrb_hidden_channels=64, so_lrrb_growth_channels=32, so_lrrb_num_blocks=3, so_lrrb_num_layers_per_block=3, so_lrrb_residual_scale=0.2, so_lrrb_kappa=0.2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            lora_rank_vae = sd["rank_vae"]
            lora_rank_unet = sd["rank_unet"]
            target_modules_vae = sd["vae_lora_target_modules"]
            target_modules_unet = sd["unet_lora_target_modules"]

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            vae_state = vae.state_dict()
            vae_state.update(sd["state_dict_vae"])
            vae.load_state_dict(vae_state)

            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)
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


        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([50], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        if self.so_lrrb is not None:
            self.so_lrrb.eval()
            self.so_lrrb.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        if self.so_lrrb is not None:
            self.so_lrrb.train()
            self.so_lrrb.requires_grad_(True)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    def forward(self, c_t, prompt):
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
        r2 = self.so_lrrb(encoded_control, model_pred) if self.so_lrrb is not None else torch.zeros_like(model_pred)
        x_denoised = encoded_control - (model_pred + r2)
        return self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def save_model(self, outf):
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k},
            "enable_so_lrrb": self.so_lrrb is not None,
            "so_lrrb_config": self.so_lrrb_config,
            "state_dict_so_lrrb": self.so_lrrb.state_dict() if self.so_lrrb is not None else None,
        }
        torch.save(sd, outf)
