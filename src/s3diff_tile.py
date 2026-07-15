import sys
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

p = "src/"
sys.path.append(p)
from model import make_1step_sched


class S3Diff(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, args=None):
        super().__init__()
        self.args = args
        self.guidance_scale = 1.07
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

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([50], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    @torch.no_grad()
    def forward(self, c_t, pos_prompt, neg_prompt=None):
        pos_caption_enc = self._encode_prompt(pos_prompt)
        lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
        if neg_prompt is not None:
            neg_caption_enc = self._encode_prompt(neg_prompt)
            neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
            model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            model_pred = pos_model_pred
        x_denoised = lq_latent - model_pred
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

    def save_model(self, outf):
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k},
        }
        torch.save(sd, outf)
