"""Step 2: Stable Diffusion 1.5 as an MDP (DDPO, Black 2023; DPOK, Fan 2023).

Each denoising step of DDIM with eta=1 is a Gaussian x_{t-1} ~ N(mu_theta(x_t, t, c), sigma_t^2 I) with sigma_t
fixed by the schedule, so log-probs of the sampled actions are exact. Classifier-free guidance is part of the policy
(the guided eps defines mu). With a LoRA on the UNet, the base policy is the same UNet with the adapter disabled.

Per-step KL between policy and base (DPOK): KL_t = ||mu_theta - mu_base||^2 / (2 sigma_t^2) (summed over latent dims).
"""
import math
from pathlib import Path

import torch

DEV = "cuda"
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_sd(dtype=torch.bfloat16):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    kw = dict(variant="fp16", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", **kw).to(DEV)
    vae = AutoencoderKL.from_pretrained(SD, subfolder="vae", **kw).to(DEV)
    te = CLIPTextModel.from_pretrained(SD, subfolder="text_encoder", **kw).to(DEV)
    tok = CLIPTokenizer.from_pretrained(SD, subfolder="tokenizer")
    sched = DDIMScheduler.from_pretrained(SD, subfolder="scheduler")
    for m in (unet, vae, te):
        m.requires_grad_(False).eval()
    return dict(unet=unet, vae=vae, te=te, tok=tok, sched=sched)


@torch.no_grad()
def encode_prompts(sd, prompts):
    ids = sd["tok"](prompts, padding="max_length", max_length=sd["tok"].model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(DEV)
    return sd["te"](ids)[0]


def add_lora(unet, rank=4, alpha=None):
    from peft import LoraConfig
    cfg = LoraConfig(r=rank, lora_alpha=alpha or rank, init_lora_weights="gaussian",
                     target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet.add_adapter(cfg)
    params = [p for n, p in unet.named_parameters() if "lora" in n]
    for p in params:
        p.data = p.data.float()  # fp32 master weights for the adapter; the frozen base stays bf16
        p.requires_grad_(True)
    return params


class Schedule:
    """DDIM eta=1 coefficients for a fixed number of steps (DDPM-equivalent variance)."""

    def __init__(self, sched, steps, eta=1.0):
        sched.set_timesteps(steps, device=DEV)
        self.t = sched.timesteps.clone()  # descending
        ac = sched.alphas_cumprod.to(DEV, torch.float32)
        ratio = sched.config.num_train_timesteps // steps
        prev = self.t - ratio
        self.a = ac[self.t]
        self.a_prev = torch.where(prev >= 0, ac[prev.clamp(min=0)], torch.as_tensor(sched.final_alpha_cumprod, device=DEV, dtype=torch.float32))
        var = (1 - self.a_prev) / (1 - self.a) * (1 - self.a / self.a_prev)
        self.sigma = eta * var.sqrt()
        self.steps = steps

    def mean(self, x, eps, k):
        """mu(x_t, eps) for step index k (x, eps: float32)."""
        a, ap, s = self.a[k], self.a_prev[k], self.sigma[k]
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        return ap.sqrt() * x0 + (1 - ap - s ** 2).clamp(min=0).sqrt() * eps


def gauss_logprob(x, mu, sigma):
    """log N(x; mu, sigma^2 I), averaged over latent dims (DDPO convention, keeps ratios well-scaled)."""
    return (-((x - mu) ** 2) / (2 * sigma ** 2) - math.log(sigma) - 0.5 * math.log(2 * math.pi)).mean(dim=(1, 2, 3))


def guided_eps(unet, x, t, cond, uncond, guidance):
    xin = torch.cat([x, x]).to(unet.dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16):  # LoRA weights are fp32, the base is bf16
        e = unet(xin, t, encoder_hidden_states=torch.cat([uncond, cond])).sample.float()
    eu, ec = e.chunk(2)
    return eu + guidance * (ec - eu)


@torch.no_grad()
def sample(sd, S, cond, uncond, guidance, generator=None, latents=None, keep_traj=True, noise=None):
    """Run the stochastic DDIM chain. Returns final latents and, if keep_traj, (x_t for each step [B, T+1, ...],
    log-probs [B, T]). `noise` (optional, [T, B, 4, 64, 64]) fixes the per-step noise (same seed -> same image)."""
    unet = sd["unet"]
    B = cond.shape[0]
    x = latents if latents is not None else torch.randn(B, 4, 64, 64, device=DEV, generator=generator)
    traj, lps = [x], []
    for k, t in enumerate(S.t):
        eps = guided_eps(unet, x, t, cond, uncond, guidance)
        mu = S.mean(x, eps, k)
        z = noise[k] if noise is not None else torch.randn(x.shape, device=DEV, generator=generator)
        xn = mu + S.sigma[k] * z
        if keep_traj:
            lps.append(gauss_logprob(xn, mu, S.sigma[k]))
            traj.append(xn)
        x = xn
    if keep_traj:
        return x, torch.stack(traj, 1), torch.stack(lps, 1)
    return x


@torch.no_grad()
def decode(sd, lat, chunk=8):
    vae = sd["vae"]
    out = []
    for k in range(0, len(lat), chunk):
        im = vae.decode((lat[k:k + chunk] / vae.config.scaling_factor).to(vae.dtype)).sample.float()
        out.append(((im + 1) / 2).clamp(0, 1))
    return torch.cat(out)


def save_jpgs(x, paths, size=None, quality=88):
    from PIL import Image
    arr = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    for a, p in zip(arr, paths):
        im = Image.fromarray(a)
        if size:
            im = im.resize((size, size), Image.LANCZOS)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        im.save(p, quality=quality)


def seed_noise(seeds, steps):
    """Per-image initial latents and per-step noise from integer seeds, so an image is fully determined by
    (prompt, seed, policy): the same seed gives comparable images across arms (base, tuned, ...)."""
    lat, noi = [], []
    for s in seeds:
        g = torch.Generator(device=DEV).manual_seed(int(s))
        lat.append(torch.randn(4, 64, 64, device=DEV, generator=g))
        noi.append(torch.randn(steps, 4, 64, 64, device=DEV, generator=g))
    return torch.stack(lat), torch.stack(noi, 1)
