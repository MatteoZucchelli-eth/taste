"""Step 2 automatic checks (CONTEXT §6g decision 9), shared by the trainer, the best-of-N baseline and the final eval.

For a set of images generated on the held-out EVAL prompts with fixed seeds:
  r_user        the optimised reward (CLIP + Phase C trunk + z_user)
  r_cons        the consensus reward (z = mu)
  siglip_user   held-out personal critic on SigLIP2 (never optimised): CLIP-specific reward hacking shows as
                r_user up while siglip_user is flat (Gao 2022; TDPO-R)
  siglip_cons   SigLIP2 consensus head (generic LAPIS appeal)
  prompt_fit    SigLIP2 image-text cosine with the prompt
  diversity     mean pairwise DINOv2 cosine distance across seeds within a prompt
  drift         DINOv2 cosine similarity to the base model's image for the same prompt and seed
  kl            path KL to the base model, sum over steps of ||mu_theta - mu_base||^2 / (2 sigma_t^2) (nats), on the
                policy's own eval trajectories (only for LoRA policies)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import rl_sd
from rl_critic import ClipEncoder, DinoEncoder, SiglipCritic, TasteReward, load_user
from rl_prompts import EVAL


class Evaluator:
    def __init__(self, seeds=4, seed_offset=900_000, prompts=None):
        u = load_user()
        self.clip = ClipEncoder()
        self.r_user = TasteReward(u["z_user"], self.clip)
        self.r_cons = TasteReward(u["z_cons"], self.clip)
        self.sig = SiglipCritic()
        self.dino = DinoEncoder()
        for m in (self.sig.enc.model, self.dino.model):
            m.to(torch.bfloat16)
        self.prompts = prompts or EVAL  # default: the held-out EVAL prompts
        self.jobs = [(i, p, seed_offset + 1000 * i + s) for i, p in enumerate(self.prompts) for s in range(seeds)]
        self.txt = self.sig.enc.text(self.prompts)
        self.base_dino = None

    @torch.no_grad()
    def score(self, imgs):
        """imgs [N, 3, H, W] in job order. Returns per-image DataFrame."""
        rows = []
        for k in range(0, len(imgs), 20):
            x = imgs[k:k + 20]
            e = self.clip(x)
            es = self.sig.enc.image(x)
            ed = self.dino(x)
            J = self.jobs[k:k + len(x)]
            fit = (es * self.txt[[j[0] for j in J]]).sum(1)
            for b, (i, p, s) in enumerate(J):
                rows.append(dict(prompt_idx=i, seed=s))
            rows_k = rows[-len(x):]
            for name, v in [("r_user", self.r_user(emb=e)), ("r_cons", self.r_cons(emb=e)),
                            ("siglip_user", self.sig(emb=es)), ("siglip_cons", self.sig.consensus(es)),
                            ("prompt_fit", fit)]:
                for b, val in enumerate(v.float().cpu().numpy()):
                    rows_k[b][name] = float(val)
            for b in range(len(x)):
                rows_k[b]["_dino"] = ed[b].float().cpu()
        D = pd.DataFrame(rows)
        dino = torch.stack(D.pop("_dino").tolist())
        if self.base_dino is not None:
            D["drift_sim"] = (dino * self.base_dino).sum(1).numpy()
        div = []
        for i in D.prompt_idx.unique():
            m = (D.prompt_idx == i).to_numpy()
            e = dino[m]
            c = e @ e.T
            n = len(e)
            div.append(float((1 - c[~torch.eye(n, dtype=bool)]).mean()))
        return D, float(np.mean(div)), dino

    def set_base(self, imgs):
        _, _, self.base_dino = self.score(imgs)

    @torch.no_grad()
    def generate(self, sd, S, guidance, with_kl=False, bs=10):
        """Images for the eval jobs under the current UNet (adapter state as set by the caller); optional path KL."""
        unet = sd["unet"]
        uncond = rl_sd.encode_prompts(sd, [""])
        out, kls = [], []
        for k in range(0, len(self.jobs), bs):
            J = self.jobs[k:k + bs]
            cond = rl_sd.encode_prompts(sd, [j[1] for j in J])
            un = uncond.expand(len(J), -1, -1)
            x, noise = rl_sd.seed_noise([j[2] for j in J], S.steps)
            kl = torch.zeros(len(J), device=x.device)
            for s_i, t in enumerate(S.t):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    eps = rl_sd.guided_eps(unet, x, t, cond, un, guidance)
                mu = S.mean(x, eps, s_i)
                if with_kl:
                    unet.disable_adapters()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        eb = rl_sd.guided_eps(unet, x, t, cond, un, guidance)
                    unet.enable_adapters()
                    kl += ((mu - S.mean(x, eb, s_i)) ** 2).sum((1, 2, 3)) / (2 * S.sigma[s_i] ** 2)
                x = mu + S.sigma[s_i] * noise[s_i]
            out.append(x)
            kls.append(kl)
        return rl_sd.decode(sd, torch.cat(out)), torch.cat(kls).cpu().numpy()

    def summary(self, D, div, kl=None):
        s = {c: float(D[c].mean()) for c in ["r_user", "r_cons", "siglip_user", "siglip_cons", "prompt_fit"]}
        if "drift_sim" in D:
            s["drift_sim"] = float(D["drift_sim"].mean())
        s["diversity"] = div
        if kl is not None:
            s["kl"] = float(np.mean(kl))
        return s


def save_eval_images(imgs, jobs, folder, size=512):
    folder = Path(folder)
    rl_sd.save_jpgs(imgs, [folder / f"p{i:02d}_s{s % 1000:02d}.jpg" for i, _, s in jobs], size=size)
