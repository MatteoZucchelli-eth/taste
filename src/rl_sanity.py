"""Step 2, build step 2 (CONTEXT §6g): the reward on generated images, before any training.

1. Checks that rl_sd's DDIM eta=1 step equals diffusers' DDIMScheduler.step (same noise).
2. Samples N seeds x the 20 TRAIN prompts from base SD 1.5 (the training sampler: DDIM eta=1), measures throughput.
3. Scores every image with the user's reward (CLIP + Phase C trunk + z_user), the consensus reward (z = mu), and the
   held-out SigLIP critic; compares the reward axes on SD images with their LAPIS range (out-of-distribution check).
-> outputs/rl/sanity/{scores.csv,summary.json,img/*.jpg}; the page is made by rl_pages.py.

Usage: python src/rl_sanity.py [--seeds 10 --steps 20 --guidance 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_sd  # noqa: E402
from rl_critic import ClipEncoder, SiglipCritic, TasteReward, load_user  # noqa: E402
from rl_prompts import TRAIN  # noqa: E402

OUT = Path("outputs/rl/sanity")


def check_ddim(sd, steps):
    from diffusers import DDIMScheduler
    S = rl_sd.Schedule(sd["sched"], steps)
    ref = DDIMScheduler.from_pretrained(rl_sd.SD, subfolder="scheduler")
    ref.set_timesteps(steps, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    x, eps, z = (torch.randn(2, 4, 64, 64, device="cuda", generator=g) for _ in range(3))
    err = 0.0
    for k in [0, steps // 2, steps - 1]:
        t = S.t[k]
        mine = S.mean(x, eps, k) + S.sigma[k] * z
        theirs = ref.step(eps, t, x, eta=1.0, variance_noise=z).prev_sample
        err = max(err, float((mine - theirs).abs().max()))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    sd = rl_sd.load_sd()
    rep["ddim_maxabs_vs_diffusers"] = check_ddim(sd, a.steps)
    print("DDIM check max |diff|:", rep["ddim_maxabs_vs_diffusers"])
    S = rl_sd.Schedule(sd["sched"], a.steps)
    uncond = rl_sd.encode_prompts(sd, [""])

    jobs = [(p, i, s) for i, p in enumerate(TRAIN) for s in range(a.seeds)]
    lat_all = []
    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(0, len(jobs), a.batch):
        J = jobs[k:k + a.batch]
        cond = rl_sd.encode_prompts(sd, [j[0] for j in J])
        x0, noise = rl_sd.seed_noise([1000 * j[1] + j[2] for j in J], a.steps)
        lat_all.append(rl_sd.sample(sd, S, cond, uncond.expand(len(J), -1, -1), a.guidance, latents=x0, noise=noise,
                                    keep_traj=False))
    torch.cuda.synchronize()
    t_sample = time.time() - t0
    lat = torch.cat(lat_all)
    t0 = time.time()
    imgs = rl_sd.decode(sd, lat)
    torch.cuda.synchronize()
    t_dec = time.time() - t0
    rep["images"] = len(jobs)
    rep["sample_s_per_image"] = t_sample / len(jobs)
    rep["decode_s_per_image"] = t_dec / len(jobs)
    print(f"sampling {rep['sample_s_per_image']:.3f}s/img ({a.steps} steps, CFG, batch {a.batch}); "
          f"decode {rep['decode_s_per_image']:.3f}s/img")
    del sd
    torch.cuda.empty_cache()

    u = load_user()
    clip = ClipEncoder()
    r_user = TasteReward(u["z_user"], clip)
    r_cons = TasteReward(u["z_cons"], clip)
    sig = SiglipCritic()
    rows = []
    for k in range(0, len(imgs), 25):
        x = imgs[k:k + 25]
        e = clip(x)
        ax = r_user.axes(emb=e).cpu().numpy()
        es = sig.enc.image(x)
        ru, rc = r_user(emb=e).cpu().numpy(), r_cons(emb=e).cpu().numpy()
        su, sc = sig(emb=es).cpu().numpy(), sig.consensus(es).cpu().numpy()
        for b in range(len(x)):
            p, i, s = jobs[k + b]
            rows.append(dict(prompt=p, prompt_idx=i, seed=s, file=f"img/p{i:02d}_s{s:02d}.jpg", r_user=ru[b],
                             r_cons=rc[b], siglip_user=su[b], siglip_cons=sc[b], **{f"ax{j}": ax[b, j] for j in range(9)}))
    rl_sd.save_jpgs(imgs, [OUT / r["file"] for r in rows], size=256)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "scores.csv", index=False)

    U = torch.load(Path("checkpoints/phase_c/shared_K8_fold0.pt"))["U"].numpy()
    lap_u, lap_c = U @ u["z_user"], U @ u["z_cons"]
    rep["lapis_r_user_mean_sd"] = [float(lap_u.mean()), float(lap_u.std())]
    rep["sd_r_user_mean_sd"] = [float(D.r_user.mean()), float(D.r_user.std())]
    rep["lapis_r_cons_mean_sd"] = [float(lap_c.mean()), float(lap_c.std())]
    rep["sd_r_cons_mean_sd"] = [float(D.r_cons.mean()), float(D.r_cons.std())]
    rep["sd_r_user_lapis_percentile_median"] = float((lap_u[None, :] < D.r_user.to_numpy()[:, None]).mean(1).mean())
    rep["axis_z_of_sd_mean_vs_lapis"] = (((D[[f"ax{j}" for j in range(9)]].mean().to_numpy()) - U.mean(0))
                                         / U.std(0)).round(2).tolist()
    rep["axis_sd_ratio_sd_vs_lapis"] = (D[[f"ax{j}" for j in range(9)]].std().to_numpy() / U.std(0)).round(2).tolist()
    rep["corr_r_user_vs_siglip_user"] = float(D.r_user.corr(D.siglip_user, method="spearman"))
    rep["corr_r_cons_vs_siglip_cons"] = float(D.r_cons.corr(D.siglip_cons, method="spearman"))
    rep["corr_r_user_vs_r_cons"] = float(D.r_user.corr(D.r_cons, method="spearman"))
    # within-prompt: is the variation across seeds of the same prompt seen the same way by both critics?
    w = D.groupby("prompt_idx").apply(lambda g: g.r_user.corr(g.siglip_user, method="spearman"), include_groups=False)
    rep["within_prompt_corr_r_user_vs_siglip_user_median"] = float(w.median())
    rep["frac_reward_var_between_prompts"] = float(D.groupby("prompt_idx").r_user.mean().var() / D.r_user.var())
    print(json.dumps(rep, indent=1))
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
