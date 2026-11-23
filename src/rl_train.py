"""Step 2, build step 4-5 (CONTEXT §6g): KL-regularised online policy gradient for SD 1.5 + LoRA (DDPO + DPOK).

Objective (reward units, KL in nats):   max_theta  E[r(x)] - beta * KL(p_theta(path) || p_base(path))
  * MDP: DDIM eta=1, T steps, CFG inside the policy (rl_sd). Reward r = z . U(CLIP(x)) (rl_critic.TasteReward);
    --z user = the user's taste, --z cons = population mean mu (consensus actor, identical otherwise).
  * Policy gradient: DDPO-style PPO-clipped ratio per denoising step on log-probs averaged over latent dims, advantages
    = reward minus the mean of its prompt group (G images per prompt per epoch), divided by a fixed scale sigma_ref
    (within-prompt reward sd of the first epoch; a constant, so beta keeps its meaning), clipped to +-5.
  * KL: DPOK per-step Gaussian KL ||mu_theta - mu_base||^2 / (2 sigma_t^2), base = the same UNet with the LoRA
    disabled, differentiated through mu_theta at the visited states. It enters the loss as beta/sigma_ref times the
    per-dim mean, which matches the per-dim mean log-probs of the PG term, so the stationary point is the objective above.
  * Logged per epoch: reward stats, training KL (nats, from the trained steps), clip fraction, grad norm. Every
    --eval-every epochs: rl_eval.Evaluator on the held-out EVAL prompts (reward, held-out SigLIP critic, prompt fit,
    diversity, drift, path KL), and the LoRA weights are saved.

Usage: python src/rl_train.py --run personal_b0.1 --z user --beta 0.1 --hours 2
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
from rl_critic import TasteReward, load_user  # noqa: E402
from rl_eval import Evaluator, save_eval_images  # noqa: E402
from rl_prompts import TRAIN  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def lora_state(unet):
    return {k: v.detach().cpu().clone() for k, v in unet.state_dict().items() if "lora" in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--z", choices=["user", "cons"], default="user")
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--max-epochs", type=int, default=10_000)
    ap.add_argument("--prompts-per-epoch", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--micro", type=int, default=4, help="samples per backward pass")
    ap.add_argument("--opt-steps", type=int, default=2, help="optimizer steps per epoch")
    ap.add_argument("--ts-frac", type=float, default=0.5, help="fraction of denoising steps trained per sample")
    ap.add_argument("--clip", type=float, default=1e-4)
    ap.add_argument("--adv-clip", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--prompt-idx", type=int, nargs="*", default=None,
                    help="train on this subset of TRAIN only (diagnostic); default all 20")
    ap.add_argument("--eval-on-train", action="store_true",
                    help="evaluate on the training prompts (fixed held-out seeds) instead of the EVAL prompts")
    a = ap.parse_args()
    out = Path("outputs/rl/runs") / a.run
    ck = Path("checkpoints/rl/runs") / a.run
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    sd = rl_sd.load_sd()
    unet = sd["unet"]
    S = rl_sd.Schedule(sd["sched"], a.steps)
    u = load_user()
    train_prompts = [TRAIN[i] for i in a.prompt_idx] if a.prompt_idx else TRAIN
    ev = Evaluator(seeds=a.eval_seeds, prompts=train_prompts if a.eval_on_train else None)
    reward = TasteReward(u["z_user"] if a.z == "user" else u["z_cons"], ev.clip)

    # base eval images first (no adapter yet): reference for drift and epoch-0 metrics
    imgs, _ = ev.generate(sd, S, a.guidance)
    ev.set_base(imgs)
    params = rl_sd.add_lora(unet, a.rank)
    if not a.no_grad_ckpt:
        unet.enable_gradient_checkpointing()
    log(f"LoRA rank {a.rank}: {sum(p.numel() for p in params) / 1e6:.2f}M trainable params")
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)
    uncond_1 = rl_sd.encode_prompts(sd, [""])
    cond_all = rl_sd.encode_prompts(sd, train_prompts)

    hist, evals = [], []
    sigma_ref = None
    D_lat = 4 * 64 * 64

    def do_eval(epoch):
        imgs, kl = ev.generate(sd, S, a.guidance, with_kl=True)
        D, div, _ = ev.score(imgs)
        s = dict(epoch=epoch, t=time.time() - T0, **ev.summary(D, div, kl))
        evals.append(s)
        pd.DataFrame(evals).to_csv(out / "eval.csv", index=False)
        D.to_csv(out / f"eval_e{epoch:04d}.csv", index=False)
        save_eval_images(imgs, ev.jobs, out / f"img_e{epoch:04d}", size=256)
        torch.save(lora_state(unet), ck / f"lora_e{epoch:04d}.pt")
        log("EVAL " + " ".join(f"{k}={v:.4f}" for k, v in s.items() if k not in ("t",)))

    do_eval(0)  # adapter initialised with B=0 -> identical to base
    epoch = 0
    while epoch < a.max_epochs and (time.time() - T0) < a.hours * 3600:
        epoch += 1
        te = time.time()
        # ---------------- sample
        unet.eval()
        pidx = rng.choice(len(train_prompts), a.prompts_per_epoch, replace=False)
        pi = np.repeat(pidx, a.group)
        N = len(pi)
        trajs, lps, rews = [], [], []
        # sample in batches of exactly `micro`: bf16 kernels depend on batch size, and a mismatch between sampling and
        # training batch sizes makes the recomputed log-probs differ by ~2e-4 (> the 1e-4 PPO clip); equal sizes -> exact
        for k in range(0, N, a.micro):
            idx = torch.tensor(pi[k:k + a.micro], device="cuda")
            cond = cond_all[idx]
            un = uncond_1.expand(len(idx), -1, -1)
            x, traj, lp = rl_sd.sample(sd, S, cond, un, a.guidance)
            img = rl_sd.decode(sd, x)
            rews.append(reward(img))
            trajs.append(traj)
            lps.append(lp)
        traj, lp_old, r = torch.cat(trajs), torch.cat(lps), torch.cat(rews).float()
        t_sample = time.time() - te
        # ---------------- advantages (group = same prompt)
        rg = r.view(a.prompts_per_epoch, a.group)
        within = rg - rg.mean(1, keepdim=True)
        if sigma_ref is None:
            sigma_ref = float(within.std() + 1e-8)
            (out / "sigma_ref.json").write_text(json.dumps(dict(sigma_ref=sigma_ref)))
            log(f"sigma_ref (within-prompt reward sd at epoch 1) = {sigma_ref:.4f}")
        adv = (within / sigma_ref).clamp(-a.adv_clip, a.adv_clip).reshape(-1)
        # ---------------- train
        unet.train()
        n_ts = max(1, int(round(a.ts_frac * a.steps)))
        perm = torch.randperm(N, device="cuda")
        per_opt = N // a.opt_steps
        stats = dict(kl=[], clipfrac=[], pg=[], gnorm=[], ratio_dev=[])
        for o in range(a.opt_steps):
            opt.zero_grad(set_to_none=True)
            chunk = perm[o * per_opt:(o + 1) * per_opt]
            for m in range(0, len(chunk), a.micro):
                b = chunk[m:m + a.micro]
                cond = cond_all[torch.tensor(pi[b.cpu().numpy()], device="cuda")]
                un = uncond_1.expand(len(b), -1, -1)
                ts = torch.randperm(a.steps)[:n_ts].tolist()
                for k in ts:
                    x_t, x_n = traj[b, k], traj[b, k + 1]
                    t = S.t[k]
                    with torch.no_grad():
                        unet.disable_adapters()
                        eb = rl_sd.guided_eps(unet, x_t, t, cond, un, a.guidance)
                        unet.enable_adapters()
                        mu_b = S.mean(x_t, eb, k)
                    eps = rl_sd.guided_eps(unet, x_t, t, cond, un, a.guidance)
                    mu = S.mean(x_t, eps, k)
                    lp = rl_sd.gauss_logprob(x_n, mu, S.sigma[k])
                    kl_mean = ((mu - mu_b) ** 2).mean((1, 2, 3)) / (2 * S.sigma[k] ** 2)
                    ratio = torch.exp(lp - lp_old[b, k])
                    A = adv[b]
                    pg = torch.maximum(-A * ratio, -A * ratio.clamp(1 - a.clip, 1 + a.clip))
                    loss = (pg + (a.beta / sigma_ref) * kl_mean).mean() / (n_ts * (per_opt / a.micro))
                    loss.backward()
                    stats["kl"].append(kl_mean.detach() * D_lat)
                    stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().detach())
                    stats["ratio_dev"].append((ratio - 1).abs().mean().detach())
                    stats["pg"].append(pg.mean().detach())
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            stats["gnorm"].append(gn.detach())
            opt.step()
        # training KL per trajectory (nats), estimated from the trained steps
        kl_traj = float(torch.cat(stats["kl"]).mean()) * a.steps
        h = dict(epoch=epoch, t=time.time() - T0, r_mean=float(r.mean()), r_sd=float(r.std()),
                 r_within_sd=float(within.std()), kl_train=kl_traj,
                 clipfrac=float(torch.stack(stats["clipfrac"]).mean()), ratio_dev=float(torch.stack(stats["ratio_dev"]).mean()),
                 gnorm=float(torch.stack(stats["gnorm"]).mean()), t_sample=t_sample, t_epoch=time.time() - te)
        hist.append(h)
        pd.DataFrame(hist).to_csv(out / "train.csv", index=False)
        log(" ".join(f"{k}={v:.4g}" for k, v in h.items()))
        if epoch % a.eval_every == 0:
            do_eval(epoch)
    if not evals or evals[-1]["epoch"] != epoch:
        do_eval(epoch)
    torch.save(lora_state(unet), ck / "lora_final.pt")
    log("done")


if __name__ == "__main__":
    main()
