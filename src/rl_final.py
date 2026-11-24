"""Step 2, build steps 3 and 6 (CONTEXT §6g decisions 8-9): baselines and automatic checks on the held-out prompts.

Arms (same 10 EVAL prompts x --seeds seeds, same per-seed noise):
  base          SD 1.5, no adapter
  bon16         best-of-N from the base model: for each (prompt, seed) slot, N candidates (other seeds), keep the one
                with the highest *personal reward*. The distribution strong-KL RL approximates.
  <run>         every finished RL run under checkpoints/rl/runs/ (personal at each beta, consensus actor), final LoRA
Writes outputs/rl/final/{per_image.csv,summary.csv,summary.json} and outputs/rl/final/img/<arm>/p??_s??.jpg (512 px,
used by the app's blind eval mode). CIs: bootstrap over (prompt, seed) slots of the paired difference vs base.

Usage: python src/rl_final.py [--seeds 8 --n 16] [--runs personal_b0.001 ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_sd  # noqa: E402
from rl_eval import Evaluator, save_eval_images  # noqa: E402

OUT = Path("outputs/rl/final")
METRICS = ["r_user", "r_cons", "siglip_user", "siglip_cons", "prompt_fit"]


@torch.no_grad()
def best_of_n(sd, S, ev, guidance, n, bs=8):
    uncond = rl_sd.encode_prompts(sd, [""])
    best, best_r, cand_r = [], [], []
    for (i, p, s) in ev.jobs:
        cond = rl_sd.encode_prompts(sd, [p]).expand(bs, -1, -1)
        imgs, rs = [], []
        for k in range(0, n, bs):
            seeds = [s + 100_000 * (1 + c) for c in range(k, min(n, k + bs))]
            x0, noise = rl_sd.seed_noise(seeds, S.steps)
            m = len(seeds)
            lat = rl_sd.sample(sd, S, cond[:m], uncond.expand(m, -1, -1), guidance, latents=x0, noise=noise,
                               keep_traj=False)
            im = rl_sd.decode(sd, lat)
            imgs.append(im)
            rs.append(ev.r_user(im))
        imgs, rs = torch.cat(imgs), torch.cat(rs)
        k = int(rs.argmax())
        best.append(imgs[k:k + 1])
        best_r.append(float(rs[k]))
        cand_r.append(rs.cpu().numpy())
    return torch.cat(best), np.array(cand_r)


def boot_ci(d, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, (reps, len(d))).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--skip-bon", action="store_true")
    ap.add_argument("--arm", nargs="*", default=None, metavar="NAME=RUN/lora_eXXXX.pt",
                    help="explicit LoRA arms (checkpoint under checkpoints/rl/runs/); replaces --runs")
    ap.add_argument("--prompt-idx", type=int, nargs="*", default=None,
                    help="evaluate on these TRAIN prompts (e.g. a diagnostic's training prompts, unseen seeds) instead of EVAL")
    ap.add_argument("--seed-offset", type=int, default=900_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.arm:
        arms = dict(x.split("=", 1) for x in a.arm)
    else:
        runs = a.runs if a.runs is not None else sorted(p.name for p in Path("checkpoints/rl/runs").iterdir()
                                                        if (p / "lora_final.pt").exists())
        arms = {r: f"{r}/lora_final.pt" for r in runs}
    runs = list(arms)  # arm names
    run_of = {k: v.split("/")[0] for k, v in arms.items()}
    sd = rl_sd.load_sd()
    from rl_prompts import EVAL, TRAIN
    prompts = [TRAIN[i] for i in a.prompt_idx] if a.prompt_idx else EVAL
    ev = Evaluator(seeds=a.seeds, prompts=prompts, seed_offset=a.seed_offset)
    rows, summ = [], []

    def add(arm, imgs, kl=None, extra=None):
        D, div, _ = ev.score(imgs)
        D["arm"] = arm
        if kl is not None:
            D["kl"] = kl
        rows.append(D)
        s = dict(arm=arm, **ev.summary(D, div, kl), **(extra or {}))
        summ.append(s)
        save_eval_images(imgs, ev.jobs, OUT / "img" / arm)
        pd.concat(rows).to_csv(OUT / "per_image.csv", index=False)
        print(arm, {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}, flush=True)

    args_of = {}
    for r in runs:
        f = Path("outputs/rl/runs") / run_of[r] / "args.json"
        args_of[r] = json.loads(f.read_text())
    steps = {v["steps"] for v in args_of.values()} or {20}
    guid = {v["guidance"] for v in args_of.values()} or {5.0}
    assert len(steps) == 1 and len(guid) == 1, "all runs must share the sampler settings"
    steps, guidance = steps.pop(), guid.pop()
    S = rl_sd.Schedule(sd["sched"], steps)

    base, _ = ev.generate(sd, S, guidance)
    ev.set_base(base)
    add("base", base)
    if not a.skip_bon:
        bon, cand = best_of_n(sd, S, ev, guidance, a.n)
        np.save(OUT / "bon_candidate_rewards.npy", cand)
        add(f"bon{a.n}", bon)
    rl_sd.add_lora(sd["unet"], args_of[runs[0]]["rank"] if runs else 4)
    for r in runs:
        assert args_of[r]["rank"] == args_of[runs[0]]["rank"]
        st = torch.load(Path("checkpoints/rl/runs") / arms[r])
        missing = sd["unet"].load_state_dict(st, strict=False)
        assert not missing.unexpected_keys and all("lora" not in k for k in missing.missing_keys)
        imgs, kl = ev.generate(sd, S, guidance, with_kl=True)
        add(r, imgs, kl, dict(beta=args_of[r]["beta"], z=args_of[r]["z"]))

    D = pd.concat(rows)
    b = D[D.arm == "base"].set_index(["prompt_idx", "seed"])
    for s in summ:
        g = D[D.arm == s["arm"]].set_index(["prompt_idx", "seed"])
        for m in METRICS:
            d = (g[m] - b.loc[g.index, m]).to_numpy()
            s[f"d_{m}"] = float(d.mean())
            s[f"d_{m}_lo"], s[f"d_{m}_hi"] = boot_ci(d)
        s["win_r_user_vs_base"] = float((g["r_user"] > b.loc[g.index, "r_user"]).mean())
        s["win_siglip_user_vs_base"] = float((g["siglip_user"] > b.loc[g.index, "siglip_user"]).mean())
    # manifest for the app's blind eval mode (app/eval.html)
    man = dict(source=f"rl_final.py, {len(summ)} arms, {a.seeds} seeds", imgBase="/gen/", arms=[s["arm"] for s in summ],
               prompts=prompts, seeds=sorted({s % 1000 for _, _, s in ev.jobs}))
    Path("app/data/eval_manifest.json").write_text(json.dumps(man, indent=1))
    S_ = pd.DataFrame(summ)
    S_.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(S_[["arm", "r_user", "d_r_user", "d_siglip_user", "d_siglip_user_lo", "d_siglip_user_hi", "d_prompt_fit",
              "diversity"] + (["kl"] if "kl" in S_ else [])].round(4).to_string())


if __name__ == "__main__":
    main()
