"""Step 2, build step 1 (CONTEXT §6g): everything the RL reward and the held-out checks need, verified on LAPIS.

  trunk   The Phase C checkpoint stores only U (axis scores of the 11,723 LAPIS images), not the MLP, so it cannot score
          generated images. Retrain the fold-0 shared model (same data, K=8, seed 0) to get a trunk, map its outputs
          onto the saved basis with a least-squares 9x9 affine map (the basis is only identified up to a linear map),
          then distil the composed trunk onto the saved U (MSE). Reports R^2 on 10% held-out images, and an end-to-end
          check: torch-side CLIP preprocessing of the raw images vs the cached clip.npz embeddings and saved U.
          -> checkpoints/rl/trunk.pt
  user    Refit the user's z in Python from the 100-pick session (MAP under N(mu, Sigma/tau) plus a side-bias term b,
          as in the app). tau = the validation choice for the 100-pair budget (outputs/phase_c/chosen_cfg.csv, fold 0,
          z-only), fixed before looking at the test pairs. b is dropped from the reward. Reports accuracy on the 20
          locked test pairs. -> checkpoints/rl/user.pt
  siglip  Held-out critic for reward-hacking checks: frozen SigLIP2 + Phase B consensus head (z-scored over LAPIS)
          + per-user linear residual with lambda by inner CV over images (phase_b_critics.fit_with_cv), fitted on the
          100 learning pairs, tested on the 20 test pairs. -> checkpoints/rl/siglip_critic.pt

Usage: python src/rl_setup.py --stage all [--session outputs/user_taste/<id>.json]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from phase_b_critics import fit_with_cv, load_emb, load_ratings  # noqa: E402
from phase_c_lowrank import setup, shared_train_set, train_shared  # noqa: E402
from rl_critic import CK, DEV, ClipEncoder, make_siglip_head, make_trunk  # noqa: E402

SESSION = Path("outputs/user_taste/2026-11-23T20-58-31-096Z_m27khg.json")
OUT = Path("outputs/rl/setup")


def r2(pred, target):
    return 1 - ((pred - target) ** 2).sum(0) / ((target - target.mean(0)) ** 2).sum(0)


# ---------------------------------------------------------------- trunk
def distil(trunk, E, U, idx, steps=4000, lr=3e-4):
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    trunk.train()
    g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(steps):
        b = idx[torch.randint(len(idx), (2048,), device=DEV, generator=g)]
        loss = F.mse_loss(trunk[1:](E[b]), U[b])  # no input dropout while distilling a fixed function
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    trunk.eval()
    return trunk


def stage_trunk(a, rep):
    _, d = load_ratings(a.data)
    image_ids = np.sort(d["image_id"].unique())
    E, pos = load_emb(a.emb, "clip", image_ids)
    ids = np.load(a.emb / "clip.npz")["ids"]
    assert (ids == image_ids).all()
    ctx = setup(SimpleNamespace(phase_b=Path("outputs/phase_b"), seed=0, folds=5, n_val=40, n_val_imgs=1000), d)
    S, sids, rd, fold_of, vids, vd, vtest, v_test_imgs = ctx
    d_tr = shared_train_set(d, sids, vids, rd, fold_of, 0, v_test_imgs)
    torch.backends.cuda.matmul.allow_tf32 = True
    U_new, _, _, mon, model = train_shared(E, d_tr, pos, 8, 6000, 0, return_model=True)
    U_sav = torch.load(Path("checkpoints/phase_c/shared_K8_fold0.pt"))["U"].to(DEV).double()
    print(f"retrained fold-0 shared model, monitor acc {mon:.4f}")

    X = torch.cat([U_new, torch.ones(len(U_new), 1, device=DEV, dtype=torch.float64)], 1)
    A = torch.linalg.lstsq(X, U_sav).solution  # (10, 9)
    rep["retrain_monitor_acc"] = mon
    rep["r2_aligned_per_axis"] = r2(X @ A, U_sav).cpu().numpy().round(4).tolist()
    print("R^2 retrained+affine vs saved U, per axis:", rep["r2_aligned_per_axis"])

    trunk = make_trunk().to(DEV)
    trunk.load_state_dict(model.trunk.state_dict())
    with torch.no_grad():  # fold the affine map into the last layer: W' = A^T W, b' = A^T b + c
        W, b = trunk[4].weight.double(), trunk[4].bias.double()
        trunk[4].weight.copy_((A[:9].T @ W).float())
        trunk[4].bias.copy_((A[:9].T @ b + A[9]).float())
    Et = torch.tensor(E, device=DEV)
    Uf = U_sav.float()
    with torch.no_grad():
        rep["r2_folded_per_axis"] = r2(trunk.eval()(Et), Uf).cpu().numpy().round(4).tolist()
    state0 = {k: v.clone() for k, v in trunk.state_dict().items()}

    # held-out check of the distillation, then distil on all images
    perm = torch.randperm(len(Et), generator=torch.Generator().manual_seed(0)).to(DEV)
    ho, tr = perm[: len(perm) // 10], perm[len(perm) // 10:]
    distil(trunk, Et, Uf, tr)
    with torch.no_grad():
        P = trunk(Et)
    rep["r2_distil_heldout_per_axis"] = r2(P[ho], Uf[ho]).cpu().numpy().round(4).tolist()
    rep["r2_distil_train_per_axis"] = r2(P[tr], Uf[tr]).cpu().numpy().round(4).tolist()
    print("distil (90% images) R^2 held-out:", rep["r2_distil_heldout_per_axis"])
    trunk.load_state_dict(state0)
    distil(trunk, Et, Uf, perm)
    with torch.no_grad():
        P = trunk(Et)
    rep["r2_final_all_per_axis"] = r2(P, Uf).cpu().numpy().round(5).tolist()
    mu = torch.load(Path("checkpoints/phase_c/shared_K8_fold0.pt"))["Z"].double().mean(0).float().to(DEV)
    rep["cons_score_corr_final"] = float(np.corrcoef((P @ mu).cpu(), (Uf @ mu).cpu())[0, 1])
    print("final R^2 (all):", rep["r2_final_all_per_axis"], "consensus-score corr", rep["cons_score_corr_final"])
    CK.mkdir(parents=True, exist_ok=True)
    torch.save(trunk.state_dict(), CK / "trunk.pt")

    # end to end: raw LAPIS images -> torch-side CLIP preprocessing -> trunk, vs cached embeddings / saved U
    paths = pd.read_csv(a.data / "piaa/piaa.csv", usecols=["image_id", "path"]).drop_duplicates("image_id") \
        .set_index("image_id")["path"]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ids), 256, replace=False)
    clip = ClipEncoder()
    embs = []
    for k in range(0, len(sub), 32):
        ims = []
        for i in sub[k:k + 32]:
            im = Image.open(a.data / "images" / paths[ids[i]]).convert("RGB")
            # SD images are square: compare on a center square crop, encoded both ways
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((512, 512), Image.BICUBIC)
            ims.append(im)
        x = torch.stack([torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255 for im in ims]).to(DEV)
        e_t = clip(x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e_p = F.normalize(clip.model.encode_image(torch.stack([clip.prep(im) for im in ims]).to(DEV)).float(), dim=-1)
        embs.append((e_t, e_p))
    e_t = torch.cat([e[0] for e in embs])
    e_p = torch.cat([e[1] for e in embs])
    e_c = Et[torch.tensor(sub, device=DEV)]
    with torch.no_grad():
        s_t, s_p, s_c = trunk(e_t) @ mu, trunk(e_p) @ mu, Uf[torch.tensor(sub, device=DEV)] @ mu
    rep["e2e_cos_torchprep_vs_pilprep"] = float((e_t * e_p).sum(1).mean())
    rep["e2e_cos_torchprep_vs_cached_fullimage"] = float((e_t * e_c).sum(1).mean())
    rep["e2e_cons_corr_torchprep_vs_pilprep"] = float(np.corrcoef(s_t.cpu(), s_p.cpu())[0, 1])
    rep["e2e_cons_corr_torchprep_vs_saved_fullimage"] = float(np.corrcoef(s_t.cpu(), s_c.cpu())[0, 1])
    print({k: round(v, 4) for k, v in rep.items() if k.startswith("e2e")})


# ---------------------------------------------------------------- user z
def fit_side(Ud, y, mu, Sinv, tau, side_sd=0.5, iters=50):
    """Newton MAP of theta = [z, b] (same model as app.js fit). Ud: (n, 9) = U[top] - U[bottom], y: 1 if top chosen."""
    K = Ud.shape[1]
    X = np.hstack([Ud, np.ones((len(Ud), 1))])
    m = np.concatenate([mu, [0.0]])
    P = np.zeros((K + 1, K + 1))
    P[:K, :K] = tau * Sinv
    P[K, K] = 1 / side_sd ** 2
    th = m.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ th))
        g = P @ (th - m) + X.T @ (p - y)
        H = P + (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        th -= step
        if np.abs(step).max() < 1e-10:
            break
    return th[:K], th[K], H


def session_arrays(sess, pos):
    def arr(part):
        top = pos[[q["top"] for q in sess[part]]].to_numpy()
        bot = pos[[q["bottom"] for q in sess[part]]].to_numpy()
        # learning pairs: y = 1 if top chosen; test pairs: choice = 0 if top chosen
        y = [q["y"] if "y" in q else 1 - q["choice"] for q in sess[part]]
        return top, bot, np.array(y, float)
    return arr("train"), arr("test")


def stage_user(a, rep):
    sess = json.loads(a.session.read_text())
    assert not sess["partial"] and len(sess["train"]) == 100 and len(sess["test"]) == 20
    ids = np.load(a.emb / "clip.npz")["ids"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    ck = torch.load(Path("checkpoints/phase_c/shared_K8_fold0.pt"))
    U, Z = ck["U"].numpy().astype(np.float64), ck["Z"].numpy().astype(np.float64)
    mu = Z.mean(0)
    Sig = np.cov(Z.T) + 1e-4 * np.eye(9)
    Sinv = np.linalg.inv(Sig)
    (ti, bi, ty), (si, sb, sy) = session_arrays(sess, pos)
    cfg = pd.read_csv("outputs/phase_c/chosen_cfg.csv")
    tau100 = float(cfg[(cfg.fold == 0) & (cfg.budget == 100)]["zonly_tau"].iloc[0])

    def test_acc(z, b=0.0):
        return float((((U[si] - U[sb]) @ z + b > 0) == (sy == 1)).mean())

    res = {}
    for tau in sorted({0.5, tau100}):
        z, b, H = fit_side(U[ti] - U[bi], ty, mu, Sinv, tau)
        res[tau] = dict(z=z, b=b, H=H, test_acc=test_acc(z), test_acc_with_b=test_acc(z, b))
    z_app = np.array(sess["z"])
    rep["app_z_vs_python_tau0.5_maxabs"] = float(np.abs(res[0.5]["z"] - z_app).max())
    rep["tau_chosen"] = tau100
    rep["test_acc"] = {f"tau{t}": r["test_acc"] for t, r in res.items()}
    rep["test_acc_with_side_bias"] = {f"tau{t}": r["test_acc_with_b"] for t, r in res.items()}
    rep["test_acc_crowd_mu"] = test_acc(mu)
    zu = res[tau100]["z"]
    rep["z_user"] = zu.round(4).tolist()
    rep["side_bias"] = float(res[tau100]["b"])
    rep["z_minus_mu_in_prior_sd"] = ((zu - mu) / np.sqrt(np.diag(Sig))).round(2).tolist()
    s_u, s_c = U @ zu, U @ mu
    rep["corr_user_vs_consensus_score_lapis"] = float(np.corrcoef(s_u, s_c)[0, 1])
    rep["sd_user_score_lapis"], rep["sd_cons_score_lapis"] = float(s_u.std()), float(s_c.std())
    print(json.dumps({k: rep[k] for k in rep if k.startswith(("app_", "tau", "test", "z_", "side", "corr", "sd_"))},
                     indent=1))
    CK.mkdir(parents=True, exist_ok=True)
    torch.save(dict(z_user=zu, z_cons=mu, tau=tau100, side_bias=res[tau100]["b"], Sig=Sig, Sinv=Sinv,
                    session=str(a.session)), CK / "user.pt")


# ---------------------------------------------------------------- SigLIP held-out critic
def stage_siglip(a, rep):
    sess = json.loads(a.session.read_text())
    z = np.load(a.emb / "siglip2.npz")
    ids, E = z["ids"], z["emb"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    head = make_siglip_head().to(DEV)
    head.load_state_dict(torch.load(Path("checkpoints/phase_b/consensus_siglip2_mlp_giaa.pt")))
    head.eval()
    Et = torch.tensor(E, device=DEV)
    with torch.no_grad():
        h = head(Et).squeeze(-1)
    m, s = float(h.mean()), float(h.std())
    (ti, bi, ty), (si, sb, sy) = session_arrays(sess, pos)
    loc = np.unique(np.concatenate([ti, bi]))
    li = pd.Series(np.arange(len(loc)), index=loc)
    El = Et[torch.tensor(loc, device=DEV)].double()
    sc = ((h - m) / s)[torch.tensor(loc, device=DEV)].double()
    pi = torch.tensor(li[ti].to_numpy(), device=DEV)
    pj = torch.tensor(li[bi].to_numpy(), device=DEV)
    t = torch.tensor(ty, device=DEV, dtype=torch.float64)
    theta, lam = fit_with_cv(sc, El, pi, pj, t, torch.Generator(device=DEV).manual_seed(0))
    full = (theta[0] * (h - m) / s).double() + Et.double() @ theta[1:]
    cons = (h - m) / s
    fs, cs = full.cpu().numpy(), cons.cpu().numpy()
    rep["siglip_lambda"] = lam
    rep["siglip_beta"] = float(theta[0])
    rep["siglip_test_acc_personal"] = float(((fs[si] - fs[sb] > 0) == (sy == 1)).mean())
    rep["siglip_test_acc_consensus"] = float(((cs[si] - cs[sb] > 0) == (sy == 1)).mean())
    rep["siglip_train_acc_personal"] = float(((fs[ti] - fs[bi] > 0) == (ty == 1)).mean())
    # agreement with the CLIP reward over all LAPIS images (same ids order in both npz files)
    u = torch.load(CK / "user.pt", weights_only=False)
    U = torch.load(Path("checkpoints/phase_c/shared_K8_fold0.pt"))["U"].numpy()
    assert (np.load(a.emb / "clip.npz")["ids"] == ids).all()
    rep["lapis_corr_siglip_personal_vs_clip_reward"] = float(np.corrcoef(fs, U @ u["z_user"])[0, 1])
    rep["lapis_corr_siglip_consensus_vs_clip_consensus"] = float(np.corrcoef(cs, U @ u["z_cons"])[0, 1])
    print({k: v for k, v in rep.items() if k.startswith(("siglip", "lapis"))})
    torch.save(dict(head=head.state_dict(), head_mean=m, head_std=s, theta=theta.cpu(), lam=lam), CK / "siglip_critic.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["trunk", "user", "siglip", "all"], default="all")
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--emb", type=Path, default=Path("embeddings"))
    ap.add_argument("--session", type=Path, default=SESSION)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "setup.json"
    rep = json.loads(f.read_text()) if f.exists() else {}
    for st, fn in [("trunk", stage_trunk), ("user", stage_user), ("siglip", stage_siglip)]:
        if a.stage in (st, "all"):
            fn(a, rep)
            f.write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
