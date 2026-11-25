# Project context (handoff)

Read this first, then `papers/INDEX.md`. Current work: **broadening beyond LAPIS to other visual datasets (§7)**. RL (§6) and text are parked.

## 1. Goal

Give a generative model **taste**: first learn and verify a **pairwise critic** of a *specific person's* taste, then use it as the
reward for a generator (end goal: design generation, so images).
- Model a **few individuals**, not a consensus (averaging erases taste; Artificial Hivemind 2025).
- **Verify the critic before training any generator.**
- Backed by top papers; keep the literature list **lean** (`papers/INDEX.md`).
- Origin: Feng 2025 (creative chess puzzles). Their hand-designed scorer did not track liking and got reward-hacked, so here the
  critic is learned from people. Chess was dropped (no per-expert data). PARA (CVPR 2022, has per-rater repeats) is a fallback; it needs registration.

## 2. Dataset: LAPIS

- HF `asmaerten/LAPIS` (ungated, **CC BY-NC 4.0**), arXiv 2504.07670. 281k ratings, **0–100 scale**, 11,723 artworks
  (`figurative/`, `abstract/`), ~24 ratings/image, **555** raters (~500 each), with demographics and art interest (VAIAK).
- Files in `data/lapis/` (HF layout): `piaa/piaa.csv` (one row per rating; read with `keep_default_na=False, na_values=[""]`),
  `giaa/{train,validation,test}.csv` (official split), `image_attributes/{image_attributes,image_prompts}.csv`, `images/`.
- No planned repeats, but 63 raters rated some images twice (614 pairs: test–retest r=0.72) → ceilings come mostly from consensus/split-half.

## 3. Success criterion (pre-registered)

For rater A on A's held-out pairs: (1) personal critic beats a consensus critic; (2) A's critic beats other raters' critics on A;
(3) compare with A's ceiling; (4) the critic's top-ranked items are good for A (best-of-n, Gao 2022).
Metrics: within-rater pairwise accuracy (ties dropped; headline = pairs with score gap ≥10), per rater, with CIs.

## 4. Step 1 results: the critic works

Rules (user): **pairwise only, everywhere** (people will only answer "this or that"; LAPIS scores only build within-rater pairs).

- **Phase A, is there personal taste?** (`src/phase_a.py`, `src/phase_a_checks.py` → `outputs/phase_a/`). Metadata only, no embeddings.
  Baseline = OLS of A's rating on the leave-one-out consensus + a cross-fitted generic metadata prediction. Taste = the residual is
  predictable from metadata out of sample (ridge λ=1000, 20 half-splits, 1000-permutation null, BH-FDR 5%). Validated with synthetic raters
  (no taste: 0% FDR; planted taste found) and pseudo-raters (0%).
  Result: **530/553 raters show significant consistent taste**; held-out gain median +2.7 pts (a lower bound). Unrelated to art interest.
  Caveat: the biggest gains are raters at chance with consensus who use coarse heuristics.
- **Phase B, critics** (`src/phase_b_embed.py`, `src/phase_b_critics.py`, `src/phase_b_page.py` → `outputs/phase_b/`, `embeddings/`).
  Frozen **CLIP ViT-L/14** + MLP Bradley–Terry consensus head (SigLIP2/DINOv2 tie). Personal = consensus + per-rater linear residual (Ren 2017).
  18 raters (8 selected for diverse taste + 10 random controls); consensus retrained without them.
  **All 18 pass criteria 1, 2, 4.** Personal vs consensus: controls median +6.9 pts, selected +15.7. Own critic ranks 1st of 18 for all 18.
  Most of the gain arrives by 250–500 pairs.
- **Phase C, shared low-rank taste model** (`src/phase_c_lowrank.py` → `outputs/phase_c/`, `checkpoints/phase_c/shared_K8_fold*.pt`).
  `s_u(x) = c(x) + z_u·V(f(x)) + w_u·f(x)`: shared K-axis basis on ~500 raters, a person = K weights z (MAP under the population prior, scale τ)
  plus a residual w with a strong prior. **K=8** is best (LAPIS taste ≈ 8-dimensional on CLIP). Needs **~3–5× fewer pairs** than Phase B
  (gap≥10, controls: 0.744 at 25 pairs, 0.769 at 50); no gain at full data.
- **Encoder LoRA: tried and dropped** (+0.2 pts, then overfit; ~10k images too few). Do not retry without much more data.
- **Active learning** (`src/active.py` → `outputs/active/`): **Phase C + infogain** (greedy D-optimal on the Laplace posterior) is best at every
  budget; uncertainty sampling hurts. Phase C + infogain at 25 pairs ≈ Phase B + random at ~150–200 pairs.
- Axis pages: `src/axes_page.py` → `outputs/phase_c/axes.html`. Axis pole names: 0 crowd-pleaser/against the grain, 1 vivid sensual painting/quiet
  minimalism, 2 raw gestural/polished detail, 3 old-master portraits/the human body, 4 open landscapes/intimate figures, 5 baroque drama/clean
  colour fields, 6 busy geometry/near-empty canvases, 7 playful naive figures/explosive abstraction, 8 lush florals/stark B&W.
- Parked: per-rater ceiling estimate. PAM∃LA (Maerten 2026, LAPIS authors) is the closest prior work, but its protocol is not comparable.

**The user's own taste** (`outputs/user_taste/2026-11-23T20-58-31-096Z_m27khg.json`, tag `me`): 100 learning pairs (random/infogain mix) +
20 locked test pairs. **The personal critic predicted 16/20 test picks (p=0.006 vs chance); the crowd model 9/20.** Traits: against the grain,
stark B&W, raw gestural marks, baroque drama.

## 5. "This or That" web app (`app/`, export `src/app_export.py`)

Phone-first static app: train picks (5 random, then infogain or `mix=1` alternating) fit a person's z (9 numbers) on the frozen Phase C model
(fold 0, K=8, τ=0.5), then 20 test pairs whose model guess is locked before display: 5 most-sure, 5 "contrarian" (personal model
sure, crowd model picks the other image), 10 random, at random positions (URL `sure`, `contra`). Chosen by simulation on 58 held-out
LAPIS raters (`src/sure_test.py` → `outputs/sure_test/`): mean 16.5/20, beats the crowd model in 93% of sessions (random pairs:
14.6/20, 61%). The app does not tell players that pairs are selected, and shows no per-pair confidence. Results compare the player
with other people (per-trait percentiles vs `app/data/ref.json`: z fitted from 50 app-style picks for those 58 raters,
`src/app_reference.py`) and say how rare their mix is: "1 in N people share your mix of <top trait> and <2nd trait>" (Gaussian
tail of both top traits on the ref crowd; ref range 1 in 11 to >1,000, median ~120; replaced the "more mainstream than X%" line on
2026-11-25 so nobody is told they are average); the share card carries score, taste label, rarity line and 3 picks. Shows score vs crowd model, top traits, share card,
challenge link (`#c=`). Tap to choose, images stacked, sides random, per-person side-bias term in the fit, taps <450 ms ignored. All images allowed.
- URL params: `train`, `test`, `mix=1`, `tag`. Sessions are POSTed to `/api/session` → `outputs/user_taste/`.
- Local: `bash app/start.sh [port]` (wraps `app/serve.py`; prints LAN address).
- Deployed: https://this-or-that-six.vercel.app (Vercel project `this-or-that`, repo MatteoZucchelli-eth/taste-website, built from `app/` by
  `app/vercel/build_site.py` into `../taste-site`; defaults 50 train / 20 test). Sessions are stored anonymously in the private Blob store
  `taste-sessions`; pull them with `taste-site/scripts/pull-sessions.mjs`.
- The fit uses damped Newton with backtracking (plain Newton diverged on real sessions; fixed 2026-11-24). JS matches Python `fit_users`.
  τ check on real players (`src/tau_check.py`): flat for 0.25–2; re-run at ~50 sessions.

## 6. Step 2: RL of a diffusion actor (PARKED 2026-11-24, not working as set up)

SD 1.5 + LoRA, DDPO/DPOK-style policy gradient (per-step KL to the base), reward r(x) = z·U(f(x)) on CLIP with the user's z.
Code `src/rl_*.py`, outputs `outputs/rl/`, literature `papers/INDEX.md` §6. Why parked:
- Only the 4-prompt diagnostic (`diag4_b0.01`) raised the reward; the 20-prompt run stayed flat (KL ~480 nats).
- Reward hacking: on the diagnostic prompts, epoch 24 raised the CLIP reward by +0.37 over base, but the held-out SigLIP critic rose only +0.05.
  Within-prompt CLIP reward vs SigLIP critic Spearman was only 0.16.
- Blind test by the user (`outputs/rl/human_eval/`, 90 answers): epoch 24 BT +0.64 [−0.10, 1.53] vs base; best-of-16 and epoch 6 ≤ base. All n.s.
- Untried if resumed: lower lr with β 0.1/0.03/0.01, choosing β on the SigLIP critic. Gotcha: sampling and training must use the same
  micro-batch (bf16 log-probs depend on batch size).

Side check: the user's LAPIS-trained critic transfers to other images (`src/design_transfer.py`, 100 locked pairs each): Crello designs
personal 58% / crowd 48% (SigLIP personal 61%, p=0.02); products 63% / 54% (p=0.006).

## 7. Broadening beyond LAPIS (user decisions, 2026-11-24)

**Text PARKED (user, 2026-11-24): visual only for now.** Unfinished: `src/phase_b_text.py` (LiteraryTaste, bge-large; the
held-out-set consensus critic reached 0.626 vs 0.58 for the other 2 votes; personal critics never finished).

Same taste mechanism over a broader space: images **and** English texts, later one taste vector over both. Each new dataset must first
pass a Phase A replication.
- **User (2026-11-24): real, human-made designs only, no AI-generated images** (PAMELA, TASTE, DesignPref are out as targets).
- Image sets: **PAMELA** (`bethgelab/PAMELA`, CC BY 4.0, arXiv 2604.07427: 5,077 AI images with English prompts, 69.9k ratings,
  199 raters, ~15/image, 1–5 slider, `participant_id` + demographics), **DesignPref** (arXiv 2511.20513, 12k pairwise UI designs,
  20 pro designers; license unchecked), **TASTE** (arXiv 2605.20731, AI graphic design, 2×5 designers).
  Pick-a-Pic v2 dropped for now (no human consensus; users judge their own prompts); revisit only if PAMELA passes.
- Text sets: **LiteraryTaste** (arXiv 2511.09310, 60 people × 100 pairs), **PRISM**, poetry studies (Chaudhuri 2024
  129×36 poems, OSF 9mw7r; brief-poetry EEG set 51×210). Still to verify: PRISM details, per-row data for the poetry sets.
- Texts are items in their own right (pick between two texts), not prompts.
- Everything becomes **within-rater pairs** (ties dropped).
- Test features **as in Phase A** (metadata only). For text: **hand-built text stats** (length, readability, lexical diversity,
  rhyme/meter, sentiment, topic tags) + dataset labels.
- Gate: **Phase A pipeline**. **Pre-registered pass rule (2026-11-24): ≥50% of raters significant at BH-FDR 5% AND median held-out
  gain ≥2 pts.** PAMELA features: one-hot group/style/type + simple pixel stats we compute (no neural models).
- Cross-modal link: **collect in the app**: text-vs-text pairs **interleaved** with image pairs (never image vs text), same players.

**Results**
- **PAMELA: PASS** (`src/phase_a_pamela.py` → `outputs/phase_a_pamela/`, pixel stats cached in `data/pamela/image_stats.csv`).
  174/198 raters (88%) sig at FDR 5%; median held-out gain **+2.05 pts** (just over the bar; LAPIS +2.7). Consensus reliability 0.80.
  Pseudo-raters 0% sig; synthetic no-taste 0%. The test is conservative here (planted 6-pt taste found in only 18% of synthetic raters,
  ~200 ratings/rater). Significance rises with n (≤100 ratings: 60%; >400: 100%).
- **LiteraryTaste (text): FAIL the gate** (`src/phase_a_text.py` → `outputs/phase_a_text/`; data `data/literarytaste/`, from GitHub
  mj-storytelling/LiteraryTaste, no license file). 20 sets × 3 raters × 100 pairs, each text in one pair only; consensus = the other 2
  votes; g cross-fitted by set; 30 hand-built text features (VADER sentiment added to requirements); λ=300 chosen on synthetic raters.
  13/60 raters (22%) sig, median gain +0.7 pts. But pooled Stouffer z **8.7** (synthetic null −1.5): taste exists, too weak per person to
  show with ~96 answers (synthetic: a planted 1 logit/SD taste is found in 20% of raters, 2 logit/SD in 78%; real ≈ the 1-logit case).
  Text taste is far less consensual than images: agreement with the other 2 raters median 0.58 (LAPIS 0.67); all 3 agree on 38% of pairs.
  "Mixture" pseudo-raters are not a clean null with 3-rater sets (always the same 2 people; z 2.4).

## 8. Environment

- RTX 5070 Ti (16 GB), 60 GB RAM. **Everything runs in Docker** (`Dockerfile`: pytorch 2.15 cu13.2 base + pinned `requirements.txt`;
  the build fails if pip replaces torch). `compose.yaml` bind-mounts the repo at `/workspace` and `~/.cache/huggingface`.
  Usage: `docker compose run --rm taste python src/<script>.py`. One-time per machine: `printf 'UID=%s\nGID=%s\n' $(id -u) $(id -g) > .env`.
- Keep scripts in `src/`, pass data paths as arguments. Papers are not in git: `bash papers/download.sh`.
- Git: commits are not pushed from this machine (no GitHub credentials).

Fetch LAPIS on a new machine:
```bash
mkdir -p data/lapis && cd data/lapis
for f in piaa/piaa.csv giaa/train.csv giaa/validation.csv giaa/test.csv image_attributes/image_attributes.csv image_attributes/image_prompts.csv; do
  mkdir -p $(dirname $f); curl -sSfL -o $f "https://huggingface.co/datasets/asmaerten/LAPIS/resolve/main/$f"; done
cd ../.. && docker compose run --rm taste python -c "from huggingface_hub import snapshot_download as s; s('asmaerten/LAPIS', repo_type='dataset', allow_patterns='images/*', local_dir='data/lapis')"
```

## 9. Working preferences of the user

- Decisions step by step; the user picks the direction. Claude proposes a recommendation and lets the user decide.
- When asked for questions "without recommendations", ask neutrally (facts and trade-offs only). Explain from first principles when asked.
- Verify each build step before launching training.
- Science-backed choices from the best papers; small reading list, keep `papers/INDEX.md` updated.
- Keep this file short: no redundant or stale content, no write-ups of side tests that aren't needed going forward.
