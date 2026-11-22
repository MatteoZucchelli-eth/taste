# Project context — handoff (written 2026-11-22)

Read this first, then `papers/INDEX.md`. This file is the full state of the project so far, so work can resume
on another machine (one with a GPU) without the original chat.

## 1. Goal

Critics often say generative models (LLMs, diffusion) can't produce original or tasteful content. The aim is to
**give a generative model taste**. The plan is to first **learn a critic** that captures the taste of *specific individual
experts*, and only later use it as the reward for a generator.

- The **end goal is design generation**, so the domain is images. Chess was dropped.
- **Step 1 (current): learn and verify a pairwise critic.** Given two items, it predicts which one a specific person prefers.
  Then check whether its scores match that person's.
- **Step 2 (deferred, do not start until the user asks):** use the critic as the RL / fine-tuning reward for a generator
  (e.g. diffusion, à la ImageReward/ReFL).
- The user wants to model a **few experts individually**, not a consensus. Averaging raters erases taste (Artificial Hivemind 2025).
- **The critic must be verified before any training of a generator.**
- Everything must be **backed by top papers**. Keep the literature list **lean**, because the user dislikes noisy reading lists.

## 2. How we got here

1. **Origin:** Feng et al. 2025, *Generating Creative Chess Puzzles* (DeepMind, NeurIPS 2025, `papers/Feng2025_...pdf`).
   They RL-train a puzzle generator with a **hand-designed scorer** (engine-based uniqueness + counter-intuitiveness, weights tuned
   by grid search on a tiny 84+41 "Golden Set"). Key facts we took from it:
   - their score correlates with puzzle *difficulty* but has **~0 correlation with user popularity** (Table 6), so it doesn't capture liking;
   - optimising it caused reward hacking (entropy collapse, extra pieces, engine exploits);
   - aesthetics was only filtered afterwards with weak rule-based detectors.
   → Motivation: in domains like design you *can't* write such rules, so learn the critic from people instead.
2. The first idea was to replicate this in chess with a learned critic. It was dropped because there is no per-expert chess data,
   expert labels are expensive, and the end goal is design anyway.
3. A synthetic-experts step was proposed (hidden scoring functions as fake experts, Gao-2022-style), then dropped together with chess.
   It can come back as a sanity check on images if it's useful.
4. Domain chosen: **personalized image aesthetics (PIAA)**.
   - **PARA** (CVPR 2022) would be ideal: 438 raters, ~1.8k images each, **repeated images inside each session, so per-rater test–retest
     consistency is available**, plus photographic/artistic experience per rater. But access needs registration at
     cv-datasets.institutecv.com (terms unread, no public mirror). The user may request it later.
   - **LAPIS was chosen** (user decision): openly downloadable, artworks curated with art historians, closer to design.

## 3. Dataset: LAPIS

- Hugging Face: `asmaerten/LAPIS` (ungated, **CC BY-NC 4.0**, non-commercial). Paper: arXiv 2504.07670 (`papers/Maerten2025_LAPIS.pdf`).
- Cleaned release: **281,174 individual ratings, 11,723 artworks, ~24 ratings/image (range 13–67)**. 552 Prolific raters
  (~500 ratings each on average). Images are split into `figurative/` and `abstract/`.
- Files:
  - `piaa/piaa.csv` (~84 MB): one row per rating, with `rating`, `response time`, `participant_id`, rater demographics,
    **art-interest score**, `imgs_known`. Read it with `pd.read_csv(..., keep_default_na=False, na_values=[""])`, otherwise `imgs_known="None"` turns into NaN.
  - `giaa/{train,validation,test}.csv`: per-image aggregates (`mean_response`, `std_response`, number of raters…). This is the official split.
  - `image_attributes/image_attributes.csv`: low-level visual stats. `image_attributes/image_prompts.csv`: a generated caption per image.
  - `images/` (~716 MB): join on the `path` column (e.g. `figurative/<file>.jpg`) or on `image_filename`.
- **There are no repeated images in LAPIS**, so the per-rater ceiling has to come from split-half reliability or from how well
  other raters predict a rater (see §5). This is the main weakness compared with PARA.
- In the paper, raters with more art interest rate differently (e.g. novices score abstract art lower). Use art interest to pick
  "expert" raters.

## 4. Literature (full notes in `papers/INDEX.md`)

| Role | Paper |
|---|---|
| Data | Maerten 2025 **LAPIS**; Yang 2022 **PARA** (fallback / later) |
| Personalization baseline | Ren 2017 *Personalized Image Aesthetics* (generic model + small learned per-user residual; active learning) |
| Pairwise critic method | Christiano 2017 (Bradley–Terry reward model); Stiennon 2020 (judge the RM against the inter-labeler ceiling) |
| Pairwise → scales, study design | Pérez-Ortiz & Mantiuk 2017 |
| Image critics + bridge to diffusion | Kirstain 2023 PickScore (CLIP + BT head); Xu 2023 ImageReward (expert comparisons → ReFL) |
| Risks / motivation | Gao 2022 RM overoptimization; Jiang 2025 Artificial Hivemind; Feng 2025 (origin) |

`papers/chess_archive/` holds the set-aside chess papers (Zaidi 2024, Maia 2020, Ruoss 2024).

**A key warning from the literature:** in PARA's benchmark, 10-shot personalization ≈ no personalization. Most of an individual's
rating is consensus. **The personalization gap is small, and showing it is the whole point.**

## 5. Pre-registered success criterion (decided before any training)

For a chosen rater A, on A's held-out images/pairs:

1. `agree(critic_A, A)` > `agree(critic_consensus, A)`, i.e. the personal critic beats one trained on everyone;
2. `agree(critic_A, A)` > `agree(critic_B, A)` for other raters B, i.e. it captured A specifically;
3. both measured against **A's ceiling**. With no repeats in LAPIS, estimate it from split-half reliability and from how well the
   mean of the other raters predicts A (leave-one-rater-out consensus);
4. also check the critic's **top-ranked** items against A's ratings (best-of-n), not only pairwise accuracy (Gao 2022).

Metrics: pairwise accuracy on within-rater pairs (ties dropped), Spearman (SROCC) per rater. Report everything per rater, with
CIs, for the few selected "experts" plus a larger rater pool for statistics.

## 6. Planned next steps (none executed yet)

**Phase A: data verification, no model training.**
1. Download the LAPIS tables (and later the images) into `data/lapis/` (keep the HF folder layout).
2. Per-rater statistics: number of ratings, rating scale and usage (does the rater use the whole scale?), response times,
   art-interest distribution, how many images pairs of raters share.
3. Build within-rater pairs from absolute ratings (drop ties; Pérez-Ortiz guide).
4. **Measure the ceilings:** for each rater, the agreement of the leave-one-out consensus with that rater, and split-half reliability.
   This tells us how much *individual* taste exists to learn before we train anything. If it's ≈0 for everyone, stop and rethink.
5. Select the candidate "experts": high art interest, many ratings, and **low agreement with consensus while staying internally consistent**.

**Phase B: critics (GPU machine).**
6. Embed every image once with a frozen encoder (CLIP ViT-L/14 or SigLIP; DINOv2 as an alternative). Cache the embeddings to disk.
7. Consensus critic: BT/regression head on the embeddings, trained on all raters (official giaa split).
8. Personal critics: Ren-2017 style, consensus + small per-rater residual head, trained with BT loss on A's pairs. Vary the number of
   ratings per rater (10/50/100/all) to get a label-efficiency curve.
9. Evaluate with the §5 criterion.

## 7. Environment state on the original (non-GPU) machine

- Only the **system Python 3.12, without pandas/torch**, no uv/conda, no GPU.
- An interrupted setup command left `.venv/` (pip install of pandas/numpy/scipy/matplotlib may be incomplete) and a
  **truncated `data/lapis/piaa/piaa.csv` (82.4 MB of ~83.6 MB). Don't use it; download it again.**
- Nothing else was run. No analysis code exists yet.

Suggested setup on the GPU machine:
```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy scipy matplotlib torch torchvision open_clip_torch huggingface_hub
mkdir -p data/lapis && cd data/lapis
for f in piaa/piaa.csv giaa/train.csv giaa/validation.csv giaa/test.csv image_attributes/image_attributes.csv image_attributes/image_prompts.csv; do
  mkdir -p $(dirname $f); curl -L -o $f "https://huggingface.co/datasets/asmaerten/LAPIS/resolve/main/$f"; done
# images (~716 MB): huggingface_hub.snapshot_download("asmaerten/LAPIS", repo_type="dataset", allow_patterns="images/*", local_dir="data/lapis")
```
Paper PDFs are not in git: run `bash papers/download.sh` to fetch them all.
Keep analysis scripts outside `data/` (e.g. in `src/`), and pass data paths as arguments.

## 8. Working preferences of the user
- Decisions are made step by step. Verify before training. Do not start step 2 (RL/generator) unless asked.
- Science-backed choices only, from the best papers. Keep the reading list small and update `papers/INDEX.md` when it changes.
- The user prefers to choose the direction (e.g. picked LAPIS over PARA, dropped chess). Propose a recommendation and let them decide.
