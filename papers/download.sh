#!/usr/bin/env bash
# Download every paper listed in INDEX.md into this folder (PDFs are not committed to git).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p chess_archive

get() { [ -s "$1" ] && { echo "skip  $1"; return; }; echo "fetch $1"; curl -sSfL -o "$1" "$2"; }

get Feng2025_GeneratingCreativeChessPuzzles.pdf  https://arxiv.org/pdf/2510.23881
get Christiano2017_DeepRLHumanPreferences.pdf    https://arxiv.org/pdf/1706.03741
get Stiennon2020_LearningToSummarizeHF.pdf       https://arxiv.org/pdf/2009.01325
get Gao2022_RewardModelOveroptimization.pdf      https://arxiv.org/pdf/2210.10760
get Kirstain2023_PickAPic_PickScore.pdf          https://arxiv.org/pdf/2305.01569
get Xu2023_ImageReward.pdf                       https://arxiv.org/pdf/2304.05977
get PerezOrtiz2017_PairwiseComparisonGuide.pdf   https://arxiv.org/pdf/1712.03686
get Jiang2025_ArtificialHivemind.pdf             https://arxiv.org/pdf/2510.22954
get Yang2022_PARA.pdf                            https://arxiv.org/pdf/2203.16754
get Maerten2025_LAPIS.pdf                        https://arxiv.org/pdf/2504.07670
get Ren2017_PersonalizedImageAesthetics.pdf      https://openaccess.thecvf.com/content_ICCV_2017/papers/Ren_Personalized_Image_Aesthetics_ICCV_2017_paper.pdf
get Maerten2026_PAMELA.pdf                      https://arxiv.org/pdf/2604.07427
get Bose2025_LoRe.pdf                            https://arxiv.org/pdf/2504.14439
get Chen2024_PAL.pdf                             https://arxiv.org/pdf/2406.08469
get Schacht2025_EDPBRL.pdf                       https://arxiv.org/pdf/2512.19057
get Black2023_DDPO.pdf                           https://arxiv.org/pdf/2305.13301
get Fan2023_DPOK.pdf                             https://arxiv.org/pdf/2305.16381
get Clark2023_DRaFT.pdf                          https://arxiv.org/pdf/2309.17400
get Wallace2023_DiffusionDPO.pdf                 https://arxiv.org/pdf/2311.12908
get Uehara2024_RLDiffusionTutorial.pdf           https://arxiv.org/pdf/2407.13734
get Liu2025_FlowGRPO.pdf                         https://arxiv.org/pdf/2505.05470
get DomingoEnrich2024_AdjointMatching.pdf        https://arxiv.org/pdf/2409.08861
get Zhang2024_TDPO_RewardOveroptimization.pdf    https://arxiv.org/pdf/2402.08552
get chess_archive/Zaidi2024_MoveBrillianceChess.pdf    https://arxiv.org/pdf/2406.11895
get chess_archive/McIlroyYoung2020_Maia.pdf            https://arxiv.org/pdf/2006.01855
get chess_archive/Ruoss2024_AmortizedPlanningChess.pdf https://arxiv.org/pdf/2402.04494
