# Source A v4 — MF Routing Score Fix
**Started:** June 23, 2026
**Status:** In progress

## Run log

- June 23, 2026 — Started v4 MF-only routing-score fix. v3 audit artifacts are preserved. Existing v3 BERTScore and embedding caches will be reused by key where possible; new Tier2-vs-Tier3 rows will be scored/embedded incrementally.

- June 23, 2026 23:10:12 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, Tier2vTier3=6864, discarded=32800.

- June 23, 2026 23:10:58 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=12845 (dropped 16488).

- June 23, 2026 23:10:59 CDT — Step A3 reused 9349 BERTScore similarities from v3 JSONL by key.

- June 23, 2026 23:10:59 CDT — Step A3 computing BERTScore for 3496 new or uncached rows.

- June 23, 2026 23:11:24 CDT — Step A3 BERTScore progress: completed 9501 of 12845 rows.

- June 23, 2026 23:12:41 CDT — Step A3 BERTScore progress: completed 10001 of 12845 rows.

- June 23, 2026 23:13:57 CDT — Step A3 BERTScore progress: completed 10501 of 12845 rows.

- June 23, 2026 23:15:19 CDT — Step A3 BERTScore progress: completed 11001 of 12845 rows.

- June 23, 2026 23:16:39 CDT — Step A3 BERTScore progress: completed 11501 of 12845 rows.

- June 23, 2026 23:18:06 CDT — Step A3 BERTScore progress: completed 12001 of 12845 rows.

- June 23, 2026 23:19:39 CDT — Step A3 BERTScore progress: completed 12501 of 12845 rows.

- June 23, 2026 23:20:38 CDT — Step A3 BERTScore progress: completed 12845 of 12845 rows.

- June 23, 2026 23:20:39 CDT — Step A3 complete. BERTScore similarity mean=0.819379, std=0.046280, min=0.497630, max=1.000000. routing_score mean=0.312773, std=0.092970, min=0.050000, max=0.950000. structural_boost_applied=92 rows (0.72%). Final row count=12845.

- June 23, 2026 23:20:39 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 23:20:39 CDT — Step A4 reused 9352 embeddings from v3 cache for unchanged Tier1 rows; 3493 rows remain.

- June 23, 2026 23:20:41 CDT — Step A4 v4 embedding checkpoint saved after batch 1; completed=9402 of 12845.

- June 23, 2026 23:20:42 CDT — Step A4 v4 embedding checkpoint saved after batch 2; completed=9452 of 12845.

- June 23, 2026 23:20:43 CDT — Step A4 v4 embedding checkpoint saved after batch 3; completed=9502 of 12845.

- June 23, 2026 23:20:45 CDT — Step A4 v4 embedding checkpoint saved after batch 4; completed=9552 of 12845.

- June 23, 2026 23:20:47 CDT — Step A4 v4 embedding checkpoint saved after batch 5; completed=9602 of 12845.

- June 23, 2026 23:20:49 CDT — Step A4 v4 embedding checkpoint saved after batch 6; completed=9652 of 12845.

- June 23, 2026 23:20:50 CDT — Step A4 v4 embedding checkpoint saved after batch 7; completed=9702 of 12845.

- June 23, 2026 23:20:51 CDT — Step A4 v4 embedding checkpoint saved after batch 8; completed=9752 of 12845.

- June 23, 2026 23:20:53 CDT — Step A4 v4 embedding checkpoint saved after batch 9; completed=9802 of 12845.

- June 23, 2026 23:20:54 CDT — Step A4 v4 embedding checkpoint saved after batch 10; completed=9852 of 12845.

- June 23, 2026 23:20:56 CDT — Step A4 v4 embedding checkpoint saved after batch 11; completed=9902 of 12845.

- June 23, 2026 23:20:57 CDT — Step A4 v4 embedding checkpoint saved after batch 12; completed=9952 of 12845.

- June 23, 2026 23:20:59 CDT — Step A4 v4 embedding checkpoint saved after batch 13; completed=10002 of 12845.

- June 23, 2026 23:21:00 CDT — Step A4 v4 embedding checkpoint saved after batch 14; completed=10052 of 12845.

- June 23, 2026 23:21:02 CDT — Step A4 v4 embedding checkpoint saved after batch 15; completed=10102 of 12845.

- June 23, 2026 23:21:04 CDT — Step A4 v4 embedding checkpoint saved after batch 16; completed=10152 of 12845.

- June 23, 2026 23:21:05 CDT — Step A4 v4 embedding checkpoint saved after batch 17; completed=10202 of 12845.

- June 23, 2026 23:21:07 CDT — Step A4 v4 embedding checkpoint saved after batch 18; completed=10252 of 12845.

- June 23, 2026 23:21:08 CDT — Step A4 v4 embedding checkpoint saved after batch 19; completed=10302 of 12845.

- June 23, 2026 23:21:10 CDT — Step A4 v4 embedding checkpoint saved after batch 20; completed=10352 of 12845.

- June 23, 2026 23:21:11 CDT — Step A4 v4 embedding checkpoint saved after batch 21; completed=10402 of 12845.

- June 23, 2026 23:21:13 CDT — Step A4 v4 embedding checkpoint saved after batch 22; completed=10452 of 12845.

- June 23, 2026 23:21:14 CDT — Step A4 v4 embedding checkpoint saved after batch 23; completed=10502 of 12845.

- June 23, 2026 23:21:16 CDT — Step A4 v4 embedding checkpoint saved after batch 24; completed=10552 of 12845.

- June 23, 2026 23:21:18 CDT — Step A4 v4 embedding checkpoint saved after batch 25; completed=10602 of 12845.

- June 23, 2026 23:21:19 CDT — Step A4 v4 embedding checkpoint saved after batch 26; completed=10652 of 12845.

- June 23, 2026 23:21:21 CDT — Step A4 v4 embedding checkpoint saved after batch 27; completed=10702 of 12845.

- June 23, 2026 23:21:22 CDT — Step A4 v4 embedding checkpoint saved after batch 28; completed=10752 of 12845.

- June 23, 2026 23:21:23 CDT — Step A4 v4 embedding checkpoint saved after batch 29; completed=10802 of 12845.

- June 23, 2026 23:21:25 CDT — Step A4 v4 embedding checkpoint saved after batch 30; completed=10852 of 12845.

- June 23, 2026 23:21:27 CDT — Step A4 v4 embedding checkpoint saved after batch 31; completed=10902 of 12845.

- June 23, 2026 23:21:28 CDT — Step A4 v4 embedding checkpoint saved after batch 32; completed=10952 of 12845.

- June 23, 2026 23:21:30 CDT — Step A4 v4 embedding checkpoint saved after batch 33; completed=11002 of 12845.

- June 23, 2026 23:21:31 CDT — Step A4 v4 embedding checkpoint saved after batch 34; completed=11052 of 12845.

- June 23, 2026 23:21:33 CDT — Step A4 v4 embedding checkpoint saved after batch 35; completed=11102 of 12845.

- June 23, 2026 23:21:34 CDT — Step A4 v4 embedding checkpoint saved after batch 36; completed=11152 of 12845.

- June 23, 2026 23:21:36 CDT — Step A4 v4 embedding checkpoint saved after batch 37; completed=11202 of 12845.

- June 23, 2026 23:21:37 CDT — Step A4 v4 embedding checkpoint saved after batch 38; completed=11252 of 12845.

- June 23, 2026 23:21:39 CDT — Step A4 v4 embedding checkpoint saved after batch 39; completed=11302 of 12845.

- June 23, 2026 23:21:41 CDT — Step A4 v4 embedding checkpoint saved after batch 40; completed=11352 of 12845.

- June 23, 2026 23:21:42 CDT — Step A4 v4 embedding checkpoint saved after batch 41; completed=11402 of 12845.

- June 23, 2026 23:21:44 CDT — Step A4 v4 embedding checkpoint saved after batch 42; completed=11452 of 12845.

- June 23, 2026 23:21:45 CDT — Step A4 v4 embedding checkpoint saved after batch 43; completed=11502 of 12845.

- June 23, 2026 23:21:47 CDT — Step A4 v4 embedding checkpoint saved after batch 44; completed=11552 of 12845.

- June 23, 2026 23:21:49 CDT — Step A4 v4 embedding checkpoint saved after batch 45; completed=11602 of 12845.

- June 23, 2026 23:21:50 CDT — Step A4 v4 embedding checkpoint saved after batch 46; completed=11652 of 12845.

- June 23, 2026 23:21:52 CDT — Step A4 v4 embedding checkpoint saved after batch 47; completed=11702 of 12845.

- June 23, 2026 23:21:54 CDT — Step A4 v4 embedding checkpoint saved after batch 48; completed=11752 of 12845.

- June 23, 2026 23:21:55 CDT — Step A4 v4 embedding checkpoint saved after batch 49; completed=11802 of 12845.

- June 23, 2026 23:21:56 CDT — Step A4 v4 embedding checkpoint saved after batch 50; completed=11852 of 12845.

- June 23, 2026 23:21:58 CDT — Step A4 v4 embedding checkpoint saved after batch 51; completed=11902 of 12845.

- June 23, 2026 23:21:59 CDT — Step A4 v4 embedding checkpoint saved after batch 52; completed=11952 of 12845.

- June 23, 2026 23:22:01 CDT — Step A4 v4 embedding checkpoint saved after batch 53; completed=12002 of 12845.

- June 23, 2026 23:22:02 CDT — Step A4 v4 embedding checkpoint saved after batch 54; completed=12052 of 12845.

- June 23, 2026 23:22:04 CDT — Step A4 v4 embedding checkpoint saved after batch 55; completed=12102 of 12845.

- June 23, 2026 23:22:05 CDT — Step A4 v4 embedding checkpoint saved after batch 56; completed=12152 of 12845.

- June 23, 2026 23:22:07 CDT — Step A4 v4 embedding checkpoint saved after batch 57; completed=12202 of 12845.

- June 23, 2026 23:22:09 CDT — Step A4 v4 embedding checkpoint saved after batch 58; completed=12252 of 12845.

- June 23, 2026 23:22:10 CDT — Step A4 v4 embedding checkpoint saved after batch 59; completed=12302 of 12845.

- June 23, 2026 23:22:12 CDT — Step A4 v4 embedding checkpoint saved after batch 60; completed=12352 of 12845.

- June 23, 2026 23:22:13 CDT — Step A4 v4 embedding checkpoint saved after batch 61; completed=12402 of 12845.

- June 23, 2026 23:22:14 CDT — Step A4 v4 embedding checkpoint saved after batch 62; completed=12452 of 12845.

- June 23, 2026 23:22:16 CDT — Step A4 v4 embedding checkpoint saved after batch 63; completed=12502 of 12845.

- June 23, 2026 23:22:17 CDT — Step A4 v4 embedding checkpoint saved after batch 64; completed=12552 of 12845.

- June 23, 2026 23:22:19 CDT — Step A4 v4 embedding checkpoint saved after batch 65; completed=12602 of 12845.

- June 23, 2026 23:22:20 CDT — Step A4 v4 embedding checkpoint saved after batch 66; completed=12652 of 12845.

- June 23, 2026 23:22:21 CDT — Step A4 v4 embedding checkpoint saved after batch 67; completed=12702 of 12845.

- June 23, 2026 23:22:23 CDT — Step A4 v4 embedding checkpoint saved after batch 68; completed=12752 of 12845.

- June 23, 2026 23:22:25 CDT — Step A4 v4 embedding checkpoint saved after batch 69; completed=12802 of 12845.

- June 23, 2026 23:22:26 CDT — Step A4 v4 embedding checkpoint saved after batch 70; completed=12845 of 12845.

- June 23, 2026 23:22:29 CDT — Step A4 complete. Total clusters=500, kept=485, discarded=15, mean_confidence_weight=0.653627, final row count after cluster filter=12845.

- June 23, 2026 23:22:31 CDT — Phase B MF epoch 1 weighted BCE train loss=1.555403, val loss=1.446381.

- June 23, 2026 23:22:31 CDT — Phase B MF epoch 5 weighted BCE train loss=1.171241, val loss=1.336991.

- June 23, 2026 23:22:32 CDT — Phase B MF epoch 10 weighted BCE train loss=1.081939, val loss=1.259745.

- June 23, 2026 23:22:33 CDT — Phase B MF epoch 20 weighted BCE train loss=0.894548, val loss=0.995179.

- June 23, 2026 23:22:35 CDT — Phase B MF epoch 30 weighted BCE train loss=0.816355, val loss=0.954304.

- June 23, 2026 23:22:36 CDT — Phase B MF epoch 40 weighted BCE train loss=0.728511, val loss=0.843039.

- June 23, 2026 23:22:38 CDT — Phase B MF epoch 50 weighted BCE train loss=0.703329, val loss=0.783639.

- June 23, 2026 23:22:38 CDT — Phase B complete. MF train BCE=0.690074, val BCE=0.782940, best val BCE=0.782940.

## Phase C — Arena held-out test evaluation

| Metric | MF |
|---|---:|
| Spearman correlation | 0.081878 |
| ROC-AUC (score > 0.5 = cloud) | 0.554220 |
| Brier score | 0.194257 |
| Precision @ threshold 0.40 | 0.046025 |
| Recall @ threshold 0.40 | 0.540984 |
| Precision @ threshold 0.50 | 0.047337 |
| Recall @ threshold 0.50 | 0.393443 |
| Precision @ threshold 0.60 | 0.046243 |
| Recall @ threshold 0.60 | 0.262295 |

Compared against v1 ROC-AUC 0.5203 and v2 ROC-AUC 0.6296. Test size: 1790. Test positive rate: 0.034078.

- June 23, 2026 23:22:38 CDT — Pre-MMLU sanity check passed; max score=0.595602.

- June 23, 2026 23:24:24 CDT — Phase C MMLU progress: evaluated 50 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:26:01 CDT — Phase C MMLU progress: evaluated 100 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:27:47 CDT — Phase C MMLU progress: evaluated 150 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:29:14 CDT — Phase C MMLU progress: evaluated 200 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:30:48 CDT — Phase C MMLU progress: evaluated 250 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:32:05 CDT — Phase C MMLU progress: evaluated 300 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:33:38 CDT — Phase C MMLU progress: evaluated 350 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:35:10 CDT — Phase C MMLU progress: evaluated 400 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:36:41 CDT — Phase C MMLU progress: evaluated 450 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 23:38:29 CDT — Phase C MMLU progress: evaluated 500 of 500 questions with weak model `qwen3:8b`.

## Phase C — MMLU validation evaluation

Sample size: 500

Weak model: `qwen3:8b`

Chosen threshold: 0.3

Routed accuracy: 1.25%

Over-routing rate: 0.00%

Under-routing rate: 98.75%

---
## Final summary

**Completed:** June 23, 2026 23:38:29 CDT
**Status:** Target not met

**Dataset**
- Raw battles loaded: 57477
- Surviving after all filters: 12845
- Surviving after cluster filter: 12845
- Tier1vTier2 battles: 1935
- Tier1vTier3 battles: 7417
- Tier2vTier3 battles: 3493

**Routing score distribution**
- Mean: 0.312773   Std: 0.092970   Min: 0.050000   Max: 0.950000

**Model performance**
| Metric | MF |
|---|---:|
| Spearman correlation | 0.081878 |
| ROC-AUC (score > 0.5 = cloud) | 0.554220 |
| Brier score | 0.194257 |
| Precision @ threshold 0.40 | 0.046025 |
| Recall @ threshold 0.40 | 0.540984 |
| Precision @ threshold 0.50 | 0.047337 |
| Recall @ threshold 0.50 | 0.393443 |
| Precision @ threshold 0.60 | 0.046243 |
| Recall @ threshold 0.60 | 0.262295 |

**MMLU routing accuracy:** 1.25%
**MMLU chosen threshold:** 0.3
**Over-routing rate:** 0.00%
**Under-routing rate:** 98.75%

**vs baselines**
- v1 ROC-AUC: 0.5203  →  v4: 0.554220
- v2 ROC-AUC: 0.6296  →  v4: 0.554220

**Artifacts**
- data/arena-preference-v4.jsonl
- models/local-cloud-classifier-v4-mf.pt
- models/local-cloud-classifier-v4-ensemble.pkl
- data/local-cloud-classifier-v4-evaluation.json

**Assessment**
The v4 MF-only model did not meet the Arena ROC-AUC target: v4 ROC-AUC is 0.554220, compared with v1 0.5203 and v2 0.6296. The SW component was removed, so the reported score is the direct MF routing signal.

The MMLU sample used local model `qwen3:8b` and produced routed accuracy 1.25%, over-routing 0.00%, and under-routing 98.75% at threshold 0.3. This out-of-distribution result indicates whether Arena preference data transfers to exam-style questions; if under-routing remains material or ROC-AUC is below target, Source A alone is not sufficient and Source B augmentation remains required before production routing.
---
