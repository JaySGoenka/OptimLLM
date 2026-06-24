# Source A v3 — Probabilistic Ensemble Classifier
**Started:** June 23, 2026 14:41:45 CDT
**Status:** In progress

## Run log

- June 23, 2026 14:41:45 CDT — Housekeeping started. Preserved unique `docs/phase-progress.md` content in `docs/progress.md`, then removed approved legacy Source A implementation/data/evaluation files.
- June 23, 2026 14:41:45 CDT — Removed files/directories: `scripts/rebuild_source_a_v2.py`, `scripts/train_source_a_classifier.py`, `data/arena-preference.jsonl`, `data/arena-preference-v2.jsonl`, `data/.phase2-cache/`, `data/local-cloud-classifier-evaluation.json`, `data/local-cloud-classifier-v2-evaluation.json`, `docs/phase-progress.md`.
- June 23, 2026 14:41:45 CDT — Kept files: `data/arena-model-tier-map.json`, `data/lmsys-sample.jsonl`, `docs/progress.md`, `models/local-cloud-classifier.pkl`, `models/local-cloud-classifier-v2.pkl`.

- June 23, 2026 14:45:03 CDT — Step A1 completed. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 14:45:03 CDT — Step A2 completed. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 14:45:24 CDT — FAILED at Step A3: native segmentation fault during `bert_score.score(...)` using `distilbert-base-uncased` after model weights loaded. Exact process error: `Segmentation fault: 11  python scripts/source_a_v3_pipeline.py`. No v3 dataset/model/evaluation artifacts were completed after this failure.

- June 23, 2026 14:48:10 CDT — Applied Step A3 stability fixes: switched from repeated `bert_score.score(...)` calls to a persistent `BERTScorer`, reduced BERTScore batch size from 16 to 4, truncated each response to 3000 normalized characters before scoring, and added resumable checkpointing at `data/.cache-v3/bertscore-f1-checkpoint.npy`.

- June 23, 2026 14:49:18 CDT — FAILED at Step A3 after applying stability fixes: native segmentation fault still occurred while loading/scoring `distilbert-base-uncased` through BERTScore. Exact process error: `Segmentation fault: 11  python scripts/source_a_v3_pipeline.py`. The crash happened before any `data/.cache-v3/bertscore-f1-checkpoint.npy` rows were written.

### Step A3 environment recovery — June 23, 2026

Initial BERTScore execution failed with native macOS error `Segmentation fault: 11` before dataset scoring began.

Failing environment:

- Python 3.11.15
- macOS arm64
- torch 2.12.1
- transformers 5.12.1
- tokenizers 0.22.2
- bert-score 0.3.13

A cloned Conda environment will be created and the BERTScore stack will be pinned to:

- torch 2.5.1
- transformers 4.45.2
- tokenizers 0.20.3
- bert-score 0.3.13

Step A3 remains in progress pending successful runtime validation.

- June 23, 2026 15:06:22 CDT — Runtime isolation found the native crash trigger: importing `faiss` before constructing/running BERTScore causes a segmentation fault inside Torch `layer_norm`. Importing `datasets` and `sklearn` before BERTScore did not crash. Patched `scripts/source_a_v3_pipeline.py` to remove the top-level FAISS import and lazy-load FAISS only inside similarity-weighted ranking functions after Step A3.

- June 23, 2026 15:08:06 CDT — Step A3 real-data validation passed on 20 Arena rows in cloned environment `optimllm-data-bertscore`. Validation results: BERTScore similarity mean=0.814043, std=0.035796, min=0.753201, max=0.879690; routing_score mean=0.463308, std=0.327869, min=0.050000, max=0.950000; structural boost applied to 1 of 20 rows (5.00%).

- June 23, 2026 14:48:03 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 14:48:49 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 15:45:17 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 15:46:06 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 17:19:18 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 17:20:04 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 17:20:04 CDT — Step A3 real-data validation mode enabled with 20 rows.

- June 23, 2026 17:20:53 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 17:21:39 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 17:21:39 CDT — Step A3 real-data validation mode enabled with 20 rows.

- June 23, 2026 17:21:42 CDT — Step A3 BERTScore progress: computed 20 of 20 rows.

- June 23, 2026 17:21:42 CDT — Step A3 complete. BERTScore similarity mean=0.814043, std=0.035796, min=0.753201, max=0.879690. routing_score mean=0.463308, std=0.327869, min=0.050000, max=0.950000. structural_boost_applied=1 rows (5.00%). Final row count=20.

- June 23, 2026 17:21:42 CDT — Step A3 real-data validation completed successfully for 20 rows; stopping before Step A4 by design.

- June 23, 2026 17:22:02 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 17:22:49 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 17:22:49 CDT — Step A3 resumed BERTScore checkpoint with 20 completed rows.

- June 23, 2026 17:23:53 CDT — Step A3 BERTScore progress: computed 500 of 9352 rows.

- June 23, 2026 17:25:03 CDT — Step A3 BERTScore progress: computed 1000 of 9352 rows.

- June 23, 2026 17:26:12 CDT — Step A3 BERTScore progress: computed 1500 of 9352 rows.

- June 23, 2026 17:27:24 CDT — Step A3 BERTScore progress: computed 2000 of 9352 rows.

- June 23, 2026 17:28:36 CDT — Step A3 BERTScore progress: computed 2500 of 9352 rows.

- June 23, 2026 17:29:49 CDT — Step A3 BERTScore progress: computed 3000 of 9352 rows.

- June 23, 2026 17:31:07 CDT — Step A3 BERTScore progress: computed 3500 of 9352 rows.

- June 23, 2026 17:32:24 CDT — Step A3 BERTScore progress: computed 4000 of 9352 rows.

- June 23, 2026 17:33:40 CDT — Step A3 BERTScore progress: computed 4500 of 9352 rows.

- June 23, 2026 17:34:58 CDT — Step A3 BERTScore progress: computed 5000 of 9352 rows.

- June 23, 2026 17:36:20 CDT — Step A3 BERTScore progress: computed 5500 of 9352 rows.

- June 23, 2026 17:37:38 CDT — Step A3 BERTScore progress: computed 6000 of 9352 rows.

- June 23, 2026 17:38:53 CDT — Step A3 BERTScore progress: computed 6500 of 9352 rows.

- June 23, 2026 17:40:12 CDT — Step A3 BERTScore progress: computed 7000 of 9352 rows.

- June 23, 2026 17:41:27 CDT — Step A3 BERTScore progress: computed 7500 of 9352 rows.

- June 23, 2026 17:42:49 CDT — Step A3 BERTScore progress: computed 8000 of 9352 rows.

- June 23, 2026 17:44:06 CDT — Step A3 BERTScore progress: computed 8500 of 9352 rows.

- June 23, 2026 17:45:26 CDT — Step A3 BERTScore progress: computed 9000 of 9352 rows.

- June 23, 2026 17:46:21 CDT — Step A3 BERTScore progress: computed 9352 of 9352 rows.

- June 23, 2026 17:46:21 CDT — Step A3 complete. BERTScore similarity mean=0.818225, std=0.046888, min=0.497630, max=1.000000. routing_score mean=0.313849, std=0.136595, min=0.050000, max=0.950000. structural_boost_applied=77 rows (0.82%). Final row count=9352.

- June 23, 2026 17:46:21 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 17:46:23 CDT — Step A4 embedding checkpoint saved after batch 1; completed=50 of 9352.

- June 23, 2026 17:46:34 CDT — FAILED at Step A4: RuntimeError: embedding batch 2 failed after 3 retries; completed=50

- June 23, 2026 18:05:03 CDT — Step A4 retry failed again at embedding batch 2 with 50 completed embeddings. Isolated the cause to row 97: prompt length was 12000 characters and Ollama returned `{"error":"the input length exceeds the context length"}` for `nomic-embed-text`. Patched `scripts/source_a_v3_pipeline.py` to truncate prompt text to 4000 normalized characters for embedding requests only; full prompt text remains preserved in `data/arena-preference-v3.jsonl`.

- June 23, 2026 18:12:48 CDT — Step A4 advanced from 50 to 8900 embeddings, then failed at batch 179. Isolated the cause to row 8900: the 4000-character embedding prompt still exceeded `nomic-embed-text` context length. Patched `scripts/source_a_v3_pipeline.py` so embedding requests fall back from the normal window to 3000, then 2000, then 1000 normalized characters only when Ollama reports a context-length failure.

- June 23, 2026 18:18:12 CDT — Step A4 completed after resuming from 8900 embeddings. Final embedding checkpoint and final embedding cache both have shape (9352, 768). Cluster filter completed with 500 clusters, 487 kept, 13 discarded, mean confidence weight 0.487822, and 9349 rows after cluster filter. The process then crashed with `Segmentation fault: 11` during Phase B after MF epoch 20, before saving the full ensemble. Patched `scripts/source_a_v3_pipeline.py` to skip BERTScore scorer construction when the full Step A3 checkpoint already exists, allowing post-A3 work to resume in the original `optimllm-data` environment instead of the BERTScore clone.

- June 23, 2026 18:22:41 CDT — Retried in original `optimllm-data` environment. Step A4 loaded final embedding cache and completed again with 500 clusters, 487 kept, 13 discarded, mean confidence weight 0.487822, and 9349 rows after cluster filter. Phase B again crashed with `Segmentation fault: 11` immediately after MF epoch 20. Confirmed `models/local-cloud-classifier-v3-mf.pt`, `models/local-cloud-sw-index-v3.faiss`, and `models/local-cloud-sw-data-v3.npy` exist, but `models/local-cloud-classifier-v3-ensemble.pkl` and `data/local-cloud-classifier-v3-evaluation.json` are not complete.

- June 23, 2026 18:27:10 CDT — Removed FAISS from the active Python prediction path to avoid the native FAISS/Torch segmentation fault. `models/local-cloud-sw-index-v3.faiss` remains preserved as the required FAISS artifact. Similarity-weighted predictions now use the saved normalized training embeddings in `models/local-cloud-sw-data-v3.npy` with NumPy cosine search.

- June 23, 2026 17:55:46 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 17:56:34 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 17:56:35 CDT — Step A3 resumed BERTScore checkpoint with 9352 completed rows.

- June 23, 2026 17:56:36 CDT — Step A3 complete. BERTScore similarity mean=0.818225, std=0.046888, min=0.497630, max=1.000000. routing_score mean=0.313849, std=0.136595, min=0.050000, max=0.950000. structural_boost_applied=77 rows (0.82%). Final row count=9352.

- June 23, 2026 17:56:36 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 17:56:36 CDT — Step A4 resumed embeddings checkpoint with 50 completed rows.

- June 23, 2026 17:56:48 CDT — FAILED at Step A4: RuntimeError: embedding batch 2 failed after 3 retries; completed=50

- June 23, 2026 17:58:01 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 17:58:48 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 17:58:48 CDT — Step A3 resumed BERTScore checkpoint with 9352 completed rows.

- June 23, 2026 17:58:49 CDT — Step A3 complete. BERTScore similarity mean=0.818225, std=0.046888, min=0.497630, max=1.000000. routing_score mean=0.313849, std=0.136595, min=0.050000, max=0.950000. structural_boost_applied=77 rows (0.82%). Final row count=9352.

- June 23, 2026 17:58:49 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 17:58:49 CDT — Step A4 resumed embeddings checkpoint with 50 completed rows.

- June 23, 2026 17:58:51 CDT — Step A4 embedding checkpoint saved after batch 2; completed=100 of 9352.

- June 23, 2026 17:58:52 CDT — Step A4 embedding checkpoint saved after batch 3; completed=150 of 9352.

- June 23, 2026 17:58:54 CDT — Step A4 embedding checkpoint saved after batch 4; completed=200 of 9352.

- June 23, 2026 17:58:55 CDT — Step A4 embedding checkpoint saved after batch 5; completed=250 of 9352.

- June 23, 2026 17:58:56 CDT — Step A4 embedding checkpoint saved after batch 6; completed=300 of 9352.

- June 23, 2026 17:58:58 CDT — Step A4 embedding checkpoint saved after batch 7; completed=350 of 9352.

- June 23, 2026 17:58:59 CDT — Step A4 embedding checkpoint saved after batch 8; completed=400 of 9352.

- June 23, 2026 17:59:01 CDT — Step A4 embedding checkpoint saved after batch 9; completed=450 of 9352.

- June 23, 2026 17:59:02 CDT — Step A4 embedding checkpoint saved after batch 10; completed=500 of 9352.

- June 23, 2026 17:59:03 CDT — Step A4 embedding checkpoint saved after batch 11; completed=550 of 9352.

- June 23, 2026 17:59:05 CDT — Step A4 embedding checkpoint saved after batch 12; completed=600 of 9352.

- June 23, 2026 17:59:06 CDT — Step A4 embedding checkpoint saved after batch 13; completed=650 of 9352.

- June 23, 2026 17:59:07 CDT — Step A4 embedding checkpoint saved after batch 14; completed=700 of 9352.

- June 23, 2026 17:59:09 CDT — Step A4 embedding checkpoint saved after batch 15; completed=750 of 9352.

- June 23, 2026 17:59:10 CDT — Step A4 embedding checkpoint saved after batch 16; completed=800 of 9352.

- June 23, 2026 17:59:11 CDT — Step A4 embedding checkpoint saved after batch 17; completed=850 of 9352.

- June 23, 2026 17:59:13 CDT — Step A4 embedding checkpoint saved after batch 18; completed=900 of 9352.

- June 23, 2026 17:59:14 CDT — Step A4 embedding checkpoint saved after batch 19; completed=950 of 9352.

- June 23, 2026 17:59:15 CDT — Step A4 embedding checkpoint saved after batch 20; completed=1000 of 9352.

- June 23, 2026 17:59:17 CDT — Step A4 embedding checkpoint saved after batch 21; completed=1050 of 9352.

- June 23, 2026 17:59:18 CDT — Step A4 embedding checkpoint saved after batch 22; completed=1100 of 9352.

- June 23, 2026 17:59:19 CDT — Step A4 embedding checkpoint saved after batch 23; completed=1150 of 9352.

- June 23, 2026 17:59:20 CDT — Step A4 embedding checkpoint saved after batch 24; completed=1200 of 9352.

- June 23, 2026 17:59:22 CDT — Step A4 embedding checkpoint saved after batch 25; completed=1250 of 9352.

- June 23, 2026 17:59:23 CDT — Step A4 embedding checkpoint saved after batch 26; completed=1300 of 9352.

- June 23, 2026 17:59:25 CDT — Step A4 embedding checkpoint saved after batch 27; completed=1350 of 9352.

- June 23, 2026 17:59:26 CDT — Step A4 embedding checkpoint saved after batch 28; completed=1400 of 9352.

- June 23, 2026 17:59:27 CDT — Step A4 embedding checkpoint saved after batch 29; completed=1450 of 9352.

- June 23, 2026 17:59:29 CDT — Step A4 embedding checkpoint saved after batch 30; completed=1500 of 9352.

- June 23, 2026 17:59:30 CDT — Step A4 embedding checkpoint saved after batch 31; completed=1550 of 9352.

- June 23, 2026 17:59:31 CDT — Step A4 embedding checkpoint saved after batch 32; completed=1600 of 9352.

- June 23, 2026 17:59:33 CDT — Step A4 embedding checkpoint saved after batch 33; completed=1650 of 9352.

- June 23, 2026 17:59:34 CDT — Step A4 embedding checkpoint saved after batch 34; completed=1700 of 9352.

- June 23, 2026 17:59:35 CDT — Step A4 embedding checkpoint saved after batch 35; completed=1750 of 9352.

- June 23, 2026 17:59:37 CDT — Step A4 embedding checkpoint saved after batch 36; completed=1800 of 9352.

- June 23, 2026 17:59:38 CDT — Step A4 embedding checkpoint saved after batch 37; completed=1850 of 9352.

- June 23, 2026 17:59:39 CDT — Step A4 embedding checkpoint saved after batch 38; completed=1900 of 9352.

- June 23, 2026 17:59:41 CDT — Step A4 embedding checkpoint saved after batch 39; completed=1950 of 9352.

- June 23, 2026 17:59:42 CDT — Step A4 embedding checkpoint saved after batch 40; completed=2000 of 9352.

- June 23, 2026 17:59:43 CDT — Step A4 embedding checkpoint saved after batch 41; completed=2050 of 9352.

- June 23, 2026 17:59:44 CDT — Step A4 embedding checkpoint saved after batch 42; completed=2100 of 9352.

- June 23, 2026 17:59:46 CDT — Step A4 embedding checkpoint saved after batch 43; completed=2150 of 9352.

- June 23, 2026 17:59:47 CDT — Step A4 embedding checkpoint saved after batch 44; completed=2200 of 9352.

- June 23, 2026 17:59:48 CDT — Step A4 embedding checkpoint saved after batch 45; completed=2250 of 9352.

- June 23, 2026 17:59:50 CDT — Step A4 embedding checkpoint saved after batch 46; completed=2300 of 9352.

- June 23, 2026 17:59:51 CDT — Step A4 embedding checkpoint saved after batch 47; completed=2350 of 9352.

- June 23, 2026 17:59:53 CDT — Step A4 embedding checkpoint saved after batch 48; completed=2400 of 9352.

- June 23, 2026 17:59:54 CDT — Step A4 embedding checkpoint saved after batch 49; completed=2450 of 9352.

- June 23, 2026 17:59:55 CDT — Step A4 embedding checkpoint saved after batch 50; completed=2500 of 9352.

- June 23, 2026 17:59:56 CDT — Step A4 embedding checkpoint saved after batch 51; completed=2550 of 9352.

- June 23, 2026 17:59:58 CDT — Step A4 embedding checkpoint saved after batch 52; completed=2600 of 9352.

- June 23, 2026 17:59:59 CDT — Step A4 embedding checkpoint saved after batch 53; completed=2650 of 9352.

- June 23, 2026 18:00:00 CDT — Step A4 embedding checkpoint saved after batch 54; completed=2700 of 9352.

- June 23, 2026 18:00:02 CDT — Step A4 embedding checkpoint saved after batch 55; completed=2750 of 9352.

- June 23, 2026 18:00:03 CDT — Step A4 embedding checkpoint saved after batch 56; completed=2800 of 9352.

- June 23, 2026 18:00:05 CDT — Step A4 embedding checkpoint saved after batch 57; completed=2850 of 9352.

- June 23, 2026 18:00:06 CDT — Step A4 embedding checkpoint saved after batch 58; completed=2900 of 9352.

- June 23, 2026 18:00:07 CDT — Step A4 embedding checkpoint saved after batch 59; completed=2950 of 9352.

- June 23, 2026 18:00:09 CDT — Step A4 embedding checkpoint saved after batch 60; completed=3000 of 9352.

- June 23, 2026 18:00:10 CDT — Step A4 embedding checkpoint saved after batch 61; completed=3050 of 9352.

- June 23, 2026 18:00:12 CDT — Step A4 embedding checkpoint saved after batch 62; completed=3100 of 9352.

- June 23, 2026 18:00:13 CDT — Step A4 embedding checkpoint saved after batch 63; completed=3150 of 9352.

- June 23, 2026 18:00:15 CDT — Step A4 embedding checkpoint saved after batch 64; completed=3200 of 9352.

- June 23, 2026 18:00:16 CDT — Step A4 embedding checkpoint saved after batch 65; completed=3250 of 9352.

- June 23, 2026 18:00:18 CDT — Step A4 embedding checkpoint saved after batch 66; completed=3300 of 9352.

- June 23, 2026 18:00:19 CDT — Step A4 embedding checkpoint saved after batch 67; completed=3350 of 9352.

- June 23, 2026 18:00:21 CDT — Step A4 embedding checkpoint saved after batch 68; completed=3400 of 9352.

- June 23, 2026 18:00:22 CDT — Step A4 embedding checkpoint saved after batch 69; completed=3450 of 9352.

- June 23, 2026 18:00:24 CDT — Step A4 embedding checkpoint saved after batch 70; completed=3500 of 9352.

- June 23, 2026 18:00:25 CDT — Step A4 embedding checkpoint saved after batch 71; completed=3550 of 9352.

- June 23, 2026 18:00:27 CDT — Step A4 embedding checkpoint saved after batch 72; completed=3600 of 9352.

- June 23, 2026 18:00:28 CDT — Step A4 embedding checkpoint saved after batch 73; completed=3650 of 9352.

- June 23, 2026 18:00:29 CDT — Step A4 embedding checkpoint saved after batch 74; completed=3700 of 9352.

- June 23, 2026 18:00:30 CDT — Step A4 embedding checkpoint saved after batch 75; completed=3750 of 9352.

- June 23, 2026 18:00:32 CDT — Step A4 embedding checkpoint saved after batch 76; completed=3800 of 9352.

- June 23, 2026 18:00:33 CDT — Step A4 embedding checkpoint saved after batch 77; completed=3850 of 9352.

- June 23, 2026 18:00:34 CDT — Step A4 embedding checkpoint saved after batch 78; completed=3900 of 9352.

- June 23, 2026 18:00:36 CDT — Step A4 embedding checkpoint saved after batch 79; completed=3950 of 9352.

- June 23, 2026 18:00:38 CDT — Step A4 embedding checkpoint saved after batch 80; completed=4000 of 9352.

- June 23, 2026 18:00:39 CDT — Step A4 embedding checkpoint saved after batch 81; completed=4050 of 9352.

- June 23, 2026 18:00:41 CDT — Step A4 embedding checkpoint saved after batch 82; completed=4100 of 9352.

- June 23, 2026 18:00:43 CDT — Step A4 embedding checkpoint saved after batch 83; completed=4150 of 9352.

- June 23, 2026 18:00:44 CDT — Step A4 embedding checkpoint saved after batch 84; completed=4200 of 9352.

- June 23, 2026 18:00:46 CDT — Step A4 embedding checkpoint saved after batch 85; completed=4250 of 9352.

- June 23, 2026 18:00:47 CDT — Step A4 embedding checkpoint saved after batch 86; completed=4300 of 9352.

- June 23, 2026 18:00:49 CDT — Step A4 embedding checkpoint saved after batch 87; completed=4350 of 9352.

- June 23, 2026 18:00:50 CDT — Step A4 embedding checkpoint saved after batch 88; completed=4400 of 9352.

- June 23, 2026 18:00:51 CDT — Step A4 embedding checkpoint saved after batch 89; completed=4450 of 9352.

- June 23, 2026 18:00:53 CDT — Step A4 embedding checkpoint saved after batch 90; completed=4500 of 9352.

- June 23, 2026 18:00:54 CDT — Step A4 embedding checkpoint saved after batch 91; completed=4550 of 9352.

- June 23, 2026 18:00:56 CDT — Step A4 embedding checkpoint saved after batch 92; completed=4600 of 9352.

- June 23, 2026 18:00:58 CDT — Step A4 embedding checkpoint saved after batch 93; completed=4650 of 9352.

- June 23, 2026 18:00:59 CDT — Step A4 embedding checkpoint saved after batch 94; completed=4700 of 9352.

- June 23, 2026 18:01:00 CDT — Step A4 embedding checkpoint saved after batch 95; completed=4750 of 9352.

- June 23, 2026 18:01:02 CDT — Step A4 embedding checkpoint saved after batch 96; completed=4800 of 9352.

- June 23, 2026 18:01:03 CDT — Step A4 embedding checkpoint saved after batch 97; completed=4850 of 9352.

- June 23, 2026 18:01:04 CDT — Step A4 embedding checkpoint saved after batch 98; completed=4900 of 9352.

- June 23, 2026 18:01:06 CDT — Step A4 embedding checkpoint saved after batch 99; completed=4950 of 9352.

- June 23, 2026 18:01:07 CDT — Step A4 embedding checkpoint saved after batch 100; completed=5000 of 9352.

- June 23, 2026 18:01:09 CDT — Step A4 embedding checkpoint saved after batch 101; completed=5050 of 9352.

- June 23, 2026 18:01:10 CDT — Step A4 embedding checkpoint saved after batch 102; completed=5100 of 9352.

- June 23, 2026 18:01:11 CDT — Step A4 embedding checkpoint saved after batch 103; completed=5150 of 9352.

- June 23, 2026 18:01:13 CDT — Step A4 embedding checkpoint saved after batch 104; completed=5200 of 9352.

- June 23, 2026 18:01:14 CDT — Step A4 embedding checkpoint saved after batch 105; completed=5250 of 9352.

- June 23, 2026 18:01:16 CDT — Step A4 embedding checkpoint saved after batch 106; completed=5300 of 9352.

- June 23, 2026 18:01:17 CDT — Step A4 embedding checkpoint saved after batch 107; completed=5350 of 9352.

- June 23, 2026 18:01:18 CDT — Step A4 embedding checkpoint saved after batch 108; completed=5400 of 9352.

- June 23, 2026 18:01:20 CDT — Step A4 embedding checkpoint saved after batch 109; completed=5450 of 9352.

- June 23, 2026 18:01:21 CDT — Step A4 embedding checkpoint saved after batch 110; completed=5500 of 9352.

- June 23, 2026 18:01:23 CDT — Step A4 embedding checkpoint saved after batch 111; completed=5550 of 9352.

- June 23, 2026 18:01:25 CDT — Step A4 embedding checkpoint saved after batch 112; completed=5600 of 9352.

- June 23, 2026 18:01:26 CDT — Step A4 embedding checkpoint saved after batch 113; completed=5650 of 9352.

- June 23, 2026 18:01:28 CDT — Step A4 embedding checkpoint saved after batch 114; completed=5700 of 9352.

- June 23, 2026 18:01:29 CDT — Step A4 embedding checkpoint saved after batch 115; completed=5750 of 9352.

- June 23, 2026 18:01:31 CDT — Step A4 embedding checkpoint saved after batch 116; completed=5800 of 9352.

- June 23, 2026 18:01:33 CDT — Step A4 embedding checkpoint saved after batch 117; completed=5850 of 9352.

- June 23, 2026 18:01:34 CDT — Step A4 embedding checkpoint saved after batch 118; completed=5900 of 9352.

- June 23, 2026 18:01:35 CDT — Step A4 embedding checkpoint saved after batch 119; completed=5950 of 9352.

- June 23, 2026 18:01:37 CDT — Step A4 embedding checkpoint saved after batch 120; completed=6000 of 9352.

- June 23, 2026 18:01:38 CDT — Step A4 embedding checkpoint saved after batch 121; completed=6050 of 9352.

- June 23, 2026 18:01:39 CDT — Step A4 embedding checkpoint saved after batch 122; completed=6100 of 9352.

- June 23, 2026 18:01:41 CDT — Step A4 embedding checkpoint saved after batch 123; completed=6150 of 9352.

- June 23, 2026 18:01:42 CDT — Step A4 embedding checkpoint saved after batch 124; completed=6200 of 9352.

- June 23, 2026 18:01:43 CDT — Step A4 embedding checkpoint saved after batch 125; completed=6250 of 9352.

- June 23, 2026 18:01:45 CDT — Step A4 embedding checkpoint saved after batch 126; completed=6300 of 9352.

- June 23, 2026 18:01:46 CDT — Step A4 embedding checkpoint saved after batch 127; completed=6350 of 9352.

- June 23, 2026 18:01:47 CDT — Step A4 embedding checkpoint saved after batch 128; completed=6400 of 9352.

- June 23, 2026 18:01:49 CDT — Step A4 embedding checkpoint saved after batch 129; completed=6450 of 9352.

- June 23, 2026 18:01:50 CDT — Step A4 embedding checkpoint saved after batch 130; completed=6500 of 9352.

- June 23, 2026 18:01:52 CDT — Step A4 embedding checkpoint saved after batch 131; completed=6550 of 9352.

- June 23, 2026 18:01:54 CDT — Step A4 embedding checkpoint saved after batch 132; completed=6600 of 9352.

- June 23, 2026 18:01:55 CDT — Step A4 embedding checkpoint saved after batch 133; completed=6650 of 9352.

- June 23, 2026 18:01:57 CDT — Step A4 embedding checkpoint saved after batch 134; completed=6700 of 9352.

- June 23, 2026 18:01:58 CDT — Step A4 embedding checkpoint saved after batch 135; completed=6750 of 9352.

- June 23, 2026 18:01:59 CDT — Step A4 embedding checkpoint saved after batch 136; completed=6800 of 9352.

- June 23, 2026 18:02:01 CDT — Step A4 embedding checkpoint saved after batch 137; completed=6850 of 9352.

- June 23, 2026 18:02:02 CDT — Step A4 embedding checkpoint saved after batch 138; completed=6900 of 9352.

- June 23, 2026 18:02:04 CDT — Step A4 embedding checkpoint saved after batch 139; completed=6950 of 9352.

- June 23, 2026 18:02:05 CDT — Step A4 embedding checkpoint saved after batch 140; completed=7000 of 9352.

- June 23, 2026 18:02:06 CDT — Step A4 embedding checkpoint saved after batch 141; completed=7050 of 9352.

- June 23, 2026 18:02:08 CDT — Step A4 embedding checkpoint saved after batch 142; completed=7100 of 9352.

- June 23, 2026 18:02:09 CDT — Step A4 embedding checkpoint saved after batch 143; completed=7150 of 9352.

- June 23, 2026 18:02:11 CDT — Step A4 embedding checkpoint saved after batch 144; completed=7200 of 9352.

- June 23, 2026 18:02:12 CDT — Step A4 embedding checkpoint saved after batch 145; completed=7250 of 9352.

- June 23, 2026 18:02:13 CDT — Step A4 embedding checkpoint saved after batch 146; completed=7300 of 9352.

- June 23, 2026 18:02:15 CDT — Step A4 embedding checkpoint saved after batch 147; completed=7350 of 9352.

- June 23, 2026 18:02:16 CDT — Step A4 embedding checkpoint saved after batch 148; completed=7400 of 9352.

- June 23, 2026 18:02:17 CDT — Step A4 embedding checkpoint saved after batch 149; completed=7450 of 9352.

- June 23, 2026 18:02:19 CDT — Step A4 embedding checkpoint saved after batch 150; completed=7500 of 9352.

- June 23, 2026 18:02:21 CDT — Step A4 embedding checkpoint saved after batch 151; completed=7550 of 9352.

- June 23, 2026 18:02:22 CDT — Step A4 embedding checkpoint saved after batch 152; completed=7600 of 9352.

- June 23, 2026 18:02:24 CDT — Step A4 embedding checkpoint saved after batch 153; completed=7650 of 9352.

- June 23, 2026 18:02:25 CDT — Step A4 embedding checkpoint saved after batch 154; completed=7700 of 9352.

- June 23, 2026 18:02:26 CDT — Step A4 embedding checkpoint saved after batch 155; completed=7750 of 9352.

- June 23, 2026 18:02:28 CDT — Step A4 embedding checkpoint saved after batch 156; completed=7800 of 9352.

- June 23, 2026 18:02:30 CDT — Step A4 embedding checkpoint saved after batch 157; completed=7850 of 9352.

- June 23, 2026 18:02:31 CDT — Step A4 embedding checkpoint saved after batch 158; completed=7900 of 9352.

- June 23, 2026 18:02:32 CDT — Step A4 embedding checkpoint saved after batch 159; completed=7950 of 9352.

- June 23, 2026 18:02:34 CDT — Step A4 embedding checkpoint saved after batch 160; completed=8000 of 9352.

- June 23, 2026 18:02:35 CDT — Step A4 embedding checkpoint saved after batch 161; completed=8050 of 9352.

- June 23, 2026 18:02:37 CDT — Step A4 embedding checkpoint saved after batch 162; completed=8100 of 9352.

- June 23, 2026 18:02:38 CDT — Step A4 embedding checkpoint saved after batch 163; completed=8150 of 9352.

- June 23, 2026 18:02:40 CDT — Step A4 embedding checkpoint saved after batch 164; completed=8200 of 9352.

- June 23, 2026 18:02:42 CDT — Step A4 embedding checkpoint saved after batch 165; completed=8250 of 9352.

- June 23, 2026 18:02:43 CDT — Step A4 embedding checkpoint saved after batch 166; completed=8300 of 9352.

- June 23, 2026 18:02:45 CDT — Step A4 embedding checkpoint saved after batch 167; completed=8350 of 9352.

- June 23, 2026 18:02:46 CDT — Step A4 embedding checkpoint saved after batch 168; completed=8400 of 9352.

- June 23, 2026 18:02:48 CDT — Step A4 embedding checkpoint saved after batch 169; completed=8450 of 9352.

- June 23, 2026 18:02:49 CDT — Step A4 embedding checkpoint saved after batch 170; completed=8500 of 9352.

- June 23, 2026 18:02:51 CDT — Step A4 embedding checkpoint saved after batch 171; completed=8550 of 9352.

- June 23, 2026 18:02:52 CDT — Step A4 embedding checkpoint saved after batch 172; completed=8600 of 9352.

- June 23, 2026 18:02:53 CDT — Step A4 embedding checkpoint saved after batch 173; completed=8650 of 9352.

- June 23, 2026 18:02:55 CDT — Step A4 embedding checkpoint saved after batch 174; completed=8700 of 9352.

- June 23, 2026 18:02:56 CDT — Step A4 embedding checkpoint saved after batch 175; completed=8750 of 9352.

- June 23, 2026 18:02:58 CDT — Step A4 embedding checkpoint saved after batch 176; completed=8800 of 9352.

- June 23, 2026 18:02:59 CDT — Step A4 embedding checkpoint saved after batch 177; completed=8850 of 9352.

- June 23, 2026 18:03:01 CDT — Step A4 embedding checkpoint saved after batch 178; completed=8900 of 9352.

- June 23, 2026 18:03:11 CDT — FAILED at Step A4: RuntimeError: embedding batch 179 failed after 3 retries; completed=8900

- June 23, 2026 18:04:16 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 18:05:02 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 18:05:03 CDT — Step A3 resumed BERTScore checkpoint with 9352 completed rows.

- June 23, 2026 18:05:04 CDT — Step A3 complete. BERTScore similarity mean=0.818225, std=0.046888, min=0.497630, max=1.000000. routing_score mean=0.313849, std=0.136595, min=0.050000, max=0.950000. structural_boost_applied=77 rows (0.82%). Final row count=9352.

- June 23, 2026 18:05:04 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 18:05:04 CDT — Step A4 resumed embeddings checkpoint with 8900 completed rows.

- June 23, 2026 18:05:05 CDT — Step A4 embedding checkpoint saved after batch 179; completed=8950 of 9352.

- June 23, 2026 18:05:07 CDT — Step A4 embedding checkpoint saved after batch 180; completed=9000 of 9352.

- June 23, 2026 18:05:08 CDT — Step A4 embedding checkpoint saved after batch 181; completed=9050 of 9352.

- June 23, 2026 18:05:10 CDT — Step A4 embedding checkpoint saved after batch 182; completed=9100 of 9352.

- June 23, 2026 18:05:11 CDT — Step A4 embedding checkpoint saved after batch 183; completed=9150 of 9352.

- June 23, 2026 18:05:12 CDT — Step A4 embedding checkpoint saved after batch 184; completed=9200 of 9352.

- June 23, 2026 18:05:13 CDT — Step A4 embedding checkpoint saved after batch 185; completed=9250 of 9352.

- June 23, 2026 18:05:15 CDT — Step A4 embedding checkpoint saved after batch 186; completed=9300 of 9352.

- June 23, 2026 18:05:17 CDT — Step A4 embedding checkpoint saved after batch 187; completed=9350 of 9352.

- June 23, 2026 18:05:17 CDT — Step A4 embedding checkpoint saved after batch 188; completed=9352 of 9352.

- June 23, 2026 18:05:19 CDT — Step A4 complete. Total clusters=500, kept=487, discarded=13, mean_confidence_weight=0.487822, final row count after cluster filter=9349.

- June 23, 2026 18:05:20 CDT — Phase B MF epoch 1 weighted MSE loss=0.118170.

- June 23, 2026 18:05:20 CDT — Phase B MF epoch 5 weighted MSE loss=0.115459.

- June 23, 2026 18:05:20 CDT — Phase B MF epoch 10 weighted MSE loss=0.115483.

- June 23, 2026 18:05:21 CDT — Phase B MF epoch 20 weighted MSE loss=0.115951.

- June 23, 2026 18:06:13 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 18:07:00 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 18:07:01 CDT — Step A3 resumed BERTScore checkpoint with 9352 completed rows.

- June 23, 2026 18:07:01 CDT — Step A3 using complete BERTScore checkpoint with 9352 rows; scorer construction skipped.

- June 23, 2026 18:07:01 CDT — Step A3 complete. BERTScore similarity mean=0.818225, std=0.046888, min=0.497630, max=1.000000. routing_score mean=0.313849, std=0.136595, min=0.050000, max=0.950000. structural_boost_applied=77 rows (0.82%). Final row count=9352.

- June 23, 2026 18:07:01 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 18:07:01 CDT — Step A4 embeddings loaded from final cache: (9352, 768).

- June 23, 2026 18:07:03 CDT — Step A4 complete. Total clusters=500, kept=487, discarded=13, mean_confidence_weight=0.487822, final row count after cluster filter=9349.

- June 23, 2026 18:07:03 CDT — Phase B MF epoch 1 weighted MSE loss=0.118439.

- June 23, 2026 18:07:04 CDT — Phase B MF epoch 5 weighted MSE loss=0.115351.

- June 23, 2026 18:07:04 CDT — Phase B MF epoch 10 weighted MSE loss=0.115312.

- June 23, 2026 18:07:05 CDT — Phase B MF epoch 20 weighted MSE loss=0.115321.

- June 23, 2026 18:08:52 CDT — Step A1 complete. Tier assignments: strong=10, medium=8, weak=41, skip=5. Raw battle tier counts: Tier1vTier2=3621, Tier1vTier3=14192, discarded=39664.

- June 23, 2026 18:09:39 CDT — Step A2 complete. Row counts: raw=57477, after clear winners=39716 (dropped 17761), after response length=37377 (dropped 2339), after refusal filter=34650 (dropped 2727), after length ratio=30664 (dropped 3986), after English=29333 (dropped 1331), after tier matchup filter=9352 (dropped 19981).

- June 23, 2026 18:09:39 CDT — Step A3 resumed BERTScore checkpoint with 9352 completed rows.

- June 23, 2026 18:09:39 CDT — Step A3 using complete BERTScore checkpoint with 9352 rows; scorer construction skipped.

- June 23, 2026 18:09:40 CDT — Step A3 complete. BERTScore similarity mean=0.818225, std=0.046888, min=0.497630, max=1.000000. routing_score mean=0.313849, std=0.136595, min=0.050000, max=0.950000. structural_boost_applied=77 rows (0.82%). Final row count=9352.

- June 23, 2026 18:09:40 CDT — Step A4 Ollama reachable at http://127.0.0.1:11435; available tags response received.

- June 23, 2026 18:09:40 CDT — Step A4 embeddings loaded from final cache: (9352, 768).

- June 23, 2026 18:09:42 CDT — Step A4 complete. Total clusters=500, kept=487, discarded=13, mean_confidence_weight=0.487822, final row count after cluster filter=9349.

- June 23, 2026 18:09:42 CDT — Phase B MF epoch 1 weighted MSE loss=0.118439.

- June 23, 2026 18:09:43 CDT — Phase B MF epoch 5 weighted MSE loss=0.115351.

- June 23, 2026 18:09:43 CDT — Phase B MF epoch 10 weighted MSE loss=0.115312.

- June 23, 2026 18:09:44 CDT — Phase B MF epoch 20 weighted MSE loss=0.115321.

- June 23, 2026 18:09:44 CDT — Phase B complete. MF train MSE=0.115321, val MSE=0.120065, mean validation nn_confidence=0.644601.

## Phase C — Arena held-out test evaluation

| Metric | MF | SW | Ensemble |
|---|---:|---:|---:|
| Spearman correlation | 0.164716 | 0.116324 | 0.089199 |
| ROC-AUC (score > 0.5 = cloud) | 0.677786 | 0.585715 | 0.555871 |
| Brier score | 0.052632 | 0.115986 | 0.071248 |
| Precision @ threshold 0.40 | 0.000000 | 0.166667 | 1.000000 |
| Recall @ threshold 0.40 | 0.000000 | 0.045455 | 0.022727 |
| Precision @ threshold 0.50 | 0.000000 | 0.000000 | 0.000000 |
| Recall @ threshold 0.50 | 0.000000 | 0.000000 | 0.000000 |
| Precision @ threshold 0.60 | 0.000000 | 0.000000 | 0.000000 |
| Recall @ threshold 0.60 | 0.000000 | 0.000000 | 0.000000 |

Compared against v1 ROC-AUC 0.5203 and v2 ROC-AUC 0.6296. Test size: 836. Test positive rate: 0.052632.

- June 23, 2026 18:11:08 CDT — Phase C MMLU progress: evaluated 50 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:13:22 CDT — Phase C MMLU progress: evaluated 100 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:15:17 CDT — Phase C MMLU progress: evaluated 150 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:16:51 CDT — Phase C MMLU progress: evaluated 200 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:18:37 CDT — Phase C MMLU progress: evaluated 250 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:20:04 CDT — Phase C MMLU progress: evaluated 300 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:21:47 CDT — Phase C MMLU progress: evaluated 350 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:23:23 CDT — Phase C MMLU progress: evaluated 400 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:25:17 CDT — Phase C MMLU progress: evaluated 450 of 500 questions with weak model `qwen3:8b`.

- June 23, 2026 18:27:26 CDT — Phase C MMLU progress: evaluated 500 of 500 questions with weak model `qwen3:8b`.

## Phase C — MMLU validation evaluation

Sample size: 500

Weak model: `qwen3:8b`

Routed accuracy: 0.00%

Over-routing rate: 0.00%

Under-routing rate: 100.00%

- June 23, 2026 18:27:26 CDT — FAILED at Phase C: JSONDecodeError: Unterminated string starting at: line 1 column 12 (char 11)

---
## Final summary

**Completed:** June 23, 2026 18:28:15 CDT
**Status:** Target not met

**Dataset**
- Raw battles loaded: 57477
- Surviving after all filters: 9352
- Surviving after cluster filter: 9349
- Tier1vTier2 battles: 1935
- Tier1vTier3 battles: 7417

**Routing score distribution**
- Mean: 0.313833   Std: 0.136510   Min: 0.050000   Max: 0.950000

**Model performance**
| Metric | MF | SW | Ensemble |
|---|---:|---:|---:|
| Spearman correlation | 0.164716 | 0.116324 | 0.089199 |
| ROC-AUC (score > 0.5 = cloud) | 0.677786 | 0.585715 | 0.555871 |
| Brier score | 0.052632 | 0.115986 | 0.071248 |
| Precision @ threshold 0.40 | 0.000000 | 0.166667 | 1.000000 |
| Recall @ threshold 0.40 | 0.000000 | 0.045455 | 0.022727 |
| Precision @ threshold 0.50 | 0.000000 | 0.000000 | 0.000000 |
| Recall @ threshold 0.50 | 0.000000 | 0.000000 | 0.000000 |
| Precision @ threshold 0.60 | 0.000000 | 0.000000 | 0.000000 |
| Recall @ threshold 0.60 | 0.000000 | 0.000000 | 0.000000 |

**MMLU routing accuracy:** 0.00%
**Over-routing rate:** 0.00%
**Under-routing rate:** 100.00%

**vs baselines**
- v1 ROC-AUC: 0.5203  →  v3: 0.555871
- v2 ROC-AUC: 0.6296  →  v3: 0.555871

**Artifacts**
- data/arena-preference-v3.jsonl
- models/local-cloud-classifier-v3-mf.pt
- models/local-cloud-sw-index-v3.faiss
- models/local-cloud-sw-data-v3.npy
- models/local-cloud-classifier-v3-ensemble.pkl
- data/local-cloud-classifier-v3-evaluation.json

**Assessment**
The probabilistic ensemble did not meaningfully outperform the prior binary classifier on held-out Arena ROC-AUC: v2 was 0.6296 and v3 ensemble is 0.555871. Because v3 predicts a continuous routing score instead of a hard binary label, Spearman correlation and Brier score are also important indicators of whether the ranking is useful.

The MMLU sample used local model `qwen3:8b` and produced routed accuracy 0.00%, over-routing 0.00%, and under-routing 100.00%. This out-of-distribution result indicates whether Arena preference data transfers to exam-style questions; if under-routing remains material or ROC-AUC is below target, Source A alone is not sufficient and Source B augmentation remains required before production routing.
---
