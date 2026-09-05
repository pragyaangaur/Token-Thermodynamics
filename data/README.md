# Data

The files here are the derived results that back the claims in FINDINGS.md. They are small enough to commit so that `repro.py` and the analysis scripts run without downloading any model.

| File | What it holds |
| --- | --- |
| `multimodel_melt3.json` | Melting temperature and energy gap for 7 pretrained models on 20 shared prompts. Read by `repro.py`. |
| `vocab_exp.json` | The 5 tiny models trained with only the vocabulary size changed. Read by `repro.py`. |
| `meltlaw_1p5b.json` | Melting temperature against top-k truncation, 10 prompts by 10 truncation levels. |
| `melt_1p5b.json` | Melting temperature at every token position of four passages. |
| `scales_1p5b.json` | Within-meaning and between-meaning logit spreads, 500 questions. |
| `where_meaning.json` | Position-resolved meaning share, 480 token positions. |
| `where_meaning_thr.json` | The same measurement with every forked continuation saved, so it can be reclustered at other thresholds. |
| `freeform_1p5b.json`, `freeform_0p5b.json` | Phrasing against meaning, free-form and short factual, on both models. |
| `ffh.json`, `ffh_0p5b.json` | Free-form hallucination detection across known, obscure and fabricated conditions. |
| `meaning_fe.json` | Whether greedy decoding picks the highest-mass meaning. |
| `sem_1p5b.json` | Reference semantic entropy on 1100 short factual questions. |
| `wikidata_raw.json` | Ground truth for 3096 questions in four relation types, from Wikidata SPARQL. |
| `fake_entities.json` | The 1000 generated non-existent entities. |

Three raw probe outputs are not committed because they total about 170 MB: `main_1p5b.json`, `main_0p5b.json` and `fake_1p5b.json`. Each holds a 300 point thermodynamic curve for every question. The README explains how to regenerate them.
