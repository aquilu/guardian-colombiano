---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 b74449bb589601c2083a8503a9bff47d3850b003fb362dcf3bd7e908d7b292f1
  data_files:
  - split: train
    path: sft_alpaca.jsonl
---

# Dataset Card: guardian_colombiano_curado

## Summary

- **Run timestamp**: 2026-11-04T19:06:46.954100Z
- **Wall clock time**: 11.1s
- **Pipeline config hash**: `2c4ff0e7e2ac07d1`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 1185 | 1185 | — | 0 | 1185 |
| TextCleaner | 1185 | 1185 | — | — | 1185 |
| ExactDeduplicator | 1185 | 1185 | — | — | 1185 |
| MinHashDeduplicator | 1185 | 1185 | — | — | 1185 |
| EmbeddingDeduplicator | 1185 | 1181 | — | — | 1181 |
| AlpacaExporter | — | 1181 | — | — | 1181 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 1185/0 passed (N/A), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 1185

## Known Limitations

- MinHash threshold used: **0.8** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.12.13

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (1181) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
