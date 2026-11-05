# Regional variants of Tiny Aya (run #22)

Suggested by a Lexsi mentor (Discord ticket-43): *does the base model's regional focus change the outcome?*
Pre-registered in `EJECUCIONES.md` (#22) **before** training. Lisa was trained with the frozen configuration of
`guardian-lisa-global` (same data, alignment pairs, hyperparameters and seed, nothing tuned per variant) on:

- **`tiny-aya-water`**: focused on European and Asia-Pacific languages, Spanish included (the in-region candidate).
- **`tiny-aya-fire`**: focused on South Asian languages (an off-region control).

Before training, the kernel checked that both variants' `tokenizer.json` and `config.json` are byte-identical to
`global`'s. Kaggle 2× T4, about 38 min each. Each variant, original and + Lisa, was evaluated **once** on the frozen
holdout v2 with the same scripts as run #20. The global numbers are the ones from #20 and were not recomputed.
Judge: GPT-5.4. CIs: paired cluster bootstrap (10,000 replicates). `*` marks a 95 % CI that excludes 0.
Script: `analisis_variantes.py`; results in `resultados/final/analisis_variantes.json`.

## Results (ES / EN)

| Model | Accuracy | Unsafe notes slip through | Approved notes blocked | Exact schema | ES/EN agreement | ASR (harmful help) | Over-refusal |
|---|---|---|---|---|---|---|---|
| global original | 62.2 / 68.9 % | 46.3 / 41.5 % | 3.0 / 0.0 % | 0 / 7 % | 77.0 % | 50.0 / 58.8 % | 0 / 0 % |
| water original | 54.0 / 56.8 % | 68.3 / 51.2 % | 0.0 / 0.0 % | 100 / 100 % | 81.1 % | 50.0 / 47.1 % | 0 / 0 % |
| fire original | 56.8 / 58.1 % | 19.5 / 17.1 % | 15.2 / 12.1 % | 51 / 88 % | 83.8 % | 47.1 / 47.1 % | 0 / 0 % |
| **global + Lisa** | **81.1 / 82.4 %** | 9.8 / 7.3 % | 15.2 / 18.2 % | 99 / 99 % | **95.9 %** | **0 / 0 %** | 3.2 / 6.5 % |
| water + Lisa | 81.1 / 75.7 % | 9.8 / 14.6 % | 12.1 / 18.2 % | 99 / 97 % | 91.9 % | **0 / 0 %** | 3.2 / 0 % |
| fire + Lisa | 78.4 / 78.4 % | 7.3 / 4.9 % | 21.2 / 27.3 % | 93 / 96 % | 89.2 % | **0 / 0 %** | 0 / 3.2 % |

## Paired differences (vs global + Lisa)

- **water + Lisa:**
  - Spanish: accuracy identical (+0.0, CI −4 to +4); 3 points less over-blocking (not significant).
  - English: **−6.8 points accuracy\*** (CI −14 to −1) and 7 points more unsafe notes slip through (CI 0 to +19).
- **fire + Lisa:**
  - accuracy −2.7 points ES and −4.1 points EN (both not significant);
  - over-blocking +6 points ES and +9 points EN (CIs touch 0).

## Reading

1. **Lisa's safety effect is robust to the base variant.** In all three bases, ASR goes from about 50 % to **0 % in
   both languages**. Each drop is significant: −47 to −50 points, CI upper bound ≤ −29.
2. **Regional focus moves the task only a little, and not where expected.**
   - water, the in-region model, ties global in Spanish after Lisa but loses 7 points in English.
   - fire, the off-region control, over-blocks more.
   - The pre-registered hypothesis "water ≥ global in ES" holds only as a tie. "fire ≤ global in ES" holds as a
     point estimate, without significance.
3. **Untuned bases differ in style, not in safety.**
   - water original follows the JSON schema perfectly, but lets far more unsafe notes through: +22 points in ES\*.
   - fire original is stricter: −27 points ES\* in slip-through, at the cost of over-blocking.
   - Fine-tuning erases most of those differences.
4. **We keep global + Lisa** as the guardian: it has the best ES/EN consistency (96 %) and the best English
   accuracy, and it is never worse in Spanish.

**Limitations:**
- one seed per variant;
- 74 moderation and 65 safety items per language;
- GPT-5.4 is the only judge for the variants (Claude judged the global models only);
- the variants' adapters are not published as separate models.
