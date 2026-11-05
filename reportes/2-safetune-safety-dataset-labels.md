**Summary.** `build_safety_dataset` supervises the **whole sequence**: the chat template, the user prompt and the
response. Only padding is set to `-100`. In the Harden trainers this auxiliary set is made of *harmful prompts* paired
with refusals, so the alignment loss also trains the model to **generate the harmful request**, and the refusal
signal is diluted by the template and prompt tokens.

**Where.** `src/safetune/runner/utils/data_utils.py` @ 3901a9f, `build_safety_dataset` (around L365–392):

```python
labels = [tid if m == 1 else -100
          for tid, m in zip(enc["input_ids"], enc["attention_mask"])]
```

Compare with `_tokenize_qa_rows` in the same file, which correctly masks `prompt_len` tokens and supervises only the
last assistant turn.

**Fix.** Build the set with `_tokenize_qa_rows(tok, rows, max_len)`, the same code path as the task data. We did
exactly that (`tokenize_dataset` on `(prompt, response)` pairs). We also excluded and logged rows whose response would
be fully truncated by `max_len`, because `_tokenize_qa_rows` already warns about them.

Repro: build the set with any chat tokenizer and check `sum(l != -100 for l in labels)` against the number of
response tokens.

Context: https://github.com/aquilu/guardian-colombiano (`alineacion_corregida` in `kaggle/codigo/entrenar_safetune.py`).
