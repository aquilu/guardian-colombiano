**Summary.** `_tokenize_qa_rows` assumes `tokenizer.padding_side == "right"`. With a left-padding tokenizer it masks
the wrong span and **supervises the prompt**. Training runs fine and the loss goes down, so nothing signals the
problem.

**Where.** `src/safetune/runner/utils/data_utils.py` @ 3901a9f, around L165–176:

```python
enc = tok(text, truncation=True, max_length=max_len, padding="max_length", add_special_tokens=False)
prompt_len = _prompt_len(tok, msgs)
for i in range(len(labels)):
    if i < prompt_len or enc["attention_mask"][i] == 0:
        labels[i] = -100
```

With left padding the first `n_pad` positions are padding. The mask `i < prompt_len` then covers pad plus only the
first `prompt_len − n_pad` prompt tokens, so the rest of the prompt (and the template) is supervised.

**How we hit it.** The Kaggle copy of Tiny Aya Global (`cohere-labs/tiny-aya/transformers/global/1`) ships a
tokenizer config with left padding. Our first Lisa run trained on prompt tokens until a token-level label check
caught it.

**Fix.** Either set `tok.padding_side = "right"` inside the function (and restore it afterwards), or compute the
masked span from the real position of the first non-pad token. An assertion that the first supervised token equals
the first response token would also catch it.
