**Summary.** For SFT on chat data (`messages`), AlignTune converts the conversation to plain `Human:` / `Assistant:`
text:
- it drops the `system` turn;
- it does not use the tokenizer's chat template.

The model is then trained on a format different from the one used at inference. In our case (Tiny Aya Global, whose
template and system prompt carry the task policy), the fine-tuned model behaved like the base model at inference.

**Where.** `src/aligntune/core/dataset_adapters.py` @ c3d648f, `adapt_dataset` (around L204–222):

```python
for msg in messages:
    role = msg.get("role", ""); content = msg.get("content", "")
    if role == "user":       conversation_text += f"Human: {content}\n"
    elif role == "assistant": conversation_text += f"Assistant: {content}\n"
```

`role == "system"` is silently ignored, and `tokenizer.apply_chat_template` is never used.

**Suggestion.**
- For chat-format SFT, keep `messages` and let TRL's `SFTTrainer` apply the chat template (it supports
  conversational datasets and assistant-only loss).
- Or render with `tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)` as the prompt and the
  last assistant turn as the completion.
- At minimum, warn when a `system` turn is dropped.

Workaround we used: pre-render the prompt with the chat template, since AlignTune leaves an already-templated
`prompt` untouched. Later we moved training to SafeTune. Context: https://github.com/aquilu/guardian-colombiano
