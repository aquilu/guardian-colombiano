**Summary.** With fp16 mixed precision (`fp16=True`, the only option on T4 GPUs), `SafeGradHFTrainer` combines a
**scaled** task gradient with an **unscaled** alignment gradient. After the Trainer unscales, the alignment term is
effectively divided by the loss scale `s` (65,536 at the start of training), so SafeGrad silently behaves almost like
plain SFT.

**Where.** `src/safetune/interventions/harden/safegrad.py` @ 3901a9f:
- `training_step`: HF `Trainer.training_step` runs `loss.backward()` through the accelerator's `GradScaler`, so
  `p.grad = s · g_user`.
- `g_align` comes from `torch.autograd.grad(...)` on the KL loss (around L253) and is **not** scaled.
- `_apply_global_surgery` (around L269–315) runs the conflict test, the projection and
  `g_final = g'_user + rho * g_align` on these mixed units, and writes the result back to `.grad`. The optimizer step
  then unscales everything by `1/s`, so the update becomes `g'_user + (rho/s) · g_align`.

**How we found it.** We added an instrumented diagnostic (12 steps, Tiny Aya Global 3.35B, LoRA r=16, 2×T4, fp16).
`scaler.get_scale()` = 65,536, and after unscaling the alignment contribution was a small fraction of what the formula
intends.

**Fix (minimal, exact).** The projection is homogeneous of degree 1 in `g_user` and degree 0 in `g_align`, so passing
`s · g_align` gives `s · (g'_user + rho · g_align)`. After the single unscale that is exactly Eq. 3–5 of the paper:

```python
def _apply_global_surgery(self, model, g_align):
    scaler = getattr(self.accelerator, "scaler", None)
    if scaler is not None and scaler.is_enabled():
        s = scaler.get_scale()
        g_align = [g * s for g in g_align]
    ...  # unchanged
```

We checked it numerically: the homogeneity error `‖g'(s·u)/s − g'(u)‖` was 0 on 60 steps, and the relative alignment
contribution became O(1).

**Suggestion.** Apply the scale (or unscale `.grad` first via `accelerator.unscale_gradients()`), and add a test that
runs one step with fp16 and one with fp32 and compares `g_final`. bf16 is unaffected because it has no scaler.

Found while building a safety-preserving moderation guardian for the Lexsi AI Safety Hackathon:
https://github.com/aquilu/guardian-colombiano (see `kaggle/codigo/entrenar_safetune.py`, `corregir_amp`).
