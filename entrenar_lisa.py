"""Fine-tuning endurecido con SafeTune · Lisa (bi-state proximal: alterna lotes de la tarea y de alineación).

Misma configuración que entrenar_sft.py (LoRA r=16, lote 2 sin acumulación, lr 1e-4 coseno, 2 épocas, plantilla
compacta); la única diferencia es el endurecimiento. Señal de alineación: datos/seguridad/alineacion.jsonl (ES/EN).
Uso: python entrenar_lisa.py --modelo <ruta o id> --salida <dir> [--dtype fp16|bf16] [--cuantizar no|4bit]
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import safetune
from safetune.runner.harden import LisaTrainer
from plantilla import compacta


class LisaLoRA(LisaTrainer):
    """Lisa con la configuración común; guarda solo el adaptador (igual que el SFT)."""

    acumulacion = 1

    def _configure_args(self, config, out_dir):
        args = super()._configure_args(config, out_dir)
        args.gradient_accumulation_steps = self.acumulacion
        args.gradient_checkpointing = True
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.lr_scheduler_type = "cosine"
        args.warmup_ratio = 0.05
        args.max_grad_norm = 0.3
        args.save_strategy = "no"
        args.report_to = []
        return args

    def _save_merged(self, model, out_dir: str) -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--datos", default="datos/guardian/train.jsonl")
    ap.add_argument("--alineacion", default="datos/seguridad/alineacion.jsonl")
    ap.add_argument("--salida", default="modelos/guardian-lisa-global")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--acumulacion", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--cuantizar", choices=["no", "4bit"], default="no")
    ap.add_argument("--max-muestras", type=int, default=0)
    a = ap.parse_args()

    safetune.configure(datasets={"safety_refusals": str(Path(a.alineacion).resolve())}, max_len=384, seed=42,
                       lora_r=16, lora_alpha=32, lora_dropout=0.05,
                       lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    dtype = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    tok = compacta(AutoTokenizer.from_pretrained(a.modelo))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.cuantizar == "4bit":
        from peft import prepare_model_for_kbit_training
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True)
        modelo = AutoModelForCausalLM.from_pretrained(a.modelo, quantization_config=q, device_map="cuda:0", dtype=dtype)
        modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=True,
                                                 gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        modelo = AutoModelForCausalLM.from_pretrained(a.modelo, device_map="cuda:0", dtype=dtype)
        modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        modelo.enable_input_require_grads()

    filas = [{"messages": json.loads(l)["messages"]} for l in open(a.datos, encoding="utf-8")]
    if a.max_muestras:
        filas = filas[: a.max_muestras]
    LisaLoRA.acumulacion = a.acumulacion
    trainer = LisaLoRA(modelo, tok, epochs=a.epocas, batch_size=a.lote, lr=a.lr, bf16=a.dtype == "bf16",
                       fp16=a.dtype == "fp16", optimizer="paged_adamw_8bit", logging_steps=10, lisa_rho=a.rho)
    ruta = trainer.train(Dataset.from_list(filas), out_dir=a.salida)
    Path(a.salida, "config_lisa.json").write_text(json.dumps(vars(a), indent=2))
    print("listo →", ruta)


if __name__ == "__main__":
    main()
