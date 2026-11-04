"""Fine-tuning endurecido con SafeTune · Lisa (bi-state proximal: alterna lotes de la tarea y de alineación).

Mismos datos y presupuesto que entrenar_sft.py (QLoRA 4-bit, LoRA r=16, lote efectivo 16, 2 épocas), para que la
única diferencia sea el endurecimiento. La señal de alineación es datos/seguridad/alineacion.jsonl (ES/EN).
Uso: .venv-train/bin/python entrenar_lisa.py --salida modelos/guardian-lisa-global
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import safetune
from safetune.runner.harden import LisaTrainer


class LisaQLoRA(LisaTrainer):
    """Lisa sobre un modelo 4-bit: acumulación de gradiente y se guarda solo el adaptador (fusionar en 4-bit degrada)."""

    acumulacion = 16

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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--max-muestras", type=int, default=0)
    a = ap.parse_args()

    safetune.configure(datasets={"safety_refusals": str(Path(a.alineacion).resolve())}, max_len=704, seed=42,
                       lora_r=16, lora_alpha=32, lora_dropout=0.05,
                       lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    tok = AutoTokenizer.from_pretrained(a.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, quantization_config=q, device_map="cuda:0", dtype=torch.bfloat16)
    modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant": False})

    filas = [{"messages": json.loads(l)["messages"]} for l in open(a.datos, encoding="utf-8")]
    if a.max_muestras:
        filas = filas[: a.max_muestras]
    trainer = LisaQLoRA(modelo, tok, epochs=a.epocas, batch_size=1, lr=a.lr, bf16=True,
                        optimizer="paged_adamw_8bit", logging_steps=5, lisa_rho=a.rho)
    ruta = trainer.train(Dataset.from_list(filas), out_dir=a.salida)
    Path(a.salida, "config_lisa.json").write_text(json.dumps(vars(a), indent=2))
    print("listo →", ruta)


if __name__ == "__main__":
    main()
