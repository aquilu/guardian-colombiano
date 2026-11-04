"""Fine-tuning SFT del guardián con AlignTune. Variante «plana», sin endurecimiento (el control del Track 2b).

Configuración común con Lisa y SafeGrad (para que la única diferencia sea el endurecimiento):
  LoRA r=16 α=32, lote 2 sin acumulación (SafeGrad no la admite), lr 1e-4 coseno, 2 épocas, plantilla compacta,
  pérdida solo en la respuesta. En Kaggle (T4): fp16 sin cuantizar.
Uso: python entrenar_sft.py --modelo <ruta o id> --salida <dir> [--dtype fp16|bf16] [--cuantizar no|4bit]
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import trl.trainer.sft_trainer as _sft
from transformers import AutoTokenizer

from aligntune.core.backend_factory import create_sft_trainer
from plantilla import compacta

# TRL calcula la entropía sobre el vocabulario completo (262k tokens) solo como métrica de log; se apaga para
# ahorrar memoria. La pérdida y los gradientes no cambian.
_sft.entropy_from_logits = lambda logits, chunk_size=128: torch.zeros(logits.shape[:-1], device=logits.device)


def preparar(datos: str, salida: Path, tok) -> Path:
    """AlignTune (c3d648f) aplana «messages» a prompt/completion en texto plano: pierde el system y no aplica la
    plantilla de chat. Por eso se le entrega el prompt ya renderizado (lo deja intacto). TRL agrega el BOS al
    tokenizar y el EOS (<|END_OF_TURN_TOKEN|>) al final de la completion."""
    plano = salida / "train_prompt_completion.jsonl"
    salida.mkdir(parents=True, exist_ok=True)
    with open(datos, encoding="utf-8") as f, open(plano, "w", encoding="utf-8") as g:
        for l in f:
            msgs = json.loads(l)["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            g.write(json.dumps({"prompt": prompt.removeprefix(tok.bos_token),
                                "completion": msgs[-1]["content"] + "<|END_RESPONSE|>"}, ensure_ascii=False) + "\n")
    ej = json.loads(open(plano, encoding="utf-8").readline())
    msgs = json.loads(open(datos, encoding="utf-8").readline())["messages"]
    if tok.bos_token + ej["prompt"] + ej["completion"] + tok.eos_token != tok.apply_chat_template(msgs, tokenize=False):
        raise SystemExit("el formato de entrenamiento no coincide con la plantilla de inferencia")
    print("plantilla idéntica: True")
    return plano


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--datos", default="datos/guardian/train.jsonl")
    ap.add_argument("--salida", default="modelos/guardian-sft-global")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--acumulacion", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--cuantizar", choices=["no", "4bit"], default="no")
    ap.add_argument("--max-pasos", type=int, default=-1)
    ap.add_argument("--max-muestras", type=int, default=None)
    a = ap.parse_args()

    tok = compacta(AutoTokenizer.from_pretrained(a.modelo))
    plano = preparar(a.datos, Path(a.salida), tok)
    cuant = None
    if a.cuantizar == "4bit":
        cuant = {"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4",
                 "bnb_4bit_compute_dtype": "bfloat16" if a.dtype == "bf16" else "float16"}
    trainer = create_sft_trainer(
        model_name=a.modelo, dataset_name=str(plano.resolve()), backend="trl", output_dir=a.salida,
        task_type="instruction_following", num_epochs=a.epocas, max_steps=a.max_pasos,
        batch_size=a.lote, gradient_accumulation_steps=a.acumulacion, learning_rate=a.lr,
        warmup_ratio=0.05, lr_scheduler_type="cosine", max_grad_norm=0.3, optim="paged_adamw_8bit",
        max_seq_length=384,          # con la plantilla compacta la secuencia más larga mide 304 tokens
        max_samples=a.max_muestras, split="train", completion_only_loss=True,
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules="all-linear",
        quantization=cuant, bf16=a.dtype == "bf16", fp16=a.dtype == "fp16", gradient_checkpointing=True,
        save_strategy="no", logging_steps=10, report_to="none", seed=42,
    )
    resultado = trainer.train()
    Path(a.salida, "resultado_entrenamiento.json").write_text(json.dumps(
        resultado if isinstance(resultado, dict) else {"training_loss": getattr(resultado, "training_loss", None)},
        indent=2, default=str))
    tok.save_pretrained(a.salida)   # guarda la plantilla compacta junto al adaptador
    Path(a.salida, "config_sft.json").write_text(json.dumps(vars(a), indent=2))
    print("listo →", a.salida)


if __name__ == "__main__":
    main()
