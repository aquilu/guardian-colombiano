"""Fine-tuning SFT (QLoRA) del guardián con AlignTune. Esta es la variante «plana», sin endurecimiento.

Uso: .venv-train/bin/python entrenar_sft.py --modelo CohereLabs/tiny-aya-global --salida modelos/guardian-sft-global
Entrena solo sobre la respuesta del asistente (assistant_only_loss) con datos/guardian/train.jsonl.
Pensado para 6 GB de VRAM (RTX 3050): 4-bit nf4, LoRA r=16, lote 1 × acumulación 16, gradient checkpointing.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
from pathlib import Path

from aligntune.core.backend_factory import create_sft_trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--datos", default="datos/guardian/train.jsonl")
    ap.add_argument("--salida", default="modelos/guardian-sft-global")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-pasos", type=int, default=-1)
    ap.add_argument("--max-muestras", type=int, default=None)
    a = ap.parse_args()

    # AlignTune lee la columna «messages»; se deja un archivo solo con esa columna.
    plano = Path(a.salida) / "train_messages.jsonl"
    plano.parent.mkdir(parents=True, exist_ok=True)
    with open(a.datos, encoding="utf-8") as f, open(plano, "w", encoding="utf-8") as g:
        for l in f:
            g.write(json.dumps({"messages": json.loads(l)["messages"]}, ensure_ascii=False) + "\n")

    trainer = create_sft_trainer(
        model_name=a.modelo,
        dataset_name=str(plano.resolve()),
        backend="trl",
        output_dir=a.salida,
        task_type="instruction_following",
        num_epochs=a.epocas,
        max_steps=a.max_pasos,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=a.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        max_seq_length=704,      # la secuencia más larga del train mide 692 tokens
        max_samples=a.max_muestras,
        dataset_text_field="messages",
        split="train",
        completion_only_loss=True,   # AlignTune pasa prompt/completion a TRL: pérdida solo en la respuesta
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="all-linear",
        quantization={"load_in_4bit": True, "bnb_4bit_use_double_quant": True,
                      "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=5,
        report_to="none",
        seed=42,
    )
    resultado = trainer.train()
    Path(a.salida, "resultado_entrenamiento.json").write_text(json.dumps(
        resultado if isinstance(resultado, dict) else {"training_loss": getattr(resultado, "training_loss", None)},
        indent=2, default=str))
    trainer.save_model() if hasattr(trainer, "save_model") else None
    print("listo →", a.salida)


if __name__ == "__main__":
    main()
