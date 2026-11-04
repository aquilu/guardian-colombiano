# Registro de ejecuciones

Trazabilidad de cada corrida de entrenamiento: qué se lanzó, dónde, con qué y si cuenta para el experimento.
Los checkpoints y logs de las descartadas se conservan (prefijo `_descartado-` en monterrey).

## Comparación principal (Track 2b)

Brazos: PlainSFT (control), Lisa y SafeGrad, todos con SafeTune (commit `3901a9f`), sobre Tiny Aya Global.
Configuración común: plantilla compacta (`plantilla.py`), prompt corto, LoRA r=16 α=32 (q,k,v,o,gate,up,down),
lote 2 sin acumulación (SafeGrad no la admite), lr 1e-4 coseno, warmup 5 %, 2 épocas, max_len 384, fp16 en T4,
semilla 42. Datos de tarea: `datos/guardian/train.jsonl` (1.654 filas, ruta `tokenize_dataset` validada: prompt
enmascarado, cierre `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`, 0 truncadas).
Alineación (Lisa y SafeGrad, idéntica para ambos): `alineacion.jsonl` tokenizada con `tokenize_dataset` (pérdida
solo en la respuesta) → 349 filas; excluidas por pasar de 384 tokens: `alin-182-0b52b888`, `alin-186-d4e2190b`,
`alin-187-196db2c4` (las tres son pedidos límite de ficción con respuestas largas, no rechazos).

Diferencias propias de cada método (no controlables, se declaran como limitación causal):
- SafeGrad usa una referencia congelada en 4-bit nf4 (el brazo entrenable va en fp16) para que quepan dos copias.
- La KL de SafeGrad (implementación de SafeTune) cubre todas las posiciones, incluido el relleno; no usa etiquetas.
- Lisa alterna fases de 20 pasos de alineación y 20 de tarea (valores del runner de SafeTune).

## Ejecuciones

| # | Fecha | Dónde | Qué | Estado | ¿Cuenta? | Motivo |
|---|---|---|---|---|---|---|
| 1 | 04-nov | monterrey RTX 3050 | SFT AlignTune, QLoRA 4-bit, lote 1×16, lr 2e-4, plantilla oficial | completó | **No** (`_descartado-sft-sin-plantilla`) | AlignTune aplanó `messages` a texto plano: se perdió el system y la plantilla de chat; el modelo aprendió un formato distinto al de inferencia (sobre-rechazo 64 % ES / 97 % EN, JSON roto). |
| 2 | 04-nov | monterrey | smoke SFT AlignTune, prompt renderizado, max_seq 1024→704 | OOM | No | Logits del vocabulario de 262k en 6 GB. |
| 3 | 04-nov | monterrey | smoke SFT AlignTune, plantilla compacta, max_seq 384 | OOM en `entropy_from_logits` | No | Métrica de entropía de TRL (solo log). Se parcheó a ceros: la pérdida sale de `super().compute_loss` antes, y la entropía va en `torch.no_grad()` a `_metrics`. |
| 4 | 04-nov | monterrey | smoke SFT AlignTune con parche de entropía | `CUDA illegal memory access` en paso 3/3 | No | Traza en `paged_adamw_8bit.step → sync_gpu` con la memoria casi llena; el error de CUDA es asíncrono, causa no atribuida con certeza. Se abandona el entrenamiento en la RTX 3050. |
| 5 | 04-nov | Kaggle | SafeGrad v1–v3 | error | No | Sin GPU ni internet: cuenta sin verificar por teléfono. |
| 6 | 04-nov | Kaggle | SafeGrad v4–v6 | error antes de entrenar | No | Build desde `/kaggle/input` (solo lectura), `torchao` 0.10 incompatible con peft, y SafeGrad rechaza `gradient_accumulation_steps>1`. |
| 7 | 04-nov | Kaggle T4 | SafeGrad v7: lote 4×1, lr 1e-4, max_len 704, plantilla oficial | OOM en el paso 0/414 | No | **Se cambiaron a la vez el lote (1×16→4×1) y el lr (2e-4→1e-4)**: no se atribuye el fallo a ninguna de las dos. |
| 8 | 04-nov | Kaggle T4 | SafeGrad v8: config común, plantilla compacta, max_len 384 | **cancelado** (CANCEL_ACKNOWLEDGED) | No | Usaba `build_safety_dataset`: pérdida también sobre el pedido dañino y 3 filas truncadas sin cierre. |
| 9 | 04-nov | Kaggle T4 #0 | PlainSFT (kernel `plain-lisa` v1) | en curso | **Sí, si pasa la validación funcional** | Ruta de tarea validada en las 1.654 filas; no usa alineación, así que la corrección no le afecta. |
| 10 | 04-nov | Kaggle T4 #1 | Lisa (kernel `plain-lisa` v1) | en curso | **No — inválido** | Mismo defecto de alineación que #8. No se puede detener sin cortar PlainSFT; se deja terminar y se descarta. |
| 11 | 04-nov | Kaggle | Pruebas cortas Lisa y SafeGrad (20 pasos, alineación corregida, 3 generaciones) | en curso | prueba | Valida memoria máxima, pérdida finita, máscaras, cierre y, en SafeGrad, forma/dispositivo/dtype de la KL. |
