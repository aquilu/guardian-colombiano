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

Corrección común (desde la ejecución #12): `tok.padding_side = "right"` explícito. SafeTune
(`_tokenize_qa_rows`) enmascara las primeras `prompt_len` posiciones y supone relleno a la derecha; el tokenizer de
Kaggle Models con transformers 5.20 rellena a la izquierda por defecto. La verificación `tokenizar_verificado`
rechaza cualquier fila con prompt o relleno supervisado o sin cierre.

Tokenizer y config comunes (desde la ejecución #14): los de HF en la revisión `af89d21` (`hf_tiny_aya/` en el
dataset). Los pesos de Kaggle Models son idénticos byte a byte a los de HF (SHA-256 de ambos `.safetensors`), pero su
`tokenizer_config.json` declara `CohereTokenizerFast`, que con transformers 5.20 segmenta distinto (293 frente a
276 tokens en la misma fila: dígitos, guiones y saltos de línea en más piezas) y rellena a la izquierda; y su
`config.json` trae `max_position_embeddings=500000` (HF: 8192). El script se niega a entrenar si los IDs de la primera
fila de la tarea y de la alineación no son idénticos a `tokens_esperados.json`.

SafeGrad — corrección de compatibilidad con AMP y ρ calibrado (desde la ejecución #18). **No es una reproducción
idéntica de la configuración del paper** (arXiv:2508.07172): el paper usa ρ = 1 por defecto (barrido 0,5–100) y no
especifica precisión ni la reducción de la KL.
- Corrección AMP: en fp16 el `.grad` de la tarea viene multiplicado por la escala s del GradScaler y `g_align` no.
  Se pasa s·g_align a `_apply_global_surgery` de SafeTune. Justificación: la proyección (Eq. 4) es homogénea de grado
  1 en g_user y de grado 0 en g_align, así que g'(s·u) = s·g'(u) y no cambia al reescalar a; el combinado queda
  s·(g'(u) + ρ·a) y el Trainer lo desescala una sola vez → exactamente Eq. 5. Verificado numéricamente (error 0 con
  la corrección; 73 % sin ella en un caso sintético).
- ρ = 4,283 × 10⁻⁴ (congelado en `kaggle/rho_safegrad.json`): ρ = mediana(‖g_user‖/‖g_align‖) en los 12 pasos del
  diagnóstico #15 (lotes de train y alineación; el test no se usó). Objetivo: contribución de alineación ≈ norma de la
  tarea (1:1 en mediana; 0,35–2,62 en esos 12 pasos).
- La KL de SafeTune se conserva tal cual (suma sobre todas las posiciones). La normalización por token queda como
  ablación posterior, separada de esta corrección.

Diferencias propias de cada método (no controlables, se declaran como limitación causal):
- SafeGrad usa una referencia congelada en 4-bit nf4 (el brazo entrenable va en fp16) para que quepan dos copias.
- La KL de SafeGrad (implementación de SafeTune) cubre todas las posiciones, incluido el relleno; no usa etiquetas.
- Lisa alterna fases de 20 pasos de alineación y 20 de tarea (valores del runner de SafeTune): con el mismo total de
  pasos dedica ~la mitad a la tarea. Una peor exactitud del guardián en Lisa puede deberse a eso y no al endurecimiento.
- SafeGrad: en la prueba corta, logits del estudiante en float32 y de la referencia en float16 (misma forma
  [2, 384, 262144], mismo dispositivo); KL del orden de 240–260 por la reducción `batchmean` sobre todas las posiciones.

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
| 9 | 04-nov | Kaggle T4 #0 | PlainSFT (kernel `plain-lisa` v1) | en curso | **No — inválido** | La validación de la ruta de tarea se hizo en monterrey (transformers 5.22, relleno a la derecha). En Kaggle (transformers 5.20, tokenizer de Kaggle Models sin `padding_side`) el relleno va a la izquierda; SafeTune enmascara las primeras `prompt_len` posiciones y dejaba gran parte del prompt dentro de la pérdida. |
| 10 | 04-nov | Kaggle T4 #1 | Lisa (kernel `plain-lisa` v1) | en curso | **No — inválido** | Mismo defecto de alineación que #8. No se puede detener sin cortar PlainSFT; se deja terminar y se descarta. |
| 11 | 04-nov | Kaggle | Pruebas cortas Lisa y SafeGrad v1 | error controlado | prueba | La verificación de máscaras detuvo ambas antes de entrenar: prompt dentro de la pérdida por el relleno a la izquierda (ver #9). |
| 12 | 04-nov | Kaggle | Pruebas cortas v2: SafeGrad (T4 #0), Lisa → PlainSFT (T4 #1); 20 pasos, 3 generaciones | en curso | prueba | `padding_side="right"` forzado; tarea y alineación pretokenizadas y verificadas fila por fila (prompt y relleno en -100, cierre supervisado). |
| 13 | 04-nov | Kaggle | Pruebas cortas v2 (20 pasos): SafeGrad, Lisa, PlainSFT | completó | prueba | Máscaras OK en los tres; pérdidas finitas; memoria máx. SafeGrad 13,9 GB, Lisa 9,1, PlainSFT 8,8; JSON y esquema exactos en SafeGrad y PlainSFT. Lisa (solo alineación en 20 pasos) aún sin formato. **Se detectó que el tokenizer de Kaggle segmenta distinto al de HF** → se repiten. |
| 13b | 04-nov | Kaggle | Prueba Lisa 60 pasos (tokenizer de Kaggle) | completó | prueba | Pérdida 1,86→1,03 al entrar a la tarea; JSON compacto con claves inventadas tras 20 pasos de tarea. Solo funcional. |
| 14 | 04-nov | Kaggle | Pruebas cortas v3 (60 pasos) con tokenizer y config de HF y comprobación de IDs | completó | prueba | Tokens idénticos a la referencia (276/72) y máscaras OK en las 1.654 filas de tarea y 349 de alineación en los tres. Pérdidas finitas. Memoria máx.: PlainSFT 8,8 GB, Lisa 9,1, SafeGrad 13,9 (igual que a 20 pasos). JSON/esquema: PlainSFT 3/3, SafeGrad 3/3, **Lisa 1/3 JSON, 0/3 esquema** (solo ~20 pasos de tarea). **Pérdidas de SafeGrad = PlainSFT al tercer decimal en los 6 registros.** |
| 15 | 04-nov | Kaggle | Diagnóstico de la cirugía de SafeGrad (12 pasos, instrumentación apagada por defecto) | completó | diagnóstico | **Defecto comprobado:** en fp16 el `.grad` de la tarea está multiplicado por la escala del GradScaler (65.536) y `g_align` (de `torch.autograd.grad`) no; SafeTune los suma tal cual y el Trainer desescala después, así que `g_align` llega dividido por 65.536: aporta entre 1,2 % y 9,3 % de la norma de la tarea. En unidades reales ‖g_align‖ = 1.267–10.185 frente a ‖g_user‖ = 1,1–1,8. Conflicto (proyección) en 8/12 pasos. SafeGrad tal cual ≈ PlainSFT. |
| 16 | 04-nov | Kaggle T4 | **PlainSFT definitivo** (kernel `plain-final`), código validado en #14 | en curso | **Sí** (control) | Configuración congelada; tokenizer HF `af89d21`; sin cambios respecto a la prueba v3. |
| 17 | 04-nov | Kaggle T4 #1 | Prueba Lisa 200 pasos con diagnóstico de fases (estados por paso, generación en cada cambio de fase) | en curso | prueba | Verificar que conserva el esquema JSON tras alternar tarea y alineación; contar pasos de cada fase. |
| 18 | 04-nov | Kaggle T4 #0 | Prueba SafeGrad 60 pasos con corrección AMP, ρ = 4,283e-4 y diagnóstico completo | en curso | prueba | Coseno, proyección, homogeneidad medida, contribución relativa de la alineación, pérdida, memoria y JSON. |

