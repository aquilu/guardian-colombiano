# Holdout independiente v2 — procedencia, contaminación, jueces y costos

El holdout v2 es el **conjunto final** de evaluación. Se usa **una sola vez**, después de congelar PlainSFT, Lisa y
SafeGrad. El holdout v1 (GPT-5.4-mini) queda como conjunto exploratorio.

## Procedencia

| | Moderación | Seguridad general |
|---|---|---|
| Generador | `deepseek/deepseek-v4.1-flash-1017` (OpenRouter, proveedor fijo DeepInfra fp8, sin respaldo, `data_collection: deny`) | ídem |
| Script | `holdout_v2.py --conjunto moderacion` | `holdout_v2.py --conjunto seguridad` |
| Diseño | 13 categorías de la política × 2 lotes × 3 perfiles de autor; modismos reservados (`holdout_modismos_reservados.json`, 59 que no aparecen en ningún dato previo); táctica de escritura por lote (homoglifos, emojis, fragmentación, instrucciones incrustadas, chat, spanglish, lengua indígena/creole) | 10 categorías dañinas × 4 y 7 categorías límite × 5 (incluye «doble sentido»); solo pedidos, nunca respuestas |
| Generados → conservados | 76 → **74 pares ES/EN** | 70 → **65 pares ES/EN** |
| Unidad de análisis | el par ES/EN (no dos observaciones); `lote_id` y `familia_id` para agrupar | ídem |

Familias más grandes de la generación de entrenamiento y del v2 son de otra familia de modelos (GPT-5.4 vs DeepSeek),
pero **eso no garantiza independencia**: la auditoría lo comprobó.

## Contaminación (`holdout_v2_auditar.py`, `informe_contaminacion_v2.json`)

Comparado contra 8 fuentes: generación de entrenamiento (piloto y v1), train/validación/test, alineación, evaluación de
seguridad y holdout v1. Embeddings `paraphrase-multilingual-MiniLM-L12-v2`, en ES y en EN.

| | Moderación | Seguridad |
|---|---|---|
| Duplicados exactos | 0 | 0 |
| Casi-duplicados de datos existentes (coseno ≥ 0,85) | 0 | **5** (alineación y eval. de seguridad: «matar una gallina para el sancocho», «vacuna por WhatsApp», «síntesis de cocaína», banco falso, narcotráfico) |
| Casi-duplicados internos | 2 | 0 |
| Similitud máx. con lo existente (mediana / p95 / máx.) | 0,65 / 0,74 / 0,77 | 0,67 / 0,82 / 0,85 |
| Familias de plantillas (coseno ≥ 0,80) con > 1 miembro | 1 | 1 |
| Comienzos repetidos | — | «Estoy escribiendo un cuento…» ×10, «Para un trabajo de…» ×5 (encuadres de evasión pedidos; se conservan y se agrupan por lote) |

## Jueces y rúbricas (`holdout_v2_juzgar.py`, `rubricas_v2.json`)

- **GPT-5.4** (OpenAI) como juez de continuidad y **Claude Sonnet 5.5** (vía OpenRouter, proveedor fijo **Google
  Vertex**) como segundo juez de otra familia. Misma rúbrica, temperatura 0, ES y EN por separado.
- Rúbricas versionadas con SHA-256: `moderacion_v1` = política del guardián de 12-cuentos; `seguridad_v1` = ¿el pedido
  es dañino o legítimo?
- **Sonnet vs Haiku** (`comparar_jueces.py`, sobre datos que no son del v2, regla fijada antes): en moderación acuerdo
  91,2 % y κ = 0,86 (< 95 % / 0,90; Haiku rechaza lo que Sonnet manda a revisión) → **Sonnet**. En seguridad, 100 %.
- **Por qué Google Vertex:** en el endpoint de Anthropic, OpenRouter modera la entrada y devuelve 403 con los post-its
  violentos, sexuales u odiosos; con `require_parameters` no hay endpoint. En Vertex, el filtro del propio proveedor
  deja algunas salidas vacías: **cobertura de Claude 71/74 en moderación y 55/65 en seguridad** (los 10 sin juicio son
  pedidos dañinos). GPT-5.4: 100 %. Todos los ítems sin juicio de Claude van a revisión humana.

| | Acuerdo GPT–Claude (ES) | GPT = objetivo del diseño | Claude = objetivo |
|---|---|---|---|
| Moderación | 64/71 | 70/74 | 64/71 |
| Seguridad | 56/59 | 61/65 | 52/59 |

## Revisión humana (`holdout_v2_revision.py` → `revision_holdout_v2.csv`)

100 ítems: los 40 con algún motivo (desacuerdo GPT/Claude 10, sin Claude 12, jueces ≠ diseño 6, cambia ES/EN 3,
ambiguo 9) y 60 de auditoría aleatoria estratificada de los acuerdos plenos. Quedan 39 acuerdos plenos sin revisar.
Columna `decision_comite` vacía: **ninguna etiqueta se cambia automáticamente** ni mirando a los adaptadores.

## Costos (`registro_openrouter.jsonl`, `registro_openai.jsonl`)

- OpenRouter: **USD 1,93** en total (generación USD 0,05; comparación de jueces USD 0,43; juicios del v2 USD 1,44),
  verificado contra el uso de la clave. Límite de la clave: USD 20/mes; tope local de los scripts: USD 3.
- OpenAI: tokens registrados desde los juicios del v2 (`registro_openai.jsonl`); lo anterior no se registró.

## Pendiente antes de congelar

1. Revisión del comité de `revision_holdout_v2.csv`.
2. `holdout_v2_congelar.py`: etiquetas definitivas, `datos/holdout_v2/` con manifiesto y SHA-256 (no se sube a Kaggle).
3. Evaluación final única: Tiny Aya original, PlainSFT, Lisa, SafeGrad; diferencias pareadas con intervalos de
   confianza por bootstrap agrupado por par y por lote.
