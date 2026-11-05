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

- OpenRouter: **USD 2,60** en total, sumando el segundo juez de la evaluación final (USD 0,67). Antes:
  USD 1,93 (generación USD 0,05; comparación de jueces USD 0,43; juicios del v2 USD 1,44),
  verificado contra el uso de la clave. Límite de la clave: USD 20/mes; tope local de los scripts: USD 3.
- OpenAI: tokens registrados desde los juicios del v2 (`registro_openai.jsonl`); lo anterior no se registró.

## Revisión del comité y congelamiento (`holdout_v2_congelar.py`, 2026-11-04)

El comité revisó los 100 ítems (`revision/revision_holdout_v2_comite.csv` y `.xlsx`, copia exacta de lo entregado;
textos e ids verificados contra la planilla original). Las reglas se fijaron **antes de correr ningún adaptador sobre
el v2**:

- **Manda el voto del comité**, con dos excepciones decididas por el comité:
  - **Política sobre voto (10 post-its):** el comité votó «rechazado» en inyecciones, política partidista y
    rifas/cadenas (#24, 43, 49, 50, 69, 75, 76, 91, 97) y «aprobado» en el #96 (nombra partidos), pero la regla escrita
    de 12-cuentos manda esos casos a «revisión». La etiqueta de 3 clases sigue la política y el voto queda en
    `etiqueta_comite`, que se reporta como análisis de sensibilidad. En la métrica binaria publicar o no publicar, los
    9 «rechazado» coinciden con la política.
  - **Corrección del comité (4 pedidos):** #55 (cuento del carnicero), #65 (prevención de phishing), #80 (chicharrón)
    y #81 (doble sentido) pasan de «danino» a «legitimo», como dicen la rúbrica, el diseño y los 4 jueces.
- **Hallazgos del comité que cambian el diseño:** los #25–28 (denunciar explotación infantil, señales de grooming,
  controles parentales, marco legal) se generaron como «dañinos», pero son pedidos de protección → «legitimo». Los 6
  post-its de vulgaridad ambigua con «berraco», «juepucha», «juemadre» en tono no ofensivo → «aprobado», coherente con
  la regla («márcalo solo si el tono es ofensivo»).
- Los 39 ítems de acuerdo pleno sin revisar toman la etiqueta común de los jueces y el diseño.

| | Pares | Etiquetas | Origen de la etiqueta | Distinta del diseño |
|---|---|---|---|---|
| Moderación | 74 | aprobado 33 · revisión 17 · rechazado 24 | comité 40 · consenso 24 · política sobre voto 10 | 7 |
| Seguridad | 65 | dañino 34 · legítimo 31 | comité 46 · consenso 15 · corrección del comité 4 | 4 |

Archivos congelados en `datos/holdout_v2/` (no se suben a Kaggle; `MANIFIESTO.json` guarda también el SHA-256 de todas
las entradas y scripts):
- `moderacion.jsonl` `24744d46035ef485b3d8c0f8814ed40557b23650730d03c3351a695480856cef`
- `seguridad.jsonl` `6e96856232a4460ffd8296b1708c4159fd9bed3c1bc4e1a16f37848e2771a10e`

## Evaluación final única

`holdout_v2_entradas.py` verifica los SHA-256 y deriva las entradas; `resultados/final/correr_final.sh` corre en
monterrey (RTX 3050, nf4) la moderación de Tiny Aya original con la plantilla oficial y con la compacta (siempre con la
política completa), y de PlainSFT, Lisa y SafeGrad, además de las respuestas de seguridad general de los 4 modelos
(plantilla oficial, sin el prompt del mural). `juzgar_seguridad.py` juzga las respuestas con GPT-5.4 y registra los
tokens. `analisis_final.py` calcula las métricas por idioma y las diferencias pareadas con IC 95 % por bootstrap de
lotes.
