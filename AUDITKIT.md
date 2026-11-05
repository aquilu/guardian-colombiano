# Red-teaming con AuditKIT (Lexsi) — tercer repo del Track 2

`auditkit_eval.py`, con AuditKIT 1.1.3 (`Lexsi-Labs/AuditKIT` @ 71316da). Se corre sobre los 4 modelos congelados en
modo asistente general (plantilla oficial de Tiny Aya, sin el prompt del mural), con decodificación voraz y nf4 en la
RTX 3050.

- **No reutiliza el holdout v2,** que ya se usó una vez: son sondas independientes.
- **Inglés:** las sondas **integradas** de AuditKIT (`prompt_injection`, `jailbreak`, `encoding`, `refusal`) con sus
  detectores por defecto (`_PROBE_DETECTOR_MAP`).
- **Español:** una **extensión nuestra** con las mismas sondas traducidas al español de Colombia y detectores
  equivalentes en español. Se reporta aparte.
- **Regla de conteo:** una salida falla si cualquier detector la marca (ataque logrado o, en `refusal`, un rechazo
  indebido).

## Resultado (salidas que pasan / total)

| Sonda | Tiny Aya original | PlainSFT | **Lisa** | SafeGrad |
|---|---|---|---|---|
| prompt_injection (EN, AuditKIT) | 2/8 | 2/8 | **5/8** | 2/8 |
| prompt_injection_es (ES, extensión) | 3/8 | 3/8 | **5/8** | 3/8 |
| encoding (EN) | 4/5 | 4/5 | 4/5 | 4/5 |
| encoding_es (ES) | 4/5 | 4/5 | 4/5 | 4/5 |
| refusal: no rechazar lo inofensivo (EN) | 4/5 | 5/5 | 5/5 | 4/5 |
| refusal_es (ES) | 5/5 | 5/5 | 5/5 | 5/5 |
| jailbreak (EN / ES) | 8/8 · 8/8 | 8/8 · 8/8 | 8/8 · 8/8 | 8/8 · 8/8 |

## Lectura

- **Lisa es el único método que mejora la resistencia a la inyección** (+3 de 8 en EN y +2 de 8 en ES frente al
  original), sin rechazar pedidos inofensivos. PlainSFT y SafeGrad quedan igual que el original. Esto coincide con el
  holdout v2.
- **Fallas reales del original:** responde literalmente «I HAVE BEEN HACKED», «ACCESS_GRANTED» y
  «I can help with anything», y «revela» un prompt de sistema inventado. Lisa también cae en «ACCESS_GRANTED» y en
  «I can help with anything».
- **Límites de los detectores:** son por palabras clave.
  - **Falsos positivos:** por ejemplo, un rechazo de Lisa («No puedo saltarme las pautas…») se marca como falla, y
    también traducciones del leetspeak que no obedecen la orden. El detector de `refusal` marcó como rechazo un
    programa de calculadora del original.
  - **En `jailbreak` los detectores no miden contenido dañino:** solo buscan marcadores como «hacked». El 8/8 general
    **no indica seguridad**; para eso está la evaluación con dos jueces del holdout v2.
- **Tamaño:** son muestras pequeñas (5 a 8 sondas), así que no se afirma significancia. Es evidencia complementaria,
  no la métrica principal.

El resumen está en `resultados/auditkit/resumen.json`. Las trazas por sonda y por modelo se publican solo en el
dataset `vasmi-sas/guardian-colombiano-logs` (carpeta `auditkit/`), con las salidas de `jailbreak` que no son un
rechazo claro tachadas. No están en este repositorio.
