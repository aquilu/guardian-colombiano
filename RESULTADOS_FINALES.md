# Resultados finales — holdout v2 congelado (evaluación única, 2026-11-04)

Holdout: `datos/holdout_v2/` (SHA-256 en `MANIFIESTO.json`), 74 pares de moderación y 65 pares de seguridad, ES y EN.
Inferencia: monterrey, RTX 3050, nf4 4-bit, decodificación voraz. Análisis: `analisis_final.py` →
`resultados/final/analisis_final.json`. Intervalos de confianza: IC 95 % por bootstrap de lotes (`lote_id`, 10.000
réplicas). «Significativo» quiere decir que el IC no incluye el 0. Las categorías con n < 10 son solo descriptivas.

## 1. Moderación del mural (tarea afinada)

Tiny Aya original con la política completa de 12-cuentos; los adaptadores con el prompt corto del entrenamiento.

| Sistema | Exactitud ES / EN | Publica lo indebido (se cuela) | Bloquea lo aprobado | Esquema exacto | Misma decisión ES = EN |
|---|---|---|---|---|---|
| Tiny Aya original, plantilla oficial | 62,2 % / 68,9 % | 46,3 % / 41,5 % | 3,0 % / 0,0 % | 0,0 % / 6,8 % | 77,0 % |
| Tiny Aya original, plantilla compacta | 59,5 % / 67,6 % | 65,8 % / 53,7 % | 0,0 % / 0,0 % | 0,0 % / 1,4 % | 86,5 % |
| PlainSFT | 81,1 % / 77,0 % | 7,3 % / 4,9 % | 21,2 % / 24,2 % | 91,9 % / 100 % | 87,8 % |
| Lisa | 81,1 % / 82,4 % | 9,8 % / 7,3 % | 15,2 % / 18,2 % | 98,7 % / 98,7 % | 95,9 % |
| SafeGrad | 79,7 % / 79,7 % | 7,3 % / 4,9 % | 24,2 % / 24,2 % | 100 % / 100 % | 91,9 % |

- Los tres adaptadores reducen entre 34 y 39 puntos la tasa de contenido indebido que se cuela, en ES y en EN. La
  reducción es significativa en los 6 casos. El original casi nunca entrega el esquema JSON exacto.
- Entre adaptadores no hay diferencias significativas. Todas las diferencias contra PlainSFT tienen un IC que
  incluye el 0.
- Costo: el sobre-bloqueo sube a 15–24 %, casi todo en «vulgaridad ambigua» (8–10 de 14 casos por modelo). El comité
  aprobó esos post-its («berraco», «juepucha» en tono no ofensivo), pero los datos de entrenamiento los etiquetaban
  como «revisión». Es una diferencia de criterio entre el diseño del entrenamiento y el comité, y no un error
  aleatorio.
- Sensibilidad: con el voto del comité como etiqueta (`exactitud_comite`), los adaptadores bajan 9–11 puntos
  (PlainSFT 70,3 / 66,2 %, Lisa 71,6 / 73,0 %, SafeGrad 68,9 / 68,9 %) y el original casi no cambia (58,1 / 67,6 %).
  En EN, PlainSFT queda por debajo del original (66,2 % frente a 67,6 %); Lisa sigue primera en los dos idiomas.

## 2. Seguridad general (deriva de seguridad; plantilla oficial, sin el prompt del mural)

Juez: GPT-5.4 sobre las 520 respuestas, con cobertura del 100 % (203.479 tokens de entrada y 13.520 de salida,
registrados en `registro_openai.jsonl`). 34 pedidos dañinos y 31 legítimos por idioma.

| Modelo | ASR ES / EN | Rechaza lo dañino | Rechaza lo legítimo | Útil en lo legítimo |
|---|---|---|---|---|
| Tiny Aya original | 50,0 % / 58,8 % | 38,2 % / 38,2 % | 0,0 % / 0,0 % | 80,6 % / 90,3 % |
| PlainSFT | 47,1 % / 38,2 % | 44,1 % / 41,2 % | 3,2 % / 0,0 % | 77,4 % / 83,9 % |
| **Lisa** | **0,0 % / 0,0 %** | **100 % / 100 %** | 3,2 % / 6,5 % | 77,4 % / 93,5 % |
| SafeGrad | 61,8 % / 47,1 % | 29,4 % / 41,2 % | 0,0 % / 3,2 % | 74,2 % / 90,3 % |

- **Lisa** reduce el ASR en 50 puntos en ES (IC −67 a −33) y en 59 en EN (IC −72 a −47) frente al original, y en 47 y
  38 puntos frente a PlainSFT, todo significativo. No pierde utilidad en lo legítimo: frente a PlainSFT, +9,7 puntos
  en EN (significativo) y 0 en ES. Revisión manual: rechaza con brevedad, da la línea 123 y ofrece alternativas.
  Efectos declarados: respuestas más cortas, voseo ocasional («Podés») y, en algunos rechazos, la frase de rechazo
  repetida.
- **PlainSFT** no degrada la seguridad del original. En EN la mejora (−20,6 puntos, IC −34 a −6), pero sigue dando
  ayuda dañina sobre todo cuando el pedido viene disfrazado de ficción.
- **SafeGrad** empeora en ES: ASR +11,8 puntos frente al original (IC +2,9 a +20,6) y +14,7 frente a PlainSFT (IC
  +5,9 a +23,5). En EN no hay diferencia significativa.
- Interpretación: SafeGrad solo *protege* la dirección de alineación (proyecta el gradiente de la tarea cuando choca
  con el de alineación, con ρ calibrado); no entrena sobre los rechazos. Lisa dedica la mitad de los pasos a entrenar
  directamente con los datos de alineación. Como el Tiny Aya original ya da ayuda dañina en el 50–59 % de estos
  pedidos, «preservar» su alineación no basta: hay que *instalarla*. Con ρ calibrado en solo 12 pasos de diagnóstico
  y la corrección AMP, SafeGrad tampoco logró preservarla en ES. Es un resultado de esta configuración, no del método
  en general.

## Conclusión para el Track 2b

En este caso (guardián del mural colombiano, Tiny Aya Global 3,35B, LoRA, 2× T4), **Lisa es el único método que
mantiene la tarea y mejora la seguridad**: exactitud de moderación igual a la de PlainSFT, ASR 0 % en ES y EN, sin
aumento significativo de rechazos a lo legítimo y con la mejor consistencia ES/EN (95,9 %). Su costo: la mitad de los
pasos son de alineación y un estilo de respuesta más corto.

## Límites

- Tamaños pequeños: 34 pedidos dañinos y 31 legítimos por idioma; los intervalos son anchos.
- Un solo juez (GPT-5.4) para las respuestas de seguridad; sumar Claude como segundo juez queda pendiente de
  presupuesto.
- Una sola semilla de entrenamiento por método.
- El ρ de SafeGrad se calibró con 12 pasos de diagnóstico (sin usar el test), y el método no es una reproducción
  idéntica del artículo.
