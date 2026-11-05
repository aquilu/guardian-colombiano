# ROI — guardián cerrado de 12-cuentos frente a Tiny Aya + Lisa

Mismo holdout v2 congelado: 74 pares, ES y EN. No se hicieron llamadas nuevas: `roi.py` recompone la cadena de
producción con las respuestas de omni-moderation y gpt-5.4-mini guardadas al etiquetar.
IC 95 % por bootstrap de lotes (`resultados/final/roi.json`).

## Calidad

| Sistema | Exactitud ES / EN | Publica lo indebido | Bloquea lo aprobado |
|---|---|---|---|
| **Cadena de producción** (reglas → omni-moderation → gpt-5.4-mini) | 82,4 % / 85,1 % | 2,4 % / 2,4 % | **33,3 %** / 24,2 % |
| gpt-5.4-mini solo (política completa) | 90,5 % / 87,8 % | 2,4 % / 2,4 % | 12,1 % / 18,2 % |
| Reglas + omni-moderation (sin LLM) | 48,6 % / 48,6 % | 53,7 % / 56,1 % | 27,3 % / 12,1 % |
| **Tiny Aya + Lisa** (local, 4 bits) | 81,1 % / 82,4 % | 9,8 % / 7,3 % | **15,2 %** / 18,2 % |
| Tiny Aya original | 62,2 % / 68,9 % | 46,3 % / 41,5 % | 3,0 % / 0,0 % |

**Lisa frente a la cadena de producción**
- **Exactitud:** sin diferencia significativa (−1,4 en ES, IC −15 a +11; −2,7 en EN).
- **Sobre-bloqueo:** Lisa bloquea **18 puntos menos** en ES (IC −31 a −8). Es justamente el problema que motivó el
  proyecto: las reglas locales mandan a revisión el habla coloquial colombiana.
- **Contenido que se cuela:** Lisa deja pasar algo más (+7,3 en ES y +4,9 en EN); los IC tocan el 0.

**Lisa frente a gpt-5.4-mini solo:** mini es mejor en la estimación puntual (+9,5 de exactitud en ES), pero ninguna
diferencia es significativa.

**Sesgo a favor de mini:** las etiquetas de referencia salen en parte de GPT-5.4, que es de la misma familia que mini
y usó la misma rúbrica.

## Costo, latencia y privacidad

| | Cadena de producción | Tiny Aya + Lisa |
|---|---|---|
| Costo por post-it | gpt-5.4-mini: 702 tokens de entrada y 109 de salida (medidos en 160 llamadas). A USD 0,75 y 4,50 por millón ≈ **USD 0,0010**. omni-moderation es gratuito. | Solo energía: 1,16 s por post-it en una RTX 3050 (lote de 8). Si se asumen 150 W de sistema y USD 0,22/kWh ≈ **USD 0,00001** |
| 1.000.000 de post-its | ≈ USD 1.020 | ≈ USD 11 de energía, más el hardware (una GPU de consumo de 6 GB) |
| Privacidad | El texto sale a OpenAI (dos servicios) | El texto nunca sale del servidor de la biblioteca |
| Dependencia | Clave de API, cuotas, cambios de modelo del proveedor | Pesos fijos y versionados (licencia CC-BY-NC, compatible con una red de bibliotecas sin fines de lucro) |
| Seguridad general del modelo | — | ASR 0 % en ES y EN con dos jueces (`RESULTADOS_FINALES.md`) |

**Lectura honesta:** con el volumen de un mural, el ahorro en dólares es pequeño (unos USD 1 por cada 1.000
post-its). El ROI de verdad está en tres puntos:
1. igual exactitud con la mitad del sobre-bloqueo del habla colombiana;
2. soberanía de los datos: memorias de víctimas y comunidades que no salen del servidor;
3. costo marginal casi nulo a escala.

El costo es dejar pasar entre 5 y 7 puntos más de contenido indebido, que en el diseño del mural todavía puede atajar
el comité.

**Supuestos:** precio de gpt-5.4-mini de agregadores públicos que citan la lista de OpenAI (confirmar en la página
oficial); 150 W y USD 0,22/kWh son supuestos, no mediciones.
