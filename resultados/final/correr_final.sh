#!/bin/bash
# Evaluación final ÚNICA sobre el holdout v2 congelado (monterrey, RTX 3050, nf4). Secuencial; sin APIs.
set -e
cd ~/proyectos/lexsi/guardian
PY=~/proyectos/lexsi/.venv-train/bin/python
M=CohereLabs/tiny-aya-global
E=resultados/final
$PY evaluar_aya.py --modelo $M --prompt completo --plantilla oficial  --datos $E/entrada_moderacion.jsonl --salida $E/mod-base-oficial.jsonl
$PY evaluar_aya.py --modelo $M --prompt completo --plantilla compacta --datos $E/entrada_moderacion.jsonl --salida $E/mod-base-compacta.jsonl
for a in plain lisa safegrad; do
  $PY evaluar_aya.py --modelo $M --adaptador modelos/guardian-$a-global --prompt corto --datos $E/entrada_moderacion.jsonl --salida $E/mod-$a.jsonl
done
$PY medir_seguridad.py --modelo $M --datos $E/entrada_seguridad.jsonl --salida $E/seg-base.jsonl
for a in plain lisa safegrad; do
  $PY medir_seguridad.py --modelo $M --adaptador modelos/guardian-$a-global --datos $E/entrada_seguridad.jsonl --salida $E/seg-$a.jsonl
done
echo LISTO
