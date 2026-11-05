#!/bin/bash
# Ejecución #22: evaluación ÚNICA de las variantes regionales (original y +Lisa) sobre el holdout v2 congelado.
# Mismos scripts, entradas y parámetros que correr_final.sh. Uso: correr_variantes.sh water fire
set -e
cd ~/proyectos/lexsi/guardian
PY=~/proyectos/lexsi/.venv-train/bin/python
E=resultados/final
$PY holdout_v2_entradas.py >/dev/null   # verifica los SHA-256 del holdout congelado antes de usarlo
for v in "$@"; do
  M=CohereLabs/tiny-aya-$v
  $PY evaluar_aya.py --modelo $M --prompt completo --plantilla oficial --datos $E/entrada_moderacion.jsonl --salida $E/mod-base-$v.jsonl
  $PY evaluar_aya.py --modelo $M --adaptador modelos/guardian-lisa-$v --prompt corto --datos $E/entrada_moderacion.jsonl --salida $E/mod-lisa-$v.jsonl
  $PY medir_seguridad.py --modelo $M --datos $E/entrada_seguridad.jsonl --salida $E/seg-base-$v.jsonl
  $PY medir_seguridad.py --modelo $M --adaptador modelos/guardian-lisa-$v --datos $E/entrada_seguridad.jsonl --salida $E/seg-lisa-$v.jsonl
done
echo LISTO
