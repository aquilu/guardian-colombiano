---
title: Guardián colombiano
emoji: 🧶
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: true
license: cc-by-nc-4.0
short_description: Tiny Aya + Lisa moderating Colombian-Spanish memories
models:
  - vasmi-sas/guardian-tiny-aya-lisa
  - CohereLabs/tiny-aya-global
datasets:
  - vasmi-sas/guardian-colombiano-holdout-v2
---

Demo del guardián de **12-cuentos**: Tiny Aya Global (3,35B) afinado con **Lisa** (SafeTune) modera post-its en
español de Colombia o en inglés y devuelve el JSON de decisión. La segunda pestaña compara la seguridad general del
modelo afinado con la de Tiny Aya original. Corre en ZeroGPU.

Proyecto: https://github.com/aquilu/guardian-colombiano
