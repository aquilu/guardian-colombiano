"""Deriva, sin modificarlo, las entradas de la evaluación final desde el holdout v2 congelado.

  - resultados/final/entrada_moderacion.jsonl: formato de evaluar_aya.py (veredicto_objetivo = etiqueta congelada).
  - resultados/final/entrada_seguridad.jsonl: una fila por idioma, formato de medir_seguridad.py
    (tipo = "danino" si la etiqueta congelada es danino, si no "limite"; el tipo del diseño queda en tipo_diseno).
Verifica primero el SHA-256 de los archivos congelados contra MANIFIESTO.json.
"""
import hashlib
import json
from pathlib import Path

H = Path("datos/holdout_v2")
man = json.loads((H / "MANIFIESTO.json").read_text())
for nombre, h in man["archivos"].items():
    assert hashlib.sha256((H / nombre).read_bytes()).hexdigest() == h, f"{nombre} no coincide con el manifiesto"
out = Path("resultados/final")
out.mkdir(parents=True, exist_ok=True)
with open(out / "entrada_moderacion.jsonl", "w", encoding="utf-8") as f:
    for l in open(H / "moderacion.jsonl", encoding="utf-8"):
        x = json.loads(l)
        f.write(json.dumps({**x, "veredicto_objetivo": x["etiqueta"]}, ensure_ascii=False) + "\n")
with open(out / "entrada_seguridad.jsonl", "w", encoding="utf-8") as f:
    for l in open(H / "seguridad.jsonl", encoding="utf-8"):
        x = json.loads(l)
        for idioma in ("es", "en"):
            f.write(json.dumps({"id": x["id"], "idioma": idioma, "categoria": x["categoria"], "lote_id": x["lote_id"],
                                "familia_id": x["familia_id"], "tipo": "danino" if x["etiqueta"] == "danino" else "limite",
                                "tipo_diseno": x["tipo"], "prompt": x[f"prompt_{idioma}"]}, ensure_ascii=False) + "\n")
print("entradas derivadas; manifiesto verificado", man["archivos"])
