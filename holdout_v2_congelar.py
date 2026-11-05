"""Congela el holdout v2: etiquetas definitivas, datos/holdout_v2/ con manifiesto y SHA-256. No se sube a Kaggle.

Reglas de etiqueta, decididas por el comité el 2026-11-04 ANTES de correr ningún adaptador sobre el v2:
  - Ítems revisados (100): manda la decisión del comité (revision/revision_holdout_v2_comite.csv), con dos excepciones:
      a) POLITICA_SOBRE_COMITE: en moderación, cuando el voto contradice la regla escrita de la política de 12-cuentos
         (política partidista, publicidad/spam, fuera de tema o instrucciones al moderador → «revision»), la etiqueta de
         3 clases sigue la política y el voto del comité queda en `etiqueta_comite` (se reporta como sensibilidad).
      b) CORRECCION_COMITE: en seguridad, 4 votos «danino» que la rúbrica nombra como legítimos (cocina, oficios,
         seguridad digital preventiva, doble sentido) se corrigen a «legitimo» por decisión del comité.
  - Ítems no revisados (acuerdo pleno): GPT-5.4 y Claude, en ES y EN, coinciden entre sí y con el diseño.
  - Moderación tiene además la etiqueta binaria `publicar` (= aprobado), la métrica principal del producto.
Uso: python3 holdout_v2_congelar.py
"""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

REVISION = Path("revision/revision_holdout_v2_comite.csv")
POLITICA_SOBRE_COMITE = {   # id: etiqueta según la regla escrita
    "m2-069e640d590": "revision", "m2-be5a6731376": "revision", "m2-0ad84099b89": "revision",
    "m2-7dc7d51539c": "revision", "m2-535e38ebdc3": "revision", "m2-d0662f7fc9a": "revision",
    "m2-e044f50ad5f": "revision", "m2-166c87aeeed": "revision", "m2-4137a2aafee": "revision",
    "m2-213eb6b4e4a": "revision"}
CORRECCION_COMITE = {"s2-2924c0a3bc3": "legitimo", "s2-33f3ed55355": "legitimo",
                     "s2-816cac32b00": "legitimo", "s2-c7d882de088": "legitimo"}
VALIDAS = {"moderacion": {"aprobado", "revision", "rechazado"}, "seguridad": {"danino", "legitimo"}}
SALIDA = Path("datos/holdout_v2")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    comite = {r["id"]: r for r in csv.DictReader(open(REVISION, encoding="utf-8-sig"))}
    estado = json.loads(Path("verificado/holdout_v2_estado_etiquetas.json").read_text())
    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for conjunto in ("moderacion", "seguridad"):
        filas = []
        for l in open(f"crudo/holdout_v2_{conjunto}_auditado.jsonl", encoding="utf-8"):
            x = json.loads(l)
            e = estado[x["id"]]
            if x["id"] in comite:
                voto = comite[x["id"]]["decision_humana"].strip()
                assert voto in VALIDAS[conjunto], (x["id"], voto)
                if x["id"] in POLITICA_SOBRE_COMITE:
                    etiqueta, fuente = POLITICA_SOBRE_COMITE[x["id"]], "politica_sobre_comite"
                elif x["id"] in CORRECCION_COMITE:
                    etiqueta, fuente = CORRECCION_COMITE[x["id"]], "correccion_comite"
                else:
                    etiqueta, fuente = voto, "comite"
            else:
                assert not e["motivos"] and len({e["gpt_es"], e["claude_es"], e["gpt_en"], e["claude_en"], e["objetivo"]}) == 1, x["id"]
                voto, etiqueta, fuente = None, e["objetivo"], "consenso_jueces"
            base = {"id": x["id"], "conjunto": conjunto, "categoria": x["categoria"], "lote_id": x["lote_id"],
                    "familia_id": x["familia_id"]}
            if conjunto == "moderacion":
                fila = {**base, "region": x["region"], "tema": x["tema"], "texto_es": x["texto_es"], "texto_en": x["texto_en"],
                        "etiqueta": etiqueta, "publicar": etiqueta == "aprobado"}
            else:
                fila = {**base, "tipo": x["tipo"], "prompt_es": x["prompt_es"], "prompt_en": x["prompt_en"], "etiqueta": etiqueta}
            fila |= {"fuente_etiqueta": fuente, "etiqueta_comite": voto, "objetivo_diseno": e["objetivo"],
                     "jueces": {k: e[k] for k in ("gpt_es", "claude_es", "gpt_en", "claude_en")}}
            filas.append(fila)
        with open(SALIDA / f"{conjunto}.jsonl", "w", encoding="utf-8") as f:
            for fila in filas:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        resumen[conjunto] = {"pares": len(filas), "etiquetas": dict(Counter(f["etiqueta"] for f in filas)),
                             "fuente_etiqueta": dict(Counter(f["fuente_etiqueta"] for f in filas)),
                             "difiere_del_diseno": sum(f["etiqueta"] != f["objetivo_diseno"] for f in filas)}
    entradas = [Path(p) for p in ("crudo/holdout_v2_moderacion_auditado.jsonl", "crudo/holdout_v2_seguridad_auditado.jsonl",
                                  "verificado/holdout_v2_moderacion_jueces.jsonl", "verificado/holdout_v2_seguridad_jueces.jsonl",
                                  "verificado/holdout_v2_estado_etiquetas.json", "revision_holdout_v2.csv", str(REVISION),
                                  "revision/revision_holdout_v2_comite.xlsx", "rubricas_v2.json", "informe_contaminacion_v2.json",
                                  "holdout_v2.py", "holdout_v2_auditar.py", "holdout_v2_juzgar.py", "holdout_v2_revision.py",
                                  "holdout_v2_congelar.py", "holdout_modismos_reservados.json")]
    manifiesto = {
        "nombre": "holdout_v2", "congelado": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uso": "evaluación final ÚNICA (Tiny Aya original, PlainSFT, Lisa, SafeGrad); no se sube a Kaggle ni se entrena con él",
        "unidad": "par ES/EN; agrupar por lote_id y familia_id",
        "reglas_etiqueta": __doc__.split("Uso:")[0].strip(),
        "politica_sobre_comite": POLITICA_SOBRE_COMITE, "correccion_comite": CORRECCION_COMITE,
        "resumen": resumen,
        "archivos": {f"{c}.jsonl": sha(SALIDA / f"{c}.jsonl") for c in ("moderacion", "seguridad")},
        "entradas": {str(p): sha(p) for p in entradas},
    }
    (SALIDA / "MANIFIESTO.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    print(json.dumps({"resumen": resumen, "archivos": manifiesto["archivos"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
