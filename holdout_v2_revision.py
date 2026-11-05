"""Planilla de revisión humana del holdout v2 (hasta 100 ítems) y estado de etiquetas antes del congelamiento.

Las etiquetas de referencia salen de la política (dos jueces con la misma rúbrica) y de la revisión humana; nunca de
las predicciones de los adaptadores. Prioridades para la planilla (cada par ES/EN entra una sola vez):
  1. desacuerdo GPT-5.4 vs Claude Sonnet 5.5 (en ES)
  2. falta el juicio de Claude (filtro del proveedor)
  3. los jueces coinciden pero no con el objetivo del diseño
  4. la etiqueta cambia entre ES y EN (deriva de la traducción)
  5. algún juez marcó «ambiguo» (seguridad)
  6. muestra aleatoria estratificada (conjunto × categoría) de los acuerdos plenos, hasta completar 100
Uso: python3 holdout_v2_revision.py
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

MAXIMO, SEMILLA = 100, 20261108


def etiqueta(j, conjunto):
    if not j:
        return None
    return j.get("veredicto") if conjunto == "moderacion" else j.get("clase")


def main() -> None:
    filas, acuerdos = [], defaultdict(list)
    estado = {}
    for conjunto in ("moderacion", "seguridad"):
        items = {json.loads(l)["id"]: json.loads(l) for l in open(f"crudo/holdout_v2_{conjunto}_auditado.jsonl", encoding="utf-8")}
        for l in open(f"verificado/holdout_v2_{conjunto}_jueces.jsonl", encoding="utf-8"):
            y = json.loads(l)
            x, J = items[y["id"]], y["jueces"]
            e = {k: etiqueta(J.get(k), conjunto) for k in ("gpt_es", "claude_es", "gpt_en", "claude_en")}
            objetivo = x["veredicto_objetivo"] if conjunto == "moderacion" else ("danino" if x["tipo"] == "danino" else "legitimo")
            ambiguo = conjunto == "seguridad" and any((J.get(k) or {}).get("ambiguo") for k in J)
            motivos = []
            if e["claude_es"] and e["gpt_es"] != e["claude_es"]:
                motivos.append("1 desacuerdo GPT/Claude")
            if not e["claude_es"] or not e["claude_en"]:
                motivos.append("2 sin juicio de Claude")
            if e["gpt_es"] == e["claude_es"] and e["gpt_es"] != objetivo:
                motivos.append("3 jueces ≠ objetivo del diseño")
            if (e["gpt_es"] != e["gpt_en"]) or (e["claude_es"] and e["claude_en"] and e["claude_es"] != e["claude_en"]):
                motivos.append("4 cambia entre ES y EN")
            if ambiguo:
                motivos.append("5 marcado ambiguo")
            fila = {"id": x["id"], "conjunto": conjunto, "categoria": x["categoria"], "tipo": x.get("tipo", ""),
                    "texto_es": x.get("texto_es") or x.get("prompt_es"), "texto_en": x.get("texto_en") or x.get("prompt_en"),
                    "objetivo_diseno": objetivo, **e,
                    "explicacion_gpt": (J.get("gpt_es") or {}).get("explicacion", ""),
                    "explicacion_claude": (J.get("claude_es") or {}).get("explicacion", ""),
                    "lote_id": x["lote_id"], "familia_id": x["familia_id"]}
            estado[x["id"]] = {"conjunto": conjunto, "objetivo": objetivo, **e, "motivos": motivos}
            if motivos:
                fila["motivo"] = " · ".join(motivos)
                fila["prioridad"] = int(motivos[0][0])
                filas.append(fila)
            else:
                acuerdos[(conjunto, x["categoria"])].append(fila)
    filas.sort(key=lambda f: (f["prioridad"], f["conjunto"], f["categoria"]))
    seleccion = filas[:MAXIMO]
    # relleno estratificado con acuerdos plenos
    azar = random.Random(SEMILLA)
    estratos = sorted(acuerdos)
    for e in estratos:
        azar.shuffle(acuerdos[e])
    while len(seleccion) < MAXIMO and any(acuerdos[e] for e in estratos):
        for e in estratos:
            if acuerdos[e] and len(seleccion) < MAXIMO:
                f = acuerdos[e].pop()
                f["motivo"], f["prioridad"] = "6 auditoría aleatoria de acuerdos", 6
                seleccion.append(f)
    columnas = ["id", "conjunto", "categoria", "tipo", "prioridad", "motivo", "texto_es", "texto_en", "objetivo_diseno",
                "gpt_es", "claude_es", "gpt_en", "claude_en", "explicacion_gpt", "explicacion_claude", "lote_id",
                "familia_id", "decision_comite", "nota_comite"]
    with open("revision_holdout_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for s in seleccion:
            w.writerow({**s, "decision_comite": "", "nota_comite": ""})
    Path("verificado/holdout_v2_estado_etiquetas.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False))
    resumen = defaultdict(int)
    for s in seleccion:
        resumen[s["motivo"].split(" · ")[0]] += 1
    no_revisados_con_motivo = len(filas) - min(len(filas), MAXIMO)
    print(json.dumps({"en_planilla": len(seleccion), "por_motivo_principal": dict(sorted(resumen.items())),
                      "con_motivo_fuera_de_la_planilla": no_revisados_con_motivo,
                      "acuerdos_plenos_sin_revisar": sum(len(v) for v in acuerdos.values())}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
