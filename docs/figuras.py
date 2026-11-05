"""Figuras del README a partir de los resultados reales (resultados/final/analisis_final.json y roi.json).
Uso: uv run --with matplotlib python docs/figuras.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
OUT = R / "docs" / "img"
A = json.loads((R / "resultados/final/analisis_final.json").read_text())
ROI = json.loads((R / "resultados/final/roi.json").read_text())["calidad"]
C = json.loads((R / "resultados/final/seg-claude.resumen.json").read_text())["por_modelo"]
SUP, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ES, EN, GRIS = "#2a78d6", "#eb6834", "#a3a29c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": T2,
                     "xtick.color": T2, "ytick.color": T2, "axes.titlecolor": T1, "figure.facecolor": SUP,
                     "axes.facecolor": SUP, "savefig.facecolor": SUP})
NOM = {"base": "Tiny Aya\noriginal", "plain": "PlainSFT", "lisa": "Lisa", "safegrad": "SafeGrad"}


def barras(ax, vals, titulo):
    xs = range(len(NOM))
    w = 0.36
    for k, (idioma, color, d) in enumerate((("ES", ES, -w / 2 - 0.01), ("EN", EN, w / 2 + 0.01))):
        v = [vals[m][k] * 100 for m in NOM]
        bs = ax.bar([x + d for x in xs], v, w, color=color, label=idioma, zorder=3)
        for b, y in zip(bs, v):
            ax.text(b.get_x() + b.get_width() / 2, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9.5, color=T1)
    ax.set_xticks(list(xs), list(NOM.values()))
    ax.set_ylim(0, 75)
    ax.set_title(titulo, fontsize=12, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks([0, 25, 50, 75], ["0%", "25%", "50%", "75%"])
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "Lisa":
            lbl.set_fontweight("bold")


gpt = {m: (A["seguridad"][m]["es"]["ASR"]["valor"], A["seguridad"][m]["en"]["ASR"]["valor"]) for m in NOM}
cla = {m: (C[m]["es"]["ASR_claude"], C[m]["en"]["ASR_claude"]) for m in NOM}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
barras(axs[0], gpt, "Judge: GPT-5.4 (all 34 harmful requests per language)")
barras(axs[1], cla, "Judge: Claude Sonnet 5.5 (items its provider let through)")
axs[0].legend(frameon=False, loc="upper right", labelcolor=T1)
fig.suptitle("Attack success rate on harmful requests — lower is better", x=0.01, ha="left", fontsize=14,
             fontweight="bold", color=T1)
fig.text(0.01, 0.005, "Holdout v2 (frozen, used once). General-assistant mode, official chat template.",
         fontsize=9, color=T2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT / "safety_asr.png", dpi=160)

# compromiso de moderación: lo que se cuela frente a lo que se bloquea (promedio ES/EN)
pts = {"Tiny Aya original": A["moderacion"]["base-oficial"], "PlainSFT": A["moderacion"]["plain"],
       "Lisa": A["moderacion"]["lisa"], "SafeGrad": A["moderacion"]["safegrad"],
       "Closed chain\n(rules + omni + gpt-5.4-mini)": ROI["cadena_12cuentos"]}
fig, ax = plt.subplots(figsize=(8, 5.2))
for nombre, d in pts.items():
    x = (d["es"]["sobre_bloqueo"]["valor"] + d["en"]["sobre_bloqueo"]["valor"]) / 2 * 100
    y = (d["es"]["se_cuela"]["valor"] + d["en"]["se_cuela"]["valor"]) / 2 * 100
    lisa = nombre == "Lisa"
    ax.scatter(x, y, s=150 if lisa else 90, color=ES if lisa else GRIS, edgecolor=SUP, linewidth=2, zorder=3)
    dx, dy, ha = {"Tiny Aya original": (1.2, -0.6, "left"), "PlainSFT": (-1.0, 4.5, "right"), "Lisa": (-1.0, 3.0, "right"),
                  "SafeGrad": (1.0, 4.5, "left")}.get(nombre, (0.0, -8.5, "center"))
    ax.annotate(nombre, (x, y), (x + dx, y + dy), ha=ha, fontsize=10, color=T1, fontweight="bold" if lisa else "normal",
                arrowprops=None if nombre in ("Tiny Aya original", "Lisa") else dict(arrowstyle="-", color=GRIS, lw=1))
ax.set_xlabel("Blocks notes the committee approved (%)  →  more over-blocking")
ax.set_ylabel("Lets through notes that should be blocked (%)")
ax.set_xlim(-2, 36)
ax.set_ylim(-10, 50)
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.grid(color=GRID, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("better", xy=(0.5, -1), xytext=(6, 14), color=T2, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=T2))
ax.set_title("Moderation trade-off on the frozen holdout (mean of ES and EN)", loc="left", fontsize=13,
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "moderation_tradeoff.png", dpi=160)
print("ok")
