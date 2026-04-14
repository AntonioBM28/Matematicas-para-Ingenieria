"""
TAREA 2 - Métodos Numéricos para EDOs
Método:  Euler Modificado (Heun)
Pasos:   h = 0.1, 0.001, 0.00001
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# ──────────────────────────────────────────────
# Definición de EDOs
# ──────────────────────────────────────────────
PROBLEMAS = [
    {
        "label": "a",
        "titulo": r"$y' = x\sqrt{y}$,  $y(1)=4$,  evaluar $y(10)$",
        "f": lambda x, y: x * math.sqrt(abs(y)),
        "x0": 1.0, "y0": 4.0, "xf": 10.0,
    },
    {
        "label": "b",
        "titulo": r"$y' = x^2 - 3y$,  $y(0)=1$,  evaluar $y(15)$",
        "f": lambda x, y: x**2 - 3*y,
        "x0": 0.0, "y0": 1.0, "xf": 15.0,
    },
    {
        "label": "c",
        "titulo": r"$y' = -y^{3/2} + 1$,  $y(0)=10$,  evaluar $y(20)$",
        "f": lambda x, y: -(abs(y)**1.5) + 1,
        "x0": 0.0, "y0": 10.0, "xf": 20.0,
    },
    {
        "label": "d",
        "titulo": r"$y' - y\tan(x) = 0$,  $y(0)=1$,  evaluar $y(5)$",
        "f": lambda x, y: y * math.tan(x),
        "x0": 0.0, "y0": 1.0, "xf": 5.0,
    },
    {
        "label": "e",
        "titulo": r"$y' = x + y^2$,  $y(1)=0$,  evaluar $y(1.5)$",
        "f": lambda x, y: x + y**2,
        "x0": 1.0, "y0": 0.0, "xf": 1.5,
    },
]

PASOS = [0.1, 0.001, 0.00001]

# ──────────────────────────────────────────────
# Método
# ──────────────────────────────────────────────
def euler_modificado(f, x0, y0, h, xf):
    n = int(round((xf - x0) / h))
    xs = np.empty(n + 1); ys = np.empty(n + 1); ys_pred = np.empty(n + 1)
    xs[0] = x0; ys[0] = y0; ys_pred[0] = float('nan')
    x, y = x0, y0
    for i in range(1, n + 1):
        x_next = x0 + i * h
        try:
            fx   = f(x, y)
            y_p  = y + h * fx
            y    = y + (h / 2) * (fx + f(x_next, y_p))
        except Exception:
            y_p = float('nan'); y = float('nan')
        x = x_next
        xs[i] = x; ys_pred[i] = y_p; ys[i] = y
    return xs, ys_pred, ys

# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
MAX_FILAS = 14

def submuestra_tabla(xs, ys_p, ys, max_f=MAX_FILAS):
    n = len(xs)
    idx = (np.round(np.linspace(0, n-1, max_f)).astype(int)
           if n > max_f else np.arange(n))
    return xs[idx], ys_p[idx], ys[idx]

def submuestra_graf(xs, ys, max_p=500):
    n = len(xs)
    if n <= max_p:
        return xs, ys
    idx = np.round(np.linspace(0, n-1, max_p)).astype(int)
    return xs[idx], ys[idx]

COLOR_EM = "#ff8a65"
COLOR_PRED = "#ffe082"

def dibujar_pagina(pdf, prob, h):
    f  = prob["f"]
    x0 = prob["x0"]; y0 = prob["y0"]; xf = prob["xf"]

    xs, ys_pred, ys = euler_modificado(f, x0, y0, h, xf)

    fig = plt.figure(figsize=(14, 11), facecolor="#0f1117")
    fig.suptitle(
        f"TAREA 2 — Problema {prob['label']}:  {prob['titulo']}    |    h = {h}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        top=0.90, bottom=0.07,
        left=0.06, right=0.96,
        hspace=0.40,
        height_ratios=[1.8, 1]
    )

    # ── Tabla ──────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0])
    ax_t.set_facecolor("#1a1d2e"); ax_t.axis("off")
    ax_t.set_title("Euler Modificado (Método de Heun)",
                   color=COLOR_EM, fontsize=12, pad=6)

    xs_t, yp_t, ys_t = submuestra_tabla(xs, ys_pred, ys)
    tabla_data = []
    for i, (x, yp, y) in enumerate(zip(xs_t, yp_t, ys_t)):
        yp_str = f"{yp:.6f}" if not math.isnan(yp) else "—"
        tabla_data.append([str(i), f"{x:.6f}", yp_str, f"{y:.6f}"])

    tbl = ax_t.table(
        cellText=tabla_data,
        colLabels=["n", "xₙ", "y* (predictor)", "yₙ (corrector)"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1, 1.22)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0f1117")
            cell.set_text_props(color="#ffe082", fontweight="bold")
        else:
            cell.set_facecolor("#1a1d2e")
            cell.set_text_props(color="white")
        cell.set_edgecolor("#2a3a5a")

    yf = ys[-1]
    ax_t.text(0.5, -0.05,
              f"y({xf}) ≈ {yf:.6f}",
              transform=ax_t.transAxes,
              ha="center", fontsize=9.5, color="#ffd54f", fontweight="bold")

    # ── Gráfica ──────────────────────────────────────
    ax_g = fig.add_subplot(gs[1])
    ax_g.set_facecolor("#0f1117")
    ax_g.tick_params(colors="white"); ax_g.spines[:].set_color("#2a3a5a")
    ax_g.set_title("Solución numérica — Euler Modificado",
                   color="#a8c8f8", fontsize=10)
    ax_g.set_xlabel("x", color="#7a90b0"); ax_g.set_ylabel("y", color="#7a90b0")

    xg, yg = submuestra_graf(xs, ys)
    ax_g.plot(xg, yg, color=COLOR_EM, linewidth=1.8, label="Euler Modificado")
    ax_g.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    ax_g.grid(color="#2a3a5a", linestyle="--", linewidth=0.5)

    pdf.savefig(fig, dpi=100)
    plt.close(fig)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
OUTPUT = "tarea2_euler_modificado.pdf"

with PdfPages(OUTPUT) as pdf:
    for prob in PROBLEMAS:
        for h in PASOS:
            print(f"  Problema {prob['label']}  h={h} ...", flush=True)
            dibujar_pagina(pdf, prob, h)

print(f"\n[OK]  PDF generado: {OUTPUT}")
