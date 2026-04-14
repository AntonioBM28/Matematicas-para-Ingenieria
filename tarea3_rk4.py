"""
TAREA 3 - Métodos Numéricos para EDOs
Método:  Runge-Kutta de 4° Orden (RK4)
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
# Método RK4 — guarda también k1..k4
# ──────────────────────────────────────────────
def rk4_detallado(f, x0, y0, h, xf):
    n = int(round((xf - x0) / h))
    xs  = np.empty(n + 1)
    ys  = np.empty(n + 1)
    K1  = np.empty(n + 1); K2 = np.empty(n + 1)
    K3  = np.empty(n + 1); K4 = np.empty(n + 1)
    xs[0] = x0; ys[0] = y0
    K1[0] = K2[0] = K3[0] = K4[0] = float('nan')
    x, y = x0, y0
    for i in range(1, n + 1):
        try:
            k1 = f(x,       y)
            k2 = f(x + h/2, y + h/2 * k1)
            k3 = f(x + h/2, y + h/2 * k2)
            k4 = f(x + h,   y + h   * k3)
            y  = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        except Exception:
            k1=k2=k3=k4=float('nan'); y=float('nan')
        x = x0 + i * h
        xs[i] = x; ys[i] = y
        K1[i] = k1; K2[i] = k2; K3[i] = k3; K4[i] = k4
    return xs, K1, K2, K3, K4, ys

# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
MAX_FILAS = 12

def submuestra_idx(n, max_f=MAX_FILAS):
    if n <= max_f:
        return np.arange(n)
    return np.round(np.linspace(0, n-1, max_f)).astype(int)

def submuestra_graf(xs, ys, max_p=500):
    n = len(xs)
    if n <= max_p:
        return xs, ys
    idx = np.round(np.linspace(0, n-1, max_p)).astype(int)
    return xs[idx], ys[idx]

def fmt(v):
    return f"{v:.6f}" if not math.isnan(v) else "—"

COLOR_RK = "#81c784"

def dibujar_pagina(pdf, prob, h):
    f  = prob["f"]
    x0 = prob["x0"]; y0 = prob["y0"]; xf = prob["xf"]

    xs, K1, K2, K3, K4, ys = rk4_detallado(f, x0, y0, h, xf)

    # Mostrar k1..k4 solo si h=0.1 (tabla manejable); de lo contrario ocultar
    mostrar_ks = (h == 0.1)

    fig = plt.figure(figsize=(18 if mostrar_ks else 14, 11), facecolor="#0f1117")
    fig.suptitle(
        f"TAREA 3 — Problema {prob['label']}:  {prob['titulo']}    |    h = {h}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        top=0.90, bottom=0.07,
        left=0.04, right=0.97,
        hspace=0.40,
        height_ratios=[1.8, 1]
    )

    # ── Tabla ──────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0])
    ax_t.set_facecolor("#1a1d2e"); ax_t.axis("off")
    ax_t.set_title("Runge-Kutta de 4° Orden",
                   color=COLOR_RK, fontsize=12, pad=6)

    idx = submuestra_idx(len(xs))
    if mostrar_ks:
        col_labels = ["n", "xₙ", "k₁", "k₂", "k₃", "k₄", "yₙ"]
        tabla_data = [
            [str(i), fmt(xs[i]),
             fmt(K1[i]), fmt(K2[i]), fmt(K3[i]), fmt(K4[i]),
             fmt(ys[i])]
            for i in idx
        ]
    else:
        col_labels = ["n", "xₙ", "yₙ"]
        tabla_data = [
            [str(i), fmt(xs[i]), fmt(ys[i])]
            for i in idx
        ]

    tbl = ax_t.table(
        cellText=tabla_data, colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5 if mostrar_ks else 8.5)
    tbl.scale(1, 1.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0f1117")
            cell.set_text_props(color="#a5d6a7", fontweight="bold")
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
    ax_g.set_title("Solución numérica — RK4",
                   color="#a8c8f8", fontsize=10)
    ax_g.set_xlabel("x", color="#7a90b0"); ax_g.set_ylabel("y", color="#7a90b0")

    xg, yg = submuestra_graf(xs, ys)
    ax_g.plot(xg, yg, color=COLOR_RK, linewidth=1.8, label="RK4")
    ax_g.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    ax_g.grid(color="#2a3a5a", linestyle="--", linewidth=0.5)

    pdf.savefig(fig, dpi=100)
    plt.close(fig)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
OUTPUT = "tarea3_rk4.pdf"

with PdfPages(OUTPUT) as pdf:
    for prob in PROBLEMAS:
        for h in PASOS:
            print(f"  Problema {prob['label']}  h={h} ...", flush=True)
            dibujar_pagina(pdf, prob, h)

print(f"\n[OK]  PDF generado: {OUTPUT}")
