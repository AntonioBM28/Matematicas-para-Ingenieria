"""
TAREA 1 - Métodos Numéricos para EDOs
Métodos: Euler, Euler Modificado, Runge-Kutta 4° Orden
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
# Métodos numéricos
# ──────────────────────────────────────────────
def euler(f, x0, y0, h, xf):
    n = int(round((xf - x0) / h))
    xs = np.empty(n + 1); ys = np.empty(n + 1)
    xs[0] = x0; ys[0] = y0
    x, y = x0, y0
    for i in range(1, n + 1):
        try:
            y = y + h * f(x, y)
        except Exception:
            y = float('nan')
        x = x0 + i * h
        xs[i] = x; ys[i] = y
    return xs, ys

def euler_modificado(f, x0, y0, h, xf):
    n = int(round((xf - x0) / h))
    xs = np.empty(n + 1); ys = np.empty(n + 1)
    xs[0] = x0; ys[0] = y0
    x, y = x0, y0
    for i in range(1, n + 1):
        x_next = x0 + i * h
        try:
            fx = f(x, y)
            y_pred = y + h * fx
            y = y + (h / 2) * (fx + f(x_next, y_pred))
        except Exception:
            y = float('nan')
        x = x_next
        xs[i] = x; ys[i] = y
    return xs, ys

def rk4(f, x0, y0, h, xf):
    n = int(round((xf - x0) / h))
    xs = np.empty(n + 1); ys = np.empty(n + 1)
    xs[0] = x0; ys[0] = y0
    x, y = x0, y0
    for i in range(1, n + 1):
        try:
            k1 = f(x, y)
            k2 = f(x + h/2, y + h/2 * k1)
            k3 = f(x + h/2, y + h/2 * k2)
            k4 = f(x + h,   y + h   * k3)
            y = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        except Exception:
            y = float('nan')
        x = x0 + i * h
        xs[i] = x; ys[i] = y
    return xs, ys

# ──────────────────────────────────────────────
# Utilidades de tabla (máx 12 filas visibles)
# ──────────────────────────────────────────────
MAX_FILAS_TABLA = 12

def submuestra_tabla(xs, ys, max_f=MAX_FILAS_TABLA):
    n = len(xs)
    if n <= max_f:
        idx = np.arange(n)
    else:
        idx = np.round(np.linspace(0, n - 1, max_f)).astype(int)
    return xs[idx], ys[idx]

def submuestra_graf(xs, ys, max_p=500):
    n = len(xs)
    if n <= max_p:
        return xs, ys
    idx = np.round(np.linspace(0, n-1, max_p)).astype(int)
    return xs[idx], ys[idx]

# ──────────────────────────────────────────────
# Dibujar una página para un problema + un h
# ──────────────────────────────────────────────
COLORES = {"Euler": "#4fc3f7", "Euler Mod.": "#ff8a65", "RK4": "#81c784"}

def dibujar_pagina(pdf, prob, h):
    f   = prob["f"]
    x0  = prob["x0"]; y0 = prob["y0"]; xf = prob["xf"]

    xs_e,  ys_e  = euler(f, x0, y0, h, xf)
    xs_em, ys_em = euler_modificado(f, x0, y0, h, xf)
    xs_rk, ys_rk = rk4(f, x0, y0, h, xf)

    fig = plt.figure(figsize=(17, 11), facecolor="#0f1117")
    fig.suptitle(
        f"TAREA 1 — Problema {prob['label']}:  {prob['titulo']}    |    h = {h}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        top=0.90, bottom=0.08,
        left=0.04, right=0.98,
        hspace=0.38, wspace=0.22,
        height_ratios=[1.6, 1]
    )

    # ── Tablas (fila 0) ──────────────────────────────
    metodos = [
        ("Euler",      xs_e,  ys_e),
        ("Euler Mod.", xs_em, ys_em),
        ("RK4",        xs_rk, ys_rk),
    ]
    for col, (nombre, xs, ys) in enumerate(metodos):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#1a1d2e"); ax.axis("off")
        ax.set_title(nombre, color=COLORES[nombre], fontsize=11, pad=4)

        xs_t, ys_t = submuestra_tabla(xs, ys)
        tabla_data  = [[str(i), f"{x:.6f}", f"{y:.6f}"]
                       for i, (x, y) in enumerate(zip(xs_t, ys_t))]
        col_labels  = ["n", "x", "y"]
        tbl = ax.table(
            cellText=tabla_data, colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
        tbl.scale(1, 1.18)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#0f1117" if r == 0 else "#1a1d2e")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#2a3a5a")

        # Valor final
        yf_val = ys[-1]
        ax.text(0.5, -0.04,
                f"y({xf}) ≈ {yf_val:.6f}",
                transform=ax.transAxes,
                ha="center", fontsize=8.5, color="#ffd54f",
                fontweight="bold")

    # ── Gráfica comparativa (fila 1, ocupa las 3 columnas) ──
    ax_g = fig.add_subplot(gs[1, :])
    ax_g.set_facecolor("#0f1117")
    ax_g.tick_params(colors="white"); ax_g.spines[:].set_color("#2a3a5a")
    ax_g.xaxis.label.set_color("white"); ax_g.yaxis.label.set_color("white")
    ax_g.set_title("Gráfica comparativa", color="#a8c8f8", fontsize=10)
    ax_g.set_xlabel("x", color="#7a90b0"); ax_g.set_ylabel("y", color="#7a90b0")

    for nombre, xs, ys in metodos:
        xg, yg = submuestra_graf(xs, ys)
        ax_g.plot(xg, yg, label=nombre, color=COLORES[nombre], linewidth=1.6)

    ax_g.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    ax_g.grid(color="#2a3a5a", linestyle="--", linewidth=0.5)

    pdf.savefig(fig, dpi=100)
    plt.close(fig)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
OUTPUT = "tarea1_euler_euler_mod_rk4.pdf"

with PdfPages(OUTPUT) as pdf:
    for prob in PROBLEMAS:
        for h in PASOS:
            print(f"  Problema {prob['label']}  h={h} ...", flush=True)
            dibujar_pagina(pdf, prob, h)

print(f"\n[OK]  PDF generado: {OUTPUT}")
