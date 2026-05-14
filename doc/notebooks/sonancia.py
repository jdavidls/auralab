# %% [markdown]
# # SONANCIA — Visualización Interactiva
#
# Explora la función de sonancia derivada de la autocorrelación de ondas:
#
#     R(n) = cos²(πrn)                              [autocorrelación en lags enteros]
#     S_k*(r; N) = Σ_{n=1}^N  w_n cos^{2k}(πrn)   [w₁=½, wₙ=1/n (n≥2)]
#     𝒮_k(r; N) = ½ [S_k*(r;N) + S_k*(1/r;N)]     [forma recíproca simétrica]
#
# Métodos implementados en `auralab.sonance`:
# - `sonance(r, k, N)` — forma discreta recíproca (principal)
# - `sonance_fourier(r, k)` — forma Fourier-Zeta convergente (N→∞)
# - `sonance_exact_reciprocal(p, q, k)` — forma exacta cerrada en racionales

# %% Imports y configuración
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ipywidgets as widgets
from IPython.display import display
from fractions import Fraction

from auralab.sonance import (
    sonance,
    sonance_star,
    sonance_fourier,
    sonance_exact,
    sonance_exact_reciprocal,
    sonance_spectrum,
    sonance_curve,
)

# %matplotlib inline
matplotlib.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#0a0a0a',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#999999',
    'text.color': '#cccccc',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'grid.color': '#1a1a1a',
    'grid.alpha': 0.8,
    'font.family': 'monospace',
    'font.size': 9,
    'figure.dpi': 110,
    'savefig.facecolor': '#0a0a0a',
})

sonance_cmap = LinearSegmentedColormap.from_list('sonance', [
    '#0a0a0a', '#1a0a2a', '#3a1040', '#802040',
    '#cc6030', '#ddaa30', '#eedd60', '#ffffcc',
])

# %% Constantes — intervalos musicales y utilidades

INTERVALS = {
    'P1': (1, 1),   'm2': (16, 15), 'M2': (9, 8),   'm3': (6, 5),
    'M3': (5, 4),   'P4': (4, 3),   'TT': (45, 32),  'P5': (3, 2),
    'm6': (8, 5),   'M6': (5, 3),   'm7': (9, 5),    'M7': (15, 8),
    'P8': (2, 1),
}

INTERVAL_RATIOS = {name: p / q for name, (p, q) in INTERVALS.items()}


def _best_rational(r: float, max_denom: int = 500):
    frac = Fraction(r).limit_denominator(max_denom)
    return frac.numerator, frac.denominator


def _annotate_intervals(ax, y_top: float, octave_range=(1.0, 2.0)):
    lo, hi = octave_range
    for name, ratio in INTERVAL_RATIOS.items():
        if lo <= ratio <= hi:
            ax.axvline(ratio, color='#ffffff', alpha=0.06, linewidth=0.6, linestyle=':')
            ax.text(ratio, y_top, name, ha='center', fontsize=6.5,
                    color='#555566', rotation=65, va='bottom')


# %% ── 1. PAISAJE INTERACTIVO ─────────────────────────────────────────────────
# Controles: k, N, rango de r, métodos a superponer.

def _plot_landscape(k, N, r_lo, r_hi, show_fourier, show_exact):
    r_grid = np.linspace(r_lo, r_hi, 1200)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor('#0a0a0a')

    S = sonance(r_grid, k=k, N=N)
    ax.plot(r_grid, S, color='#6688cc', linewidth=1.6, label=f'𝒮_k (k={k}, N={N})', zorder=3)

    if show_fourier:
        Sf = sonance_fourier(r_grid, k=k, reg=1e-7, reciprocal=True)
        sf_pos  = np.where(Sf > 0, Sf, np.nan)
        sf_norm = sf_pos / np.nanmax(sf_pos) * S.max()
        ax.plot(r_grid, sf_norm, color='#cc8844', linewidth=1.0, alpha=0.75,
                label='Fourier-Zeta (norm)', zorder=2)

    if show_exact:
        for name, (p, q) in INTERVALS.items():
            ratio = p / q
            if r_lo <= ratio <= r_hi:
                s_ex = sonance_exact_reciprocal(p, q, k=k)
                ax.plot(ratio, s_ex, 'o', color='#55dd88', markersize=5,
                        markeredgewidth=0.5, markeredgecolor='#003311', zorder=5)
                ax.text(ratio, s_ex + S.max() * 0.04, name,
                        ha='center', fontsize=7, color='#55dd88')

    _annotate_intervals(ax, y_top=-S.max() * 0.06, octave_range=(r_lo, r_hi))

    ax.set_xlim(r_lo, r_hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('r = f₂/f₁')
    ax.set_ylabel('𝒮_k(r)')
    ax.set_title(
        '𝒮_k(r; N) = ½[S_k*(r;N) + S_k*(1/r;N)] / W_N',
        fontsize=10, color='#bbbbbb',
    )
    ax.legend(loc='upper right', fontsize=8, framealpha=0.25)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


_w_k = widgets.IntSlider(value=20, min=2, max=150, step=2,
                          description='k (resolución)', style={'description_width': '110px'},
                          layout=widgets.Layout(width='460px'))
_w_N = widgets.IntSlider(value=500, min=50, max=2000, step=50,
                          description='N (lags)', style={'description_width': '110px'},
                          layout=widgets.Layout(width='460px'))
_w_rlo = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.05,
                               description='r mín', style={'description_width': '80px'},
                               layout=widgets.Layout(width='360px'))
_w_rhi = widgets.FloatSlider(value=2.0, min=1.0, max=4.0, step=0.05,
                               description='r máx', style={'description_width': '80px'},
                               layout=widgets.Layout(width='360px'))
_w_fourier = widgets.Checkbox(value=False, description='Fourier-Zeta (N→∞)')
_w_exact   = widgets.Checkbox(value=True,  description='Exacto en intervalos')

display(widgets.VBox([
    widgets.HBox([_w_k, _w_N]),
    widgets.HBox([_w_rlo, _w_rhi]),
    widgets.HBox([_w_fourier, _w_exact]),
]))
widgets.interactive_output(_plot_landscape, {
    'k': _w_k, 'N': _w_N,
    'r_lo': _w_rlo, 'r_hi': _w_rhi,
    'show_fourier': _w_fourier,
    'show_exact': _w_exact,
})

# %% ── 2. COMPARACIÓN MULTI-k ─────────────────────────────────────────────────
# Resolución progresiva: cómo el paisaje se afila con k creciente.

def _plot_multik(N, k_list_str):
    try:
        k_vals = [int(x.strip()) for x in k_list_str.split(',') if x.strip()]
    except ValueError:
        print("Introduce valores de k separados por comas.")
        return

    r_grid = np.linspace(1.0, 2.0, 1500)
    palette = ['#cc555590', '#cc884490', '#ccaa3390', '#55aa5590',
               '#4488cc', '#8866cc', '#cc44aa90', '#44cccc90']

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor('#0a0a0a')

    s_max = 0.0
    curves = []
    for kv in k_vals:
        S = sonance(r_grid, k=kv, N=N)
        curves.append(S)
        s_max = max(s_max, S.max())

    for kv, S, col in zip(k_vals, curves, palette):
        ax.plot(r_grid, S, color=col, linewidth=1.3, alpha=0.9, label=f'k={kv}')

    _annotate_intervals(ax, y_top=-s_max * 0.06)
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('r = f₂/f₁')
    ax.set_ylabel('𝒮_k(r)')
    ax.set_title(f'Resolución progresiva — N={N}', fontsize=10, color='#bbbbbb')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.25, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


_w_N2     = widgets.IntSlider(value=500, min=50, max=2000, step=50,
                               description='N', style={'description_width': '40px'},
                               layout=widgets.Layout(width='400px'))
_w_klist  = widgets.Text(value='5, 10, 20, 50, 100',
                          description='k valores',
                          style={'description_width': '80px'},
                          layout=widgets.Layout(width='400px'))

display(widgets.HBox([_w_N2, _w_klist]))
widgets.interactive_output(_plot_multik, {'N': _w_N2, 'k_list_str': _w_klist})

# %% ── 3. CONVERGENCIA N→∞ ────────────────────────────────────────────────────
# sonance(r, k, N) vs. forma exacta cerrada al variar N.

def _plot_convergence(k, r_str):
    try:
        r = float(r_str)
    except ValueError:
        print("Introduce un ratio numérico válido (e.g. 1.5  ó  1.333).")
        return

    p, q    = _best_rational(r)
    s_exact = sonance_exact_reciprocal(p, q, k=k)
    N_vals  = np.unique(np.logspace(1.5, 3.3, 60).astype(int))
    S_N     = np.array([float(sonance(r, k=k, N=int(nv))) for nv in N_vals])

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.semilogx(N_vals, S_N, color='#6688cc', linewidth=1.8,
                label=f'𝒮_k(r={r:.4f}, k={k}, N)')
    ax.axhline(s_exact, color='#55dd88', linewidth=1.2, linestyle='--',
               label=f'Exacto N→∞ = {s_exact:.5f}  ({p}/{q})')
    ax.set_xlabel('N (número de lags)')
    ax.set_ylabel('𝒮_k(r)')
    ax.set_title(f'Convergencia 𝒮_k(r={r:.4f}) con N  —  k={k}', fontsize=10, color='#bbbbbb')
    ax.legend(fontsize=8, framealpha=0.25)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


_w_k3   = widgets.IntSlider(value=20, min=2, max=100, step=2,
                              description='k', style={'description_width': '40px'},
                              layout=widgets.Layout(width='360px'))
_w_r3   = widgets.Text(value='1.5', description='r',
                        style={'description_width': '40px'},
                        layout=widgets.Layout(width='180px'))
_w_btn3 = widgets.Button(description='Calcular', button_style='info',
                          layout=widgets.Layout(width='100px'))
_out3   = widgets.Output()

def _on_btn3(_):
    _out3.clear_output(wait=True)
    with _out3:
        _plot_convergence(_w_k3.value, _w_r3.value)

_w_btn3.on_click(_on_btn3)
display(widgets.HBox([_w_k3, _w_r3, _w_btn3]))
display(_out3)

# %% ── 4. LAGS — CONTRIBUCIÓN POR LAG ─────────────────────────────────────────
# Qué lags n dominan la suma para un ratio dado.

def _plot_lags(r_str, k, N):
    try:
        r = float(r_str)
    except ValueError:
        print("Introduce un ratio numérico válido.")
        return

    n     = np.arange(1, N + 1, dtype=float)
    w     = np.where(n == 1, 0.5, 1.0 / n)
    cos2k = np.cos(np.pi * r * n) ** (2 * k)
    contribs = cos2k * w
    total    = contribs.sum()
    rel      = contribs / total if total > 0 else contribs

    p, q  = _best_rational(r)
    R_vals = np.cos(np.pi * r * n) ** 2

    fig, (ax_r, ax_c) = plt.subplots(2, 1, figsize=(13, 6),
                                      gridspec_kw={'height_ratios': [1, 3]},
                                      sharex=True)
    fig.patch.set_facecolor('#0a0a0a')

    ax_r.bar(n, R_vals, width=0.8,
             color=[plt.cm.Blues(0.3 + 0.7 * float(v)) for v in R_vals],
             edgecolor='none')
    ax_r.set_ylabel('R(n) = cos²(πrn)', fontsize=8)
    ax_r.set_ylim(0, 1.15)

    s_val = float(sonance(r, k=k, N=N))
    s_ex  = sonance_exact_reciprocal(p, q, k=k)
    ax_r.set_title(
        f'r = {r:.4f} ≈ {p}/{q}   k={k}   𝒮_k={s_val:.4f}   exacto={s_ex:.4f}   1/q={1/q:.4f}',
        fontsize=9, color='#bbbbbb',
    )

    heat   = np.minimum(rel * N * 0.8, 1.0)
    colors = [sonance_cmap(float(h)) for h in heat]
    ax_c.bar(n, contribs, width=0.8, color=colors, edgecolor='none')

    thresh = contribs.max() * 0.15
    for i, c in enumerate(contribs):
        if c > thresh:
            ax_c.text(n[i], c + contribs.max() * 0.03, f'n={int(n[i])}',
                      ha='center', fontsize=7, color='#cccccc')

    ax_c.set_xlabel('n  (lags en periodos de f₁)')
    ax_c.set_ylabel('w_n · cos²ᵏ(πrn)')
    ax_c.set_xlim(0, N + 1)

    sm   = plt.cm.ScalarMappable(cmap=sonance_cmap, norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax_c, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Peso relativo', fontsize=8)
    plt.tight_layout()
    plt.show()


_w_r4   = widgets.Text(value='1.5', description='r',
                        style={'description_width': '40px'},
                        layout=widgets.Layout(width='180px'))
_w_k4   = widgets.IntSlider(value=20, min=2, max=150, step=2,
                              description='k', style={'description_width': '40px'},
                              layout=widgets.Layout(width='340px'))
_w_N4   = widgets.IntSlider(value=64, min=16, max=300, step=8,
                              description='N', style={'description_width': '40px'},
                              layout=widgets.Layout(width='340px'))
_w_btn4 = widgets.Button(description='Calcular', button_style='info',
                          layout=widgets.Layout(width='100px'))
_out4   = widgets.Output()

def _on_btn4(_):
    _out4.clear_output(wait=True)
    with _out4:
        _plot_lags(_w_r4.value, _w_k4.value, _w_N4.value)

_w_btn4.on_click(_on_btn4)
display(widgets.HBox([_w_r4, _w_btn4]))
display(widgets.HBox([_w_k4, _w_N4]))
display(_out4)

# %% ── 5. MAPA DE TRÍADAS ─────────────────────────────────────────────────────
# Heatmap de sonancia total para pares de intervalos (r₁, r₂) en una octava.

def _plot_triads(k, N, res):
    r1_range = np.linspace(1.0, 2.0, res)
    r2_range = np.linspace(1.0, 2.0, res)
    R1, R2   = np.meshgrid(r1_range, r2_range)

    r1_flat = R1.ravel()
    r2_flat = R2.ravel()
    r3_flat = r2_flat / r1_flat

    all_r = np.concatenate([r1_flat, r2_flat, r3_flat])
    all_s = sonance(all_r, k=k, N=N)
    n2    = res * res
    Z = (all_s[:n2] + all_s[n2:2*n2] + all_s[2*n2:]).reshape(res, res)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#0a0a0a')
    im = ax.pcolormesh(R1, R2, Z, cmap=sonance_cmap, shading='auto')

    chords = {
        'Mayor': (5/4, 3/2), 'Menor': (6/5, 3/2),  'Sus4': (4/3, 3/2),
        'Dim':   (6/5, 45/32), 'Aug': (5/4, 8/5),  '7dom': (5/4, 9/5),
        'Maj7':  (5/4, 15/8), 'P5+P8': (3/2, 2),
    }
    for name, (cr1, cr2) in chords.items():
        if 1.0 <= cr1 <= 2.0 and 1.0 <= cr2 <= 2.0:
            ax.plot(cr1, cr2, 'o', color='white', markersize=5,
                    markeredgewidth=0.6, markeredgecolor='black', zorder=5)
            ax.annotate(name, (cr1, cr2), xytext=(5, 5),
                        textcoords='offset points', fontsize=7, color='#cccccc')

    plt.colorbar(im, ax=ax, shrink=0.75, label='𝒮(r₁) + 𝒮(r₂) + 𝒮(r₂/r₁)')
    ax.set_xlabel('r₁ = f₂/f₁')
    ax.set_ylabel('r₂ = f₃/f₁')
    ax.set_title(f'Mapa de sonancia de tríadas  —  k={k}', fontsize=11, color='#cccccc')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


_w_k5   = widgets.IntSlider(value=20, min=5, max=80, step=5,
                              description='k', style={'description_width': '90px'},
                              layout=widgets.Layout(width='360px'))
_w_N5   = widgets.IntSlider(value=500, min=100, max=1500, step=100,
                              description='N', style={'description_width': '90px'},
                              layout=widgets.Layout(width='360px'))
_w_res5 = widgets.IntSlider(value=80, min=30, max=200, step=10,
                              description='resolución', style={'description_width': '90px'},
                              layout=widgets.Layout(width='360px'))
_w_btn5 = widgets.Button(description='Calcular', button_style='info',
                          layout=widgets.Layout(width='100px'))
_out5   = widgets.Output()

def _on_btn5(_):
    _out5.clear_output(wait=True)
    with _out5:
        _plot_triads(_w_k5.value, _w_N5.value, _w_res5.value)

_w_btn5.on_click(_on_btn5)
display(widgets.HBox([_w_k5, _w_N5, _w_res5, _w_btn5]))
display(_out5)

# %% ── 6. CURVA DE TIMBRE ─────────────────────────────────────────────────────
# Cuánta sonancia tiene un timbre armónico U desplazado por r respecto a T.

def _plot_timbre_curve(k, n_partials, amp_decay):
    partials = np.arange(1, n_partials + 1, dtype=float)
    amps     = partials ** (-amp_decay)

    r_grid = np.linspace(1.0, 2.0, 800)
    curve  = sonance_curve(
        freqs_fixed=partials,    amps_fixed=amps,
        freqs_template=partials, amps_template=amps,
        r_grid=r_grid, k=k,
    )

    s_pure = sonance(r_grid, k=k)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.plot(r_grid, curve, color='#cc8844', linewidth=1.6,
            label=f'Timbre armónico (parciales={n_partials}, decay={amp_decay:.1f})')
    ax.plot(r_grid, s_pure / s_pure.max() * curve.max(),
            color='#6688cc', linewidth=1.0, alpha=0.6, linestyle='--',
            label='Tono puro (norm.)')

    _annotate_intervals(ax, y_top=-curve.max() * 0.06)
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('r  (desplazamiento)')
    ax.set_ylabel('Sonancia de timbre')
    ax.set_title(f'Curva de sonancia de timbre  —  k={k}', fontsize=10, color='#bbbbbb')
    ax.legend(fontsize=8, framealpha=0.25)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


_w_k6    = widgets.IntSlider(value=20, min=5, max=100, step=5,
                               description='k', style={'description_width': '100px'},
                               layout=widgets.Layout(width='420px'))
_w_npart = widgets.IntSlider(value=6, min=1, max=16, step=1,
                               description='parciales', style={'description_width': '100px'},
                               layout=widgets.Layout(width='420px'))
_w_decay = widgets.FloatSlider(value=1.0, min=0.0, max=3.0, step=0.1,
                                 description='decaimiento', style={'description_width': '100px'},
                                 layout=widgets.Layout(width='420px'))

display(widgets.VBox([_w_k6, _w_npart, _w_decay]))
widgets.interactive_output(_plot_timbre_curve, {
    'k': _w_k6, 'n_partials': _w_npart, 'amp_decay': _w_decay,
})

# %% ── 7. TABLA DE INTERVALOS ─────────────────────────────────────────────────
# Resumen numérico: convergencia teórica, exacta y discreta.

def _table_intervals(k, N):
    header = f"{'Int.':<6} {'p/q':<8} {'½(1/p+1/q)':<14} {'exacto N→∞':<14} {'discreto':>10}"
    print(header)
    print('─' * len(header))
    for name, (p, q) in INTERVALS.items():
        if p / q <= 1 or p / q > 2:
            continue
        limit = 0.5 * (1/p + 1/q)
        exact = sonance_exact_reciprocal(p, q, k=k)
        disc  = float(sonance(p / q, k=k, N=N))
        print(f"{name:<6} {p}/{q:<6} {limit:<14.6f} {exact:<14.6f} {disc:>10.6f}")


_w_k7 = widgets.IntSlider(value=20, min=5, max=200, step=5,
                            description='k', style={'description_width': '40px'},
                            layout=widgets.Layout(width='360px'))
_w_N7 = widgets.IntSlider(value=500, min=100, max=2000, step=100,
                            description='N', style={'description_width': '40px'},
                            layout=widgets.Layout(width='360px'))

display(widgets.HBox([_w_k7, _w_N7]))
widgets.interactive_output(_table_intervals, {'k': _w_k7, 'N': _w_N7})
