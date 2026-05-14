# %%
"""
Notebook-style explorer for continuous sonance.

Run in an editor with cell support or as a script:

    python notebooks/continuous_sonance_explorer.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))


# %%
# Global parameters
t_max = 64.0
n_t = 20_000
r_min = 1.0
r_max = 2.0
n_r = 2_000
lam = 0.08
beta = 24.0
max_denominator = 10

t_grid = np.linspace(0.0, t_max, n_t)
r_grid = np.linspace(r_min, r_max, n_r)

reference_intervals = {
    "1/1": 1.0,
    "16/15": 16 / 15,
    "6/5": 6 / 5,
    "5/4": 5 / 4,
    "4/3": 4 / 3,
    "45/32": 45 / 32,
    "3/2": 3 / 2,
    "5/3": 5 / 3,
    "15/8": 15 / 8,
    "2/1": 2.0,
}

focus_ratio = 2 ** (7 / 12)
focus_window = 0.06


# %%
# Core definitions
def recurrence(r: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    return 0.5 * np.cos(2 * np.pi * t) + 0.5 * np.cos(2 * np.pi * r * t)


def recurrence_salience(corr: np.ndarray, beta_value: float) -> np.ndarray:
    return np.exp(beta_value * (corr - 1.0))


def recurrence_weight(
    r: float, t: np.ndarray, lam_value: float, beta_value: float
) -> np.ndarray:
    return np.exp(-lam_value * t) * recurrence_salience(
        recurrence(r, t), beta_value
    )


def sonance_continuous(
    r: float, t: np.ndarray, lam_value: float, beta_value: float
) -> float:
    weight = recurrence_weight(r, t, lam_value, beta_value)
    return lam_value * np.trapezoid(weight, t)


def t_continua(r: float, t: np.ndarray, lam_value: float, beta_value: float) -> float:
    weight = recurrence_weight(r, t, lam_value, beta_value)
    norm = np.trapezoid(weight, t)
    return np.trapezoid(t * weight, t) / norm


def sonance_profile(
    ratios: np.ndarray, t: np.ndarray, lam_value: float, beta_value: float
) -> tuple[np.ndarray, np.ndarray]:
    sonance = np.empty_like(ratios)
    times = np.empty_like(ratios)
    for index, ratio in enumerate(ratios):
        weight = recurrence_weight(float(ratio), t, lam_value, beta_value)
        norm = np.trapezoid(weight, t)
        sonance[index] = lam_value * norm
        times[index] = np.trapezoid(t * weight, t) / norm
    return sonance, times


# %%
# Rational helpers
def simple_rationals(max_q: int) -> list[Fraction]:
    fractions: set[Fraction] = set()
    for q in range(1, max_q + 1):
        for p in range(q, 2 * q + 1):
            frac = Fraction(p, q)
            if 1 <= frac <= 2:
                fractions.add(frac)
    return sorted(fractions)


def continued_fraction_coefficients(x: float, max_terms: int = 12) -> list[int]:
    coeffs: list[int] = []
    value = x
    for _ in range(max_terms):
        integer = int(np.floor(value))
        coeffs.append(integer)
        fractional = value - integer
        if np.isclose(fractional, 0.0):
            break
        value = 1.0 / fractional
    return coeffs


def convergents_from_cf(coeffs: list[int]) -> list[Fraction]:
    result: list[Fraction] = []
    for end in range(1, len(coeffs) + 1):
        frac = Fraction(coeffs[end - 1], 1)
        for coeff in reversed(coeffs[: end - 1]):
            frac = coeff + Fraction(1, frac)
        result.append(frac)
    return result


def convergents_near_ratio(x: float, max_terms: int = 8, max_q: int = 64) -> list[Fraction]:
    convergents = convergents_from_cf(continued_fraction_coefficients(x, max_terms))
    return [frac for frac in convergents if 1 <= frac <= 2 and frac.denominator <= max_q]


def show_or_close(fig: plt.Figure) -> None:
    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()


# %%
# Compute base profiles
sonance_values, time_values = sonance_profile(r_grid, t_grid, lam, beta)
simple_ratios = simple_rationals(max_denominator)
focus_convergents = convergents_near_ratio(focus_ratio)


# %%
# 1) Autocorrelation examples
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
for label, ratio in reference_intervals.items():
    mask = t_grid <= 16.0
    ax.plot(t_grid[mask], recurrence(ratio, t_grid[mask]), linewidth=1.2, label=label)

ax.set_title("Autocorrelacion analitica R_r(t)")
ax.set_xlabel("t (periodos de la fundamental)")
ax.set_ylabel("R_r(t)")
ax.set_ylim(-1.05, 1.05)
ax.grid(alpha=0.25)
ax.legend(ncol=2)
show_or_close(fig)


# %%
# 2) Continuous sonance profile S(r)
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
ax.plot(r_grid, sonance_values, color="tab:blue", linewidth=2.0)

for frac in simple_ratios:
    x = float(frac)
    index = int(np.abs(r_grid - x).argmin())
    ax.scatter(r_grid[index], sonance_values[index], color="tab:red", s=14, zorder=3)

for label in [Fraction(1, 1), Fraction(5, 4), Fraction(4, 3), Fraction(3, 2), Fraction(2, 1)]:
    x = float(label)
    index = int(np.abs(r_grid - x).argmin())
    ax.annotate(
        f"{label.numerator}/{label.denominator}",
        (r_grid[index], sonance_values[index]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
    )

ax.set_title(f"S(r) continua, lambda={lam:.2f}, beta={beta:.1f}")
ax.set_xlabel("ratio r")
ax.set_ylabel("S(r)")
ax.grid(alpha=0.25)
show_or_close(fig)


# %%
# 3) Associated effective time t_continua(r)
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
ax.plot(r_grid, time_values, color="tab:green", linewidth=2.0)

for frac in simple_ratios:
    x = float(frac)
    index = int(np.abs(r_grid - x).argmin())
    ax.scatter(r_grid[index], time_values[index], color="tab:red", s=14, zorder=3)

ax.set_title(f"t_continua(r), lambda={lam:.2f}, beta={beta:.1f}")
ax.set_xlabel("ratio r")
ax.set_ylabel("tiempo efectivo")
ax.grid(alpha=0.25)
show_or_close(fig)


# %%
# 4) Parameter sweep for S(r)
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

lambda_values = [lam / 2.0, lam, lam * 2.0]
beta_values = [max(beta / 2.0, 1.0), beta, beta * 2.0]

for lam_value in lambda_values:
    profile, _ = sonance_profile(r_grid, t_grid, lam_value, beta)
    ax.plot(
        r_grid,
        profile,
        linewidth=1.3,
        label=fr"$\lambda={lam_value:.2f}, \beta={beta:.0f}$",
    )

for beta_value in beta_values:
    if np.isclose(beta_value, beta):
        continue
    profile, _ = sonance_profile(r_grid, t_grid, lam, beta_value)
    ax.plot(
        r_grid,
        profile,
        linewidth=1.1,
        linestyle="--",
        label=fr"$\lambda={lam:.2f}, \beta={beta_value:.0f}$",
    )

ax.set_title("Sensibilidad paramétrica de S(r)")
ax.set_xlabel("ratio r")
ax.set_ylabel("S(r)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
show_or_close(fig)


# %%
# 5) Local structure around a focus ratio and its convergents
local_mask = (r_grid >= focus_ratio - focus_window) & (r_grid <= focus_ratio + focus_window)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

axes[0].plot(r_grid[local_mask], sonance_values[local_mask], color="tab:blue", linewidth=2.0)
axes[0].axvline(focus_ratio, color="black", linestyle=":", linewidth=1.0)

for frac in focus_convergents:
    x = float(frac)
    if focus_ratio - focus_window <= x <= focus_ratio + focus_window:
        index = int(np.abs(r_grid - x).argmin())
        axes[0].scatter(r_grid[index], sonance_values[index], color="tab:orange", s=24)
        axes[0].annotate(
            f"{frac.numerator}/{frac.denominator}",
            (r_grid[index], sonance_values[index]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

axes[0].set_title(
    "Zoom local de S(r) alrededor de 2^(7/12) con convergentes de fracciones continuas"
)
axes[0].set_xlabel("ratio r")
axes[0].set_ylabel("S(r)")
axes[0].grid(alpha=0.25)

axes[1].plot(t_grid[t_grid <= 24], recurrence(focus_ratio, t_grid[t_grid <= 24]), color="tab:purple")
axes[1].set_title("R_r(t) para el ratio focal 2^(7/12)")
axes[1].set_xlabel("t (periodos de la fundamental)")
axes[1].set_ylabel("R_r(t)")
axes[1].set_ylim(-1.05, 1.05)
axes[1].grid(alpha=0.25)
show_or_close(fig)


# %%
# 6) Numeric summary for selected intervals
selected = [1.0, 5 / 4, 4 / 3, 3 / 2, 45 / 32, 2 ** (7 / 12), 2.0]
summary = []
for ratio in selected:
    summary.append(
        {
            "ratio": ratio,
            "S": sonance_continuous(ratio, t_grid, lam, beta),
            "t_continua": t_continua(ratio, t_grid, lam, beta),
            "best_simple": Fraction(ratio).limit_denominator(16),
        }
    )

summary
