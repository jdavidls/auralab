"""
Sonance S_k(r) — derived from wave autocorrelation, defined for continuous r.

Derivation (doc/notebooks/sonancia_desarrollo_matematico.md):

    s(t) = sin(2π f₁ t) + sin(2π f₂ t),   r = f₂/f₁

    R(τ)    = ½cos(2πf₁τ) + ½cos(2πf₂τ)      [autocorrelation]
    R(n)    = cos²(πrn)                         [integer lags τ=nT₁]
    S_k(r)  = Σ_{n=1}^N cos^{2k}(πrn) / n      [sonance sum §4.1]

Reciprocal regularised form (doc/sonancia_sintesis_compacta.md §20):

    S_k*(r; N) = Σ_{n=1}^N  w_n cos^{2k}(πrn),   w_1=½, w_n=1/n (n≥2)

    𝒮_k(r; N) = ½ [ S_k*(r;N) + S_k*(1/r;N) ]     symmetric: 𝒮_k(r) = 𝒮_k(1/r)

The reciprocal form accepts any r ∈ ℝ⁺ without requiring a p/q representation.
For rational r = p/q (gcd=1), the two branches detect closure times q and p
respectively, recovering both arithmetically from the continuous parameter r.

Normalised by W_N = Σ w_n = H_N − ½, the function is bounded and converges
as N → ∞.

Fourier-Zeta form (§13, convergent, N=∞):

    cos^{2k}(πx) = Σ_{ℓ=-k}^k  c_ℓ^(k) e^{2πiℓx},   c_ℓ = C(2k,k+ℓ)/4^k

    𝒮_k(r; s) = ½ Σ_{ℓ=-k}^k c_ℓ [ Li_s(e^{2πiℓr}) + Li_s(e^{2πiℓ/r}) ]

At s → 1⁺ the Laurent expansion reads  A_k(q)/(s−1) + C_k(p,q) + O(s−1),
where A_k(q) → 1/q and C_k(p,q) encodes the pre-resonant signature.
"""
from __future__ import annotations

import numpy as np
from math import comb


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def _weights(N: int) -> np.ndarray:
    """Modified harmonic weights: w_1 = 1/2, w_n = 1/n for n ≥ 2."""
    w = 1.0 / np.arange(1, N + 1, dtype=float)
    w[0] = 0.5
    return w


def _W(N: int) -> float:
    """Sum of modified weights W_N = H_N − 1/2."""
    return float(_weights(N).sum())


# ---------------------------------------------------------------------------
# Single-branch (directional) sonance
# ---------------------------------------------------------------------------

def sonance_star(
    r: float | np.ndarray,
    k: int = 20,
    N: int | None = None,
) -> np.ndarray:
    """Directional regularised sonance S_k*(r; N) = Σ w_n cos^{2k}(πrn).

    The first-lag weight is w_1 = 1/2 instead of 1 to suppress the dc
    overcount. Equivalent to S_k(r;N) − ½cos^{2k}(πr).

    Not symmetric under r ↔ 1/r. Use `sonance` for the symmetric form.

    Parameters
    ----------
    r : float or (R,) array — frequency ratios.
    k : int — resolution (k ≳ q² resolves denominator q).
    N : int or None — lag count; None → max(500, 50*k).

    Returns
    -------
    (R,) array, unnormalised (grows as W_N ≈ H_N).
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    N = N or _default_N(k)
    w = _weights(N)                                     # (N,)
    n = np.arange(1, N + 1, dtype=float)
    cos2k = np.cos(np.pi * r[:, None] * n[None, :]) ** (2 * k)  # (R, N)
    return (cos2k * w).sum(axis=1)                      # (R,)


# ---------------------------------------------------------------------------
# Reciprocal symmetric sonance — primary interface
# ---------------------------------------------------------------------------

def sonance(
    r: float | np.ndarray,
    k: int = 20,
    N: int | None = None,
) -> float | np.ndarray:
    """Reciprocal regularised sonance 𝒮_k(r; N), normalised ∈ [0, 1].

    𝒮_k(r; N) = ½ [S_k*(r;N) + S_k*(1/r;N)] / W_N

    Properties
    ----------
    * Defined for any r > 0; no p/q representation needed.
    * Symmetric: 𝒮_k(r) = 𝒮_k(1/r).
    * Converges as N → ∞ to ½[Res_k(q) + Res_k(p)] for rational r = p/q.
    * As k → ∞: → ½(1/q + 1/p) = (p+q)/(2pq), recovering both p and q.
    * k ≳ q² needed to resolve denominator q (and k ≳ p² for numerator p).

    Parameters
    ----------
    r : float or (R,) array — ratio(s). Typically in (1, 2] (one octave).
    k : int — resolution parameter.
    N : int or None — lag count; None → max(500, 50*k).

    Returns
    -------
    float or (R,) array in [0, 1].
    """
    scalar = np.ndim(r) == 0
    r = np.atleast_1d(np.asarray(r, dtype=float))
    N = N or _default_N(k)
    W = _W(N)

    Sf = sonance_star(r,       k, N)
    Sr = sonance_star(1.0 / r, k, N)
    result = 0.5 * (Sf + Sr) / W

    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Exact N→∞ closed form (for rational r = p/q)
# ---------------------------------------------------------------------------

def sonance_exact(q: int, k: int = 20) -> float:
    """Exact N→∞ limit Res_k(q) for the *directional* branch S_k*(r;N)/W_N.

    Res_k(q) = (1/q) Σ_{m=0}^{q-1} cos^{2k}(πm/q)   →  1/q  as k → ∞

    Depends only on the denominator q (not the numerator p).
    For the full reciprocal sonance at r = p/q:

        lim_{N→∞} 𝒮_k(p/q; N) = ½ [Res_k(q) + Res_k(p)]

    Parameters
    ----------
    q : int — denominator of the reduced fraction.
    k : int — resolution parameter.
    """
    m = np.arange(q, dtype=float)
    return float(np.sum(np.cos(np.pi * m / q) ** (2 * k)) / q)


def sonance_exact_reciprocal(p: int, q: int, k: int = 20) -> float:
    """Exact N→∞ limit of the reciprocal sonance 𝒮_k(p/q) = ½[Res_k(q)+Res_k(p)].

    As k → ∞: → ½(1/q + 1/p) = (p+q)/(2pq).
    """
    return 0.5 * (sonance_exact(q, k) + sonance_exact(p, k))


# ---------------------------------------------------------------------------
# Fourier-Zeta convergent form (N = ∞)
# ---------------------------------------------------------------------------

def sonance_fourier(
    r: float | np.ndarray,
    k: int = 20,
    reg: float = 1e-9,
    reciprocal: bool = True,
) -> float | np.ndarray:
    """Convergent N→∞ sonance via Fourier expansion of cos^{2k}.

    For each Fourier mode j = 1, …, k the identity

        Σ_{n=1}^∞ cos(2πjrn)/n  =  −ln|2 sin(πjr)|   (jr ∉ ℤ)

    turns the infinite sum into a closed-form finite sum of k terms.
    This is the "finite part" after subtracting the logarithmically
    divergent piece  c_0 · H_N  (equivalently, the constant term of
    the Laurent expansion of 𝒮_k(r; s) at s = 1).

    Non-reciprocal form (j single branch):
        S_k^F(r) = −Σ_{j=1}^k a_j ln|2 sin(πjr)|
        where a_j = 2 C(2k, k−j) / 4^k

    Reciprocal form (default):
        𝒮_k^F(r) = −½ Σ_{j=1}^k a_j [ln|2 sin(πjr)| + ln|2 sin(πj/r)|]

    Both forms have logarithmic singularities at rationals r = p/q:
    the j = q term diverges (+∞) on the r-branch when q|j, and
    the j = p term diverges on the 1/r-branch. This is correct:
    consonant ratios are detected as singularities, with strength
    proportional to the Fourier coefficient a_q → 0 for large q.

    Parameters
    ----------
    r       : float or (R,) array.
    k       : int — resolution (number of Fourier terms).
    reg     : float — regularisation floor to avoid log(0).
    reciprocal : bool — if True (default) include the 1/r branch.

    Returns
    -------
    float or (R,) array. Finite for irrational r; +∞ at consonant rationals.
    """
    scalar = np.ndim(r) == 0
    r = np.atleast_1d(np.asarray(r, dtype=float))
    inv4k = 4.0 ** (-k)

    result = np.zeros(r.shape)
    for j in range(1, k + 1):
        a_j = 2.0 * comb(2 * k, k - j) * inv4k
        s_fwd = np.abs(2.0 * np.sin(np.pi * j * r))
        s_fwd = np.maximum(s_fwd, reg)
        result -= a_j * np.log(s_fwd)

        if reciprocal:
            s_bwd = np.abs(2.0 * np.sin(np.pi * j / r))
            s_bwd = np.maximum(s_bwd, reg)
            result -= a_j * np.log(s_bwd)

    if reciprocal:
        result *= 0.5

    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Multi-tone / timbre extension (§23–24, v2 doc)
# ---------------------------------------------------------------------------

def sonance_spectrum(
    freqs: list[float] | np.ndarray,
    amps: list[float] | np.ndarray,
    k: int = 20,
    N: int | None = None,
) -> float:
    """Sonance of a spectrum/timbre as amplitude-weighted sum over pairs.

    Son_k(ℱ) = Σ_{i<j} w_ij · 𝒮_k(fⱼ/fᵢ)

    where w_ij = Aᵢ² Aⱼ² / Σ_{m<n} Aₘ² Aₙ²  (normalised energy weight §26.3).

    Parameters
    ----------
    freqs : (M,) — partial frequencies (any units; only ratios matter).
    amps  : (M,) — partial amplitudes.
    k, N  : resolution and lag count (passed to `sonance`).

    Returns
    -------
    float — total normalised sonance.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps  = np.asarray(amps,  dtype=float)
    M = len(freqs)

    # Build pair ratios and amplitude weights
    ratios, weights = [], []
    for i in range(M):
        for j in range(i + 1, M):
            rho = freqs[j] / freqs[i]
            w   = (amps[i] * amps[j]) ** 2
            ratios.append(rho)
            weights.append(w)

    if not ratios:
        return 0.0

    ratios  = np.array(ratios)
    weights = np.array(weights)
    weights = weights / weights.sum()          # normalise
    s_vals  = sonance(ratios, k=k, N=N)
    return float(np.dot(weights, s_vals))


def sonance_curve(
    freqs_fixed: list[float] | np.ndarray,
    amps_fixed: list[float] | np.ndarray,
    freqs_template: list[float] | np.ndarray,
    amps_template: list[float] | np.ndarray,
    r_grid: np.ndarray,
    k: int = 20,
    N: int | None = None,
) -> np.ndarray:
    """Sonance curve of a timbre displaced by interval r (§24 v2 doc).

    Son_k^{T,U}(r) = Σ_{a,b} Aₐ² Bᵦ² 𝒮_k(βᵦ r / αₐ)

    where αₐ are the partial ratios of the fixed timbre T and βᵦ those of
    template U. The curve shows which displacement intervals r are favoured
    by recurrence between the two timbres.

    Parameters
    ----------
    freqs_fixed    : (M,) — frequency multiples of fixed timbre (e.g. [1,2,3,4]).
    amps_fixed     : (M,) — amplitudes of fixed timbre.
    freqs_template : (L,) — frequency multiples of template timbre.
    amps_template  : (L,) — amplitudes of template timbre.
    r_grid         : (K,) — displacement ratios to evaluate.
    k, N           : resolution and lag count.

    Returns
    -------
    (K,) — sonance value at each displacement ratio.
    """
    fa = np.asarray(freqs_fixed,    dtype=float)
    Aa = np.asarray(amps_fixed,     dtype=float)
    fb = np.asarray(freqs_template, dtype=float)
    Bb = np.asarray(amps_template,  dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)

    curve = np.zeros(len(r_grid))
    total_w = 0.0
    for i, (alpha, A) in enumerate(zip(fa, Aa)):
        for j, (beta, B) in enumerate(zip(fb, Bb)):
            w = (A * B) ** 2
            total_w += w
            rho = beta * r_grid / alpha    # (K,)
            curve += w * sonance(rho, k=k, N=N)

    return curve / total_w if total_w > 0 else curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_N(k: int) -> int:
    return max(500, 50 * k)
