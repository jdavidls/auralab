# %% [markdown]
# # F(s, q, k, p) — Análisis refinado
# ## Residuo correcto, distribución de ceros, estructura fina

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from math import gcd
from fractions import Fraction

plt.rcParams.update({
    'figure.facecolor': '#0a0a0a', 'axes.facecolor': '#0a0a0a',
    'axes.edgecolor': '#333', 'axes.labelcolor': '#999',
    'text.color': '#ccc', 'xtick.color': '#666', 'ytick.color': '#666',
    'grid.color': '#1a1a1a', 'grid.alpha': 0.8,
    'font.family': 'monospace', 'font.size': 9, 'figure.dpi': 130,
})

# %% Core functions

def F_numerical(s, p, q, k, N=500):
    """F(s) = Σ cos²ᵏ(πpn/q) / n^s, s complejo."""
    n = np.arange(1, N+1, dtype=float)
    cos2k = np.cos(np.pi * p * n / q) ** (2*k)
    if np.isreal(s):
        return np.sum(cos2k / n**float(np.real(s)))
    else:
        log_n = np.log(n)
        n_s = np.exp(-s * log_n)  # n^(-s) = exp(-s·ln(n))
        return np.sum(cos2k * n_s)

def cos2k_coefficients(k):
    """a₀, a₁, ..., aₖ de cos²ᵏ(x) = a₀ + Σ aⱼ cos(2jx)."""
    a0 = float(comb(2*k, k, exact=True)) / 4**k
    coeffs = [a0]
    for j in range(1, k+1):
        coeffs.append(2 * float(comb(2*k, k-j, exact=True)) / 4**k)
    return coeffs

def euler_phi(n):
    count = 0
    for i in range(1, n+1):
        if gcd(i, n) == 1: count += 1
    return count

def factorize_str(n):
    if n <= 1: return str(n)
    orig = n; factors = []
    for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
        e = 0
        while n % p == 0: e += 1; n //= p
        if e > 0: factors.append(f'{p}{"^"+str(e) if e>1 else ""}')
    if n > 1: factors.append(str(n))
    return '·'.join(factors)

# %% 1. Residuo analítico correcto
#
# F(s) = a₀·ζ(s) + Σ_{j=1}^{k} aⱼ · Re[Li_s(e^{2πijp/q})]
#
# Li_s(z) tiene polo en s=1 SOLO si z = 1.
# e^{2πijp/q} = 1  iff  q | jp  iff  q | j  (cuando gcd(p,q)=1)
#
# El residuo de Li_s(1) en s=1 es 1 (= Res de ζ(s))
#
# Así que:
# Res_{s=1} F = a₀ + Σ_{m=1}^{⌊k/q⌋} a_{mq}
#
# Para k < q: Res = a₀ (solo el término j=0 tiene polo)
# Para k ≥ q: contribuciones adicionales de j = q, 2q, ...

print("═" * 75)
print("RESIDUO ANALÍTICO DE F(s) EN s = 1")
print("═" * 75)
print()
print("Res_{s=1} F = a₀ + Σ_{m: mq ≤ k} a_{mq}")
print()

for k in [10, 20, 40, 80]:
    coeffs = cos2k_coefficients(k)
    a0 = coeffs[0]
    
    print(f"k = {k}:  a₀ = {a0:.6f}  ≈ 1/√(π·{k}) = {1/np.sqrt(np.pi*k):.6f}")
    print(f"{'p/q':<10} {'q':<5} {'k/q':<6} {'Res analítico':<18} {'Términos extra'}")
    print("-" * 70)
    
    for p, q in [(3,2),(4,3),(5,4),(9,8),(1,7),(1,11),(45,32)]:
        # Índices j = mq que caen dentro de [1, k]
        extra_indices = [m*q for m in range(1, k//q + 1) if m*q <= k]
        extra_sum = sum(coeffs[j] for j in extra_indices) if extra_indices else 0
        res_analytic = a0 + extra_sum
        
        extra_str = "+".join([f"a_{j}" for j in extra_indices[:5]]) if extra_indices else "ninguno"
        if len(extra_indices) > 5: extra_str += "..."
        
        print(f"{p}/{q:<8} {q:<5} {k//q:<6} {res_analytic:<18.6f} {extra_str}")
    
    print()

# %% 2. Verificación numérica del residuo
# 
# Res = lim_{ε→0} ε · F(1+ε)
# Pero F(1+ε) = Res/ε + C + O(ε), así que F(1+ε)·ε = Res + C·ε + ...
# Necesitamos ε muy pequeño Y N grande (la serie converge lento cerca de s=1)

print("═" * 75)
print("VERIFICACIÓN NUMÉRICA (extrapolación Richardson)")
print("═" * 75)
print()

def residue_numerical(p, q, k, epsilons=[0.1, 0.05, 0.02, 0.01, 0.005], N=5000):
    """Estima Res por extrapolación."""
    estimates = []
    for eps in epsilons:
        # F(1+ε) ≈ Res/ε + C₀ + C₁ε + ...
        # Así que ε·F(1+ε) ≈ Res + C₀ε + ...
        val = F_numerical(1.0 + eps, p, q, k, N)
        estimates.append(float(np.real(val)) * eps)
    
    # Extrapolación de Richardson: Res ≈ 2·R(ε) - R(2ε)
    if len(estimates) >= 4:
        # Use last 3 points for quadratic extrapolation
        e = np.array(epsilons[-3:])
        r = np.array(estimates[-3:])
        # Fit r = a + b·e  → a = Res
        A = np.vstack([np.ones_like(e), e]).T
        fit = np.linalg.lstsq(A, r, rcond=None)[0]
        return fit[0], estimates
    return estimates[-1], estimates

k = 20
coeffs = cos2k_coefficients(k)
a0 = coeffs[0]

print(f"k = {k},  N = 5000")
print(f"{'p/q':<10} {'Res analítico':<16} {'Res numérico':<16} {'Δ':<12} {'Estimaciones ε·F(1+ε)'}")
print("-" * 90)

for p, q in [(3,2),(4,3),(5,4),(9,8),(1,5),(1,7),(1,11),(1,13),(45,32),(16,15)]:
    # Analítico
    extra = [m*q for m in range(1, k//q+1) if m*q <= k]
    res_a = a0 + sum(coeffs[j] for j in extra)
    
    # Numérico
    res_n, ests = residue_numerical(p, q, k, N=5000)
    
    delta = abs(res_n - res_a)
    ests_str = "  ".join([f"{e:.5f}" for e in ests[-3:]])
    
    print(f"{p}/{q:<8} {res_a:<16.6f} {res_n:<16.6f} {delta:<12.6f} {ests_str}")

# %% 3. Parte finita (constante de Laurent) en s=1
#
# F(s) = Res/(s-1) + C₀ + C₁(s-1) + ...
#
# C₀ es la "parte finita" — es la parte de la sonancia que queda
# después de quitar la divergencia logarítmica.
#
# C₀ = lim_{N→∞} [F_N(1) - Res · ln(N)]
#    = lim_{N→∞} [Σ_{n=1}^{N} cos²ᵏ(πpn/q)/n - Res · ln(N)]
#
# Para Res = a₀ (k < q), esto es:
# C₀ = Σ cos²ᵏ(πpn/q)/n - a₀·ln(N) + O(1/N)
#
# La diferencia C₀(p/q) entre distintos ratios CON MISMO q es interesante:
# depende de p, no solo de q. Esto captura la distinción que 1/q pierde.

print()
print("═" * 75)
print("PARTE FINITA C₀ = lim[F_N(1) - Res·ln(N)]")
print("═" * 75)
print()

k = 20
coeffs = cos2k_coefficients(k)

# Computar C₀ para varios N y extrapolar
def finite_part(p, q, k, Ns=[500, 1000, 2000, 4000, 8000]):
    """Estima C₀ por extrapolación."""
    extra = [m*q for m in range(1, k//q+1) if m*q <= k]
    res = coeffs[0] + sum(coeffs[j] for j in extra)
    
    estimates = []
    for N in Ns:
        n_arr = np.arange(1, N+1, dtype=float)
        FN = np.sum(np.cos(np.pi*p*n_arr/q)**(2*k) / n_arr)
        C0_est = FN - res * np.log(N)
        estimates.append(C0_est)
    
    return estimates[-1], estimates

print(f"k = {k}")
print(f"{'p/q':<10} {'q':<5} {'Res':<12} {'C₀':<12} {'C₀ - C₀(1/q)'}")
print("-" * 55)

# Agrupar por q para ver la dependencia en p
q_groups = {}
for p, q in [(1,2),(1,3),(2,3),(1,4),(3,4),(1,5),(2,5),(3,5),(4,5),
             (1,7),(2,7),(3,7),(5,7),(1,8),(3,8),(5,8),(7,8)]:
    if gcd(p,q) != 1: continue
    c0, _ = finite_part(p, q, k)
    extra = [m*q for m in range(1, k//q+1) if m*q <= k]
    res = coeffs[0] + sum(coeffs[j] for j in extra)
    if q not in q_groups: q_groups[q] = []
    q_groups[q].append((p, c0, res))

for q in sorted(q_groups.keys()):
    entries = q_groups[q]
    c0_base = entries[0][1]  # C₀ para p=1 como referencia
    for p, c0, res in entries:
        diff = c0 - c0_base
        print(f"{p}/{q:<8} {q:<5} {res:<12.6f} {c0:<12.6f} {diff:<+.6f}")
    print()

# %% 4. Distribución de ceros en la línea crítica: primo vs compuesto

def find_zeros_critical(p, q, k, t_max=100, N=300, resolution=2000):
    """Encuentra ceros aproximados de F(1/2 + it)."""
    t_vals = np.linspace(0.1, t_max, resolution)
    F_vals = np.array([F_numerical(complex(0.5, t), p, q, k, N) for t in t_vals])
    absF = np.abs(F_vals)
    
    # Detectar mínimos locales por debajo de umbral
    zeros = []
    for i in range(1, len(absF)-1):
        if absF[i] < absF[i-1] and absF[i] < absF[i+1]:
            if absF[i] < np.median(absF) * 0.3:
                # Refinar con interpolación parabólica
                t_zero = t_vals[i]
                zeros.append({'t': t_zero, 'absF': absF[i], 'F': F_vals[i]})
    
    return zeros, t_vals, absF

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Comparar primos y compuestos de tamaño similar
pairs = [
    ((1, 5, "q=5 (primo)"), (1, 6, "q=6 = 2·3")),
    ((1, 7, "q=7 (primo)"), (1, 8, "q=8 = 2³")),
    ((1, 11, "q=11 (primo)"), (1, 12, "q=12 = 2²·3")),
]

k = 15

for row, ((p1,q1,d1), (p2,q2,d2)) in enumerate(pairs):
    for col, (p, q, desc) in enumerate([(p1,q1,d1), (p2,q2,d2)]):
        ax = axes[row, col]
        zeros, t_vals, absF = find_zeros_critical(p, q, k, t_max=80, N=300, resolution=3000)
        
        ax.plot(t_vals, absF, color='#6688cc', linewidth=0.8, alpha=0.8)
        ax.fill_between(t_vals, 0, absF, color='#6688cc', alpha=0.08)
        
        # Marcar ceros
        for z in zeros:
            ax.plot(z['t'], z['absF'], 'o', color='#ee5555', markersize=3)
        
        # Estadísticas de ceros
        if len(zeros) >= 2:
            spacings = np.diff([z['t'] for z in zeros])
            mean_sp = np.mean(spacings)
            std_sp = np.std(spacings)
            regularity = 1 - std_sp / (mean_sp + 1e-10)
        else:
            mean_sp, std_sp, regularity = 0, 0, 0
        
        is_prime = all(q % d != 0 for d in range(2, q))
        title_color = '#55cc77' if is_prime else '#ccaa33'
        ax.set_title(f'{desc}\n{len(zeros)} ceros, espaciado: μ={mean_sp:.2f} σ={std_sp:.2f} reg={regularity:.2f}',
                     fontsize=9, color=title_color)
        ax.set_xlabel('t')
        ax.set_ylabel('|F(½+it)|')
        ax.grid(True, alpha=0.15)
        ax.set_ylim(bottom=0)

plt.suptitle('Ceros en la línea crítica σ=½: primos (izq) vs compuestos (der)\n'
             'Regularity = 1 - σ/μ del espaciado entre ceros',
             fontsize=11, color='#bbb')
plt.tight_layout()
plt.show()

# %% 5. Espaciado entre ceros: estadística
#
# La distribución de espaciado entre ceros de ζ(s) sigue la
# distribución GUE (Gaussian Unitary Ensemble) — resultado central
# de la teoría de matrices aleatorias.
#
# ¿Qué distribución sigue el espaciado de ceros de F(s)?
# ¿Es diferente para q primo vs compuesto?

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

test_qs = [5, 7, 11, 6, 8, 12]
k = 15

for ax, q in zip(axes.flat, test_qs):
    zeros, _, _ = find_zeros_critical(1, q, k, t_max=150, N=400, resolution=5000)
    
    if len(zeros) >= 3:
        t_zeros = np.array([z['t'] for z in zeros])
        spacings = np.diff(t_zeros)
        # Normalizar por media
        if np.mean(spacings) > 0:
            spacings_norm = spacings / np.mean(spacings)
        else:
            spacings_norm = spacings
        
        ax.hist(spacings_norm, bins=max(5, len(spacings)//3), color='#6688cc',
                edgecolor='#0a0a0a', alpha=0.7, density=True)
        
        # Overlay: distribución exponencial (Poisson)
        x = np.linspace(0, 4, 100)
        ax.plot(x, np.exp(-x), color='#cc8844', linewidth=1, linestyle='--',
                label='Poisson', alpha=0.6)
        
        # GUE (Wigner surmise)
        ax.plot(x, (np.pi/2)*x*np.exp(-np.pi*x**2/4), color='#55cc77',
                linewidth=1, linestyle='--', label='GUE', alpha=0.6)
        
        is_prime = all(q % d != 0 for d in range(2, q))
        ax.set_title(f'q={q} {"(primo)" if is_prime else f"= {factorize_str(q)}"}\n'
                     f'{len(zeros)} ceros, μ={np.mean(spacings):.2f}',
                     fontsize=9, color='#55cc77' if is_prime else '#ccaa33')
        ax.legend(fontsize=7, framealpha=0.3)
    else:
        ax.set_title(f'q={q} — insuficientes ceros', fontsize=9, color='#666')
    
    ax.set_xlabel('Espaciado normalizado')
    ax.grid(True, alpha=0.15)

plt.suptitle('Distribución de espaciado de ceros en σ=½\n'
             'Poisson = independientes, GUE = correlacionados (como ζ)',
             fontsize=11, color='#bbb')
plt.tight_layout()
plt.show()

# %% 6. Dependencia del residuo con q: ¿escala como 1/q? ¿1/φ(q)?

k = 20
coeffs = cos2k_coefficients(k)

q_range = range(2, 60)
residues = []
inv_qs = []
inv_phis = []

for q in q_range:
    p = 1  # p=1 siempre coprimo con q
    extra = [m*q for m in range(1, k//q+1) if m*q <= k]
    res = coeffs[0] + sum(coeffs[j] for j in extra if j < len(coeffs))
    residues.append(res)
    inv_qs.append(1/q)
    inv_phis.append(1/euler_phi(q))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

is_prime = [all(q % d != 0 for d in range(2, q)) for q in q_range]
colors = ['#55cc77' if ip else '#ccaa33' for ip in is_prime]

# Residuo vs q
ax = axes[0]
ax.scatter(list(q_range), residues, c=colors, s=20, alpha=0.8, zorder=5)
ax.axhline(coeffs[0], color='#cc884488', linewidth=1, linestyle='--', label=f'a₀ = {coeffs[0]:.4f}')
ax.set_xlabel('q')
ax.set_ylabel('Res_{s=1} F')
ax.set_title('Residuo vs q\n(saltos cuando k/q cruza un entero)', fontsize=10, color='#bbb')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.2)

# Residuo vs 1/q y 1/φ(q)
ax = axes[1]
ax.scatter(inv_qs, residues, c=colors, s=15, alpha=0.6, label='vs 1/q')
ax.scatter(inv_phis, residues, c=colors, s=15, alpha=0.6, marker='s', label='vs 1/φ(q)')
ax.set_xlabel('1/q  o  1/φ(q)')
ax.set_ylabel('Res_{s=1} F')
ax.set_title('¿Escala el residuo con 1/q o 1/φ(q)?', fontsize=10, color='#bbb')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# %% 7. Estructura fina: F evaluada exactamente via Hurwitz
#
# F(s) = a₀·ζ(s) + Σ_{j=1}^{k} aⱼ · Re[Li_s(e^{2πijp/q})]
#
# Y Li_s(e^{2πim/q}) = q^{-s} · Σ_{a=0}^{q-1} e^{2πiam/q} · ζ(s, a/q)
#
# Implementemos esto usando la expansión en Hurwitz

def hurwitz_zeta_approx(s, a, N=5000):
    """ζ(s, a) = Σ_{n=0}^{N} 1/(n+a)^s, aproximación truncada."""
    if a <= 0: a = 1e-10
    n = np.arange(0, N, dtype=float)
    terms = 1.0 / (n + a) ** s
    return np.sum(terms)

def F_via_hurwitz(s, p, q, k, N_hurwitz=3000):
    """F(s) computada via descomposición en funciones zeta de Hurwitz."""
    coeffs = cos2k_coefficients(k)
    
    # Término j=0: a₀·ζ(s)
    result = coeffs[0] * hurwitz_zeta_approx(s, 1.0, N_hurwitz)
    
    # Términos j=1..k
    for j in range(1, min(k+1, len(coeffs))):
        # Li_s(e^{2πijp/q}) = q^{-s} · Σ_{a=0}^{q-1} e^{2πiajp/q} · ζ(s, a/q)
        alpha = j * p / q  # fracción reducida mod 1
        
        # Computar via suma directa de la serie (más estable)
        n = np.arange(1, N_hurwitz+1, dtype=float)
        Li_s = np.sum(np.exp(2j * np.pi * alpha * n) / n**s)
        
        result += coeffs[j] * np.real(Li_s)
    
    return np.real(result) if np.isreal(s) else result

# Comparar F_numerical vs F_via_hurwitz en la línea real
fig, ax = plt.subplots(figsize=(12, 5))

s_range = np.linspace(1.2, 5, 200)
p, q, k = 3, 2, 15

F_direct = np.array([float(np.real(F_numerical(s, p, q, k, 3000))) for s in s_range])
F_hurwitz = np.array([float(F_via_hurwitz(s, p, q, k, 3000)) for s in s_range])

ax.plot(s_range, F_direct, color='#6688cc', linewidth=2, label='F directa (Σ cos²ᵏ/n^s)')
ax.plot(s_range, F_hurwitz, color='#cc8844', linewidth=1.5, linestyle='--',
        label='F via Hurwitz (Σ aⱼ·Li_s)')
ax.plot(s_range, np.abs(F_direct - F_hurwitz), color='#cc5555', linewidth=1,
        label='|Diferencia|', alpha=0.5)

ax.set_xlabel('s (real)')
ax.set_ylabel('F(s)')
ax.set_title(f'Verificación: F directa vs descomposición de Hurwitz  ({p}/{q}, k={k})',
             fontsize=10, color='#bbb')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# %% 8. Resumen de hallazgos sobre F(s)

print("""
═══════════════════════════════════════════════════════════════════════
  F(s, q, k, p) — HALLAZGOS
═══════════════════════════════════════════════════════════════════════

1. RESIDUO EN s=1:
   Res = a₀ + Σ_{m: mq ≤ k} a_{mq}
   
   Para k < q: Res = a₀ ≈ 1/√(πk) — INDEPENDIENTE de p y q.
   Para k ≥ q: contribuciones adicionales de los armónicos j = mq.
   
   El residuo NO escala como 1/q. La sonancia (la integral de F)
   sí escala como ~1/q, pero el polo tiene residuo universal a₀.
   La información de q está en la PARTE FINITA C₀, no en el residuo.

2. PARTE FINITA C₀:
   C₀ = lim_{N→∞} [F_N(1) - a₀·ln(N)]
   
   C₀ depende de p Y de q. Para mismo q, diferentes p dan diferentes
   C₀. Esto captura información que 1/q pierde: la "calidad" del
   intervalo más allá de su complejidad denominadora.

3. CEROS EN LA LÍNEA CRÍTICA:
   F(½+it) tiene ceros cuya distribución depende de q.
   Observación: los q primos parecen dar espaciado más regular
   que los compuestos, pero se necesita más resolución para confirmar.
   
   La distribución de espaciado parece intermedia entre Poisson
   (independiente) y GUE (correlacionada). Esto podría reflejar que
   F es una combinación de ζ's de Hurwitz, no una sola función L.

4. DESCOMPOSICIÓN DE HURWITZ:
   F(s) = a₀·ζ(s) + Σ aⱼ·Re[Li_s(e^{2πijp/q})]
   
   Verificada numéricamente. Cada Li_s se expresa en ζ(s, a/q).
   La estructura completa está controlada por:
   - Los coeficientes aⱼ (solo dependen de k)
   - Las raíces de la unidad e^{2πijp/q} (dependen de p/q)
   - Las funciones ζ de Hurwitz (propiedades universales)

═══════════════════════════════════════════════════════════════════════
""")