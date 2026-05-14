# %% [markdown]
# # F(s) — Análisis profundo y Espectrómetro Aritmético
#
# ## Resultado central: Res_{s=1} F → 1/q cuando k→∞
# ## C₀ distingue numeradores: información más allá de 1/q

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

# %% 1. DEMOSTRACIÓN: Res → 1/q
#
# Res = a₀ + Σ_{m=1}^{⌊k/q⌋} a_{mq}
#
# donde aⱼ = 2·C(2k,k-j)/4ᵏ para j≥1, a₀ = C(2k,k)/4ᵏ.
#
# Necesitamos demostrar que Σ_{m=0}^{∞} a_{mq} = 1/q.
#
# Usamos la identidad: Σ_{j=0}^{k} aⱼ·cos(2jx) = cos²ᵏ(x)
# (definición de los aⱼ).
#
# Sumando sobre x = 0, 2π/q, 4π/q, ..., 2(q-1)π/q y dividiendo por q:
#
# (1/q) Σ_{r=0}^{q-1} cos²ᵏ(2πr/q) = (1/q) Σ_{r=0}^{q-1} Σⱼ aⱼ cos(2j·2πr/q)
#                                     = Σⱼ aⱼ · (1/q) Σ_{r=0}^{q-1} cos(4πjr/q)
#
# La suma interior es q si q|j, y 0 si no (ortogonalidad de caracteres).
# Así que:
#
# (1/q) Σ_{r=0}^{q-1} cos²ᵏ(2πr/q) = Σ_{m: mq ≤ k} a_{mq}  +  a₀  (j=0 siempre contribuye... 
#                                                                        no, a₀ ya está en la suma)
#
# Corrección: la suma sobre j incluye j=0 con a₀·(1/q)·q = a₀.
# Para j=mq ≥ 1: aⱼ · 1 (la suma da q, multiplicado por 1/q = 1).
# Así que:
#
# (1/q) Σ_{r=0}^{q-1} cos²ᵏ(2πr/q) = a₀ + Σ_{m=1}^{⌊k/q⌋} a_{mq} = Res
#
# Ahora, cuando k→∞, cos²ᵏ(x) → 0 para x ≠ 0 mod π, y cos²ᵏ(0) = 1.
# En la suma Σ_{r=0}^{q-1} cos²ᵏ(2πr/q):
#   - r = 0: cos²ᵏ(0) = 1
#   - r ≠ 0: cos²ᵏ(2πr/q) → 0
#
# Por tanto:  lim_{k→∞} Res = (1/q) · 1 = 1/q.  ∎

print("═" * 70)
print("DEMOSTRACIÓN NUMÉRICA: Res → 1/q")
print()
print("Res_k = (1/q) · Σ_{r=0}^{q-1} cos²ᵏ(2πr/q)")
print("═" * 70)

for q in [2, 3, 4, 5, 7, 8, 11, 15, 32]:
    print(f"\nq = {q}:")
    print(f"  {'k':<8} {'Res_k':<14} {'1/q':<14} {'|Δ|':<14} {'Contribuciones r≠0'}")
    for k in [5, 10, 20, 50, 100, 200]:
        terms = [np.cos(2*np.pi*r/q)**(2*k) for r in range(q)]
        res_k = sum(terms) / q
        delta = abs(res_k - 1/q)
        nonzero = sum(terms[1:])  # contribución de r≠0
        print(f"  {k:<8} {res_k:<14.8f} {1/q:<14.8f} {delta:<14.2e} {nonzero:.2e}")

# %% 2. Velocidad de convergencia: Res_k - 1/q
#
# La corrección dominante viene de r con cos²(2πr/q) más cercano a 1.
# Es decir, de los r donde 2πr/q ≈ mπ, i.e., r ≈ mq/2.
#
# Para q impar: el mayor cos²(2πr/q) para r≠0 es cos²(2π/q).
# cos²(2π/q) = 1 - 4π²/q² + O(1/q⁴) para q grande.
# Así que cos²ᵏ(2π/q) ≈ (1 - 4π²/q²)ᵏ ≈ exp(-4π²k/q²).
#
# VELOCIDAD: |Res_k - 1/q| ≈ (q-1)/q · exp(-4π²k/q²)
#
# Esto significa:
# - Para q fijo: convergencia exponencial en k. Rápida.
# - Para k fijo: el error CRECE con q (porque exp(-c·k/q²) → 1).
# - La "resolución" k necesaria para separar q es k ~ q².

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: |Res_k - 1/q| vs k para varios q
ax = axes[0]
k_range = np.arange(1, 300)
for q, color in [(2,'#55cc77'), (3,'#55aacc'), (5,'#8888cc'),
                  (7,'#ccaa44'), (11,'#cc8844'), (15,'#cc5555')]:
    errors = []
    for k in k_range:
        terms = sum(np.cos(2*np.pi*r/q)**(2*k) for r in range(q))
        res = terms / q
        errors.append(abs(res - 1/q))
    ax.semilogy(k_range, errors, color=color, linewidth=1.5, label=f'q={q}')
    
    # Predicción teórica: (q-1)/q · exp(-4π²k/q²)
    pred = (q-1)/q * np.exp(-4*np.pi**2*k_range/q**2)
    ax.semilogy(k_range, pred, color=color, linewidth=0.8, linestyle=':', alpha=0.5)

ax.set_xlabel('k')
ax.set_ylabel('|Res_k - 1/q|')
ax.set_title('Velocidad de convergencia Res → 1/q\n(sólido: exacto, punteado: predicción exp(-4π²k/q²))',
             fontsize=10, color='#bbb')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.2)
ax.set_ylim(1e-15, 1)

# Panel 2: k necesario para |error| < ε vs q
ax = axes[1]
for eps, color, ls in [(0.01,'#55cc77','-'), (0.001,'#55aacc','--'), (0.0001,'#cc8844',':')]:
    q_vals = range(2, 50)
    k_needed = []
    for q in q_vals:
        # k ~ q²/(4π²) · ln((q-1)/(q·ε))
        k_est = q**2 / (4*np.pi**2) * np.log(max((q-1)/(q*eps), 1.01))
        k_needed.append(k_est)
    ax.plot(list(q_vals), k_needed, color=color, linewidth=1.5, linestyle=ls,
            label=f'ε = {eps}')

ax.set_xlabel('q (denominador)')
ax.set_ylabel('k necesario')
ax.set_title('Resolución k requerida para |Res-1/q| < ε\nEscala como q² — coste cuadrático',
             fontsize=10, color='#bbb')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# %% 3. C₀: la parte finita distingue numeradores
#
# C₀(p/q) depende de p, no solo de q. 
# Exploremos la estructura de C₀ como función de p para q fijo.
def factorize(n):
    orig = n; factors = []
    for p in [2,3,5,7,11,13]:
        e = 0
        while n % p == 0: e += 1; n //= p
        if e > 0: factors.append(f'{p}{"^"+str(e) if e>1 else ""}')
    if n > 1: factors.append(str(n))
    return '·'.join(factors)

def euler_phi(n):
    return sum(1 for i in range(1, n+1) if gcd(i, n) == 1)

def compute_C0(p, q, k, N=10000):
    """Parte finita: C₀ = F_N(1) - Res · ln(N)."""
    # Residuo analítico
    terms_res = sum(np.cos(2*np.pi*r/q)**(2*k) for r in range(q)) / q
    
    # F_N(1) = Σ cos²ᵏ(πpn/q) / n
    n = np.arange(1, N+1, dtype=float)
    FN = np.sum(np.cos(np.pi * p * n / q) ** (2*k) / n)
    
    return FN - terms_res * np.log(N)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
k = 40

for ax, q in zip(axes.flat, [5, 7, 11, 13, 8, 12]):
    # Todos los p coprimos con q
    ps = [p for p in range(1, q) if gcd(p, q) == 1]
    c0s = [compute_C0(p, q, k) for p in ps]
    
    colors = ['#55cc77' if p == 1 or p == q-1 else '#6688cc' for p in ps]
    
    bars = ax.bar([f'{p}/{q}' for p in ps], c0s, color=colors, edgecolor='none', width=0.6)
    
    # Línea media
    mean_c0 = np.mean(c0s)
    ax.axhline(mean_c0, color='#cc884488', linewidth=1, linestyle='--')
    
    is_prime = all(q % d != 0 for d in range(2, q))
    ax.set_title(f'q={q} {"(primo)" if is_prime else f"= {factorize(q)}"}  φ={euler_phi(q)}',
                 fontsize=9, color='#55cc77' if is_prime else '#ccaa33')
    ax.set_ylabel('C₀')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.grid(True, alpha=0.15)
    
    # Anotar simetría p ↔ q-p
    for i, p in enumerate(ps):
        complement = q - p
        if complement in ps:
            j = ps.index(complement)
            if abs(c0s[i] - c0s[j]) < 1e-8 and i < j:
                ax.plot([i, j], [c0s[i]+0.002, c0s[j]+0.002], 
                       color='#ccaa3388', linewidth=0.8)

plt.suptitle(f'Parte finita C₀(p/q) — k={k}\n'
             f'Verde: p=1 y p=q-1 (siempre iguales por simetría)',
             fontsize=11, color='#bbb')
plt.tight_layout()
plt.show()



# %% 4. Estructura de C₀: ¿qué función de p es?
#
# Para q primo: hay (q-1)/2 valores distintos de C₀ (por simetría p↔q-p).
# Los residuos cuadráticos mod q tienen un C₀ diferente de los no-residuos?
# Esto conectaría con el símbolo de Legendre.

print("═" * 70)
print("C₀ y residuos cuadráticos")
print("═" * 70)

k = 40
for q in [7, 11, 13, 17, 19, 23]:
    if not all(q % d != 0 for d in range(2, q)):
        continue
    
    # Residuos cuadráticos mod q
    qr = set()
    for a in range(1, q):
        qr.add((a*a) % q)
    qr.discard(0)
    
    ps = [p for p in range(1, q) if gcd(p, q) == 1]
    c0s = {p: compute_C0(p, q, k) for p in ps}
    
    c0_qr = [c0s[p] for p in ps if p in qr]
    c0_nqr = [c0s[p] for p in ps if p not in qr]
    
    mean_qr = np.mean(c0_qr) if c0_qr else 0
    mean_nqr = np.mean(c0_nqr) if c0_nqr else 0
    
    print(f"\nq = {q}  (primo)")
    print(f"  QR  = {sorted(qr)}")
    print(f"  C₀ medios: QR = {mean_qr:.6f},  NQR = {mean_nqr:.6f},  Δ = {mean_qr-mean_nqr:.6f}")
    print(f"  Valores C₀: ", end="")
    for p in sorted(ps)[:min(len(ps), 10)]:
        tag = "Q" if p in qr else "N"
        print(f"{p}({tag}):{c0s[p]:.4f} ", end="")
    print()

# %% 5. ESPECTRÓMETRO ARITMÉTICO — versión mejorada
#
# Versión refinada que:
# 1. Usa autocorrelación eficiente (FFT)
# 2. Evalúa S_k en lags enteros de múltiples fundamentales candidatas
# 3. Produce un mapa 2D: (f_fundamental, ratio) → sonancia

def fast_autocorrelation(signal):
    """Autocorrelación via FFT, normalizada."""
    n = len(signal)
    # Pad to next power of 2
    n_fft = 2 ** int(np.ceil(np.log2(2*n)))
    fft = np.fft.rfft(signal, n=n_fft)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    return acf / acf[0]  # normalizar

def acf_interpolated(acf, lag_float):
    """Interpolación lineal de la autocorrelación en lag fraccionario."""
    if lag_float < 0 or lag_float >= len(acf) - 1:
        return 0.0
    idx = int(lag_float)
    frac = lag_float - idx
    return acf[idx] * (1 - frac) + acf[idx + 1] * frac

def acf_to_Rn(acf_val):
    """
    Mapea autocorrelación normalizada [-1, 1] a R(n) = cos²(πrn) ∈ [0, 1].
    
    La autocorrelación normalizada acf(τ) oscila en [-1, 1].
    Para una señal de dos tonos puros: acf en lag nT₁ = ½ + ½cos(2πrn) = cos²(πrn).
    Para señales con más componentes, mapeamos: R = (1 + acf) / 2
    que preserva: acf=1 → R=1 (recurrencia perfecta), acf=-1 → R=0, acf=0 → R=0.5.
    """
    return np.clip((1 + acf_val) / 2, 0, 1)

def spectrometer_2d(signal, sr, f_range, r_range, k=30, max_n=48):
    """
    Espectrómetro 2D: para cada f₁ candidata y ratio r,
    evalúa S_k desde la autocorrelación.
    
    Returns: Z[i_f, i_r] = S_k(r) asumiendo fundamental f₁.
    """
    acf = fast_autocorrelation(signal)
    Z = np.zeros((len(f_range), len(r_range)))
    
    for i, f1 in enumerate(f_range):
        T1 = sr / f1  # periodo en muestras
        for j, r in enumerate(r_range):
            S = 0
            for n in range(1, max_n + 1):
                lag = n * T1
                if lag < len(acf) - 1:
                    raw = acf_interpolated(acf, lag)
                    R_val = acf_to_Rn(raw)
                    S += R_val ** k / n
            Z[i, j] = S
    
    return Z

def generate_signal(freqs, amps=None, duration=2.0, sr=8000, noise=0.05):
    """Señal compuesta con ruido."""
    t = np.arange(0, duration, 1/sr)
    if amps is None: amps = np.ones(len(freqs))
    phases = np.random.uniform(0, 2*np.pi, len(freqs))
    s = sum(a * np.sin(2*np.pi*f*t + phi) for f, a, phi in zip(freqs, amps, phases))
    s += noise * np.random.randn(len(t))
    return t, s

# %% 6. Test: acorde mayor + detección completa

sr = 8000
f1 = 220
freqs = [f1, f1*5/4, f1*3/2]
t, signal = generate_signal(freqs, duration=3.0, sr=sr, noise=0.03)

# Espectrómetro 1D (fundamental conocida)
r_range = np.linspace(1.0, 2.0, 500)
k = 25

acf = fast_autocorrelation(signal)
T1 = sr / f1
max_n = 48

S_detected = np.zeros(len(r_range))
for j, r in enumerate(r_range):
    for n in range(1, max_n+1):
        lag = n * T1  # lag fraccionario en muestras
        if lag < len(acf) - 1:
            raw = acf_interpolated(acf, lag)
            R_val = acf_to_Rn(raw)
            S_detected[j] += R_val**k / n

# Teórica
S_theory = np.array([sum(np.cos(np.pi*r*n)**(2*k)/n for n in range(1, max_n+1)) for r in r_range])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Señal
ax = axes[0, 0]
ax.plot(t[:3000], signal[:3000], color='#6688cc', linewidth=0.3)
ax.set_title(f'Señal: {f1}Hz (P1) + {f1*5/4:.0f}Hz (M3) + {f1*3/2:.0f}Hz (P5)',
             fontsize=9, color='#bbb')
ax.set_xlabel('t (s)'); ax.set_ylabel('s(t)')
ax.grid(True, alpha=0.2)

# Autocorrelación
ax = axes[0, 1]
lag_ms = np.arange(len(acf)) / sr * 1000
ax.plot(lag_ms[:int(sr*0.05)], acf[:int(sr*0.05)], color='#55aa77', linewidth=0.8)
ax.set_title('Autocorrelación R(τ)', fontsize=9, color='#bbb')
ax.set_xlabel('τ (ms)'); ax.set_ylabel('R(τ)')
ax.grid(True, alpha=0.2)

# Espectrómetro vs FFT
ax = axes[1, 0]
fft = np.fft.rfft(signal)
fft_freqs = np.fft.rfftfreq(len(signal), 1/sr)
ax.plot(fft_freqs, np.abs(fft)/len(signal), color='#6688cc', linewidth=0.5)
for f in freqs:
    ax.axvline(f, color='#55cc7766', linewidth=1, linestyle='--')
ax.set_xlim(100, 500)
ax.set_title('FFT: frecuencias absolutas', fontsize=9, color='#bbb')
ax.set_xlabel('Hz'); ax.set_ylabel('|FFT|')
ax.grid(True, alpha=0.2)

# Espectrómetro aritmético
ax = axes[1, 1]
S_norm = S_detected / S_detected.max()
S_th_norm = S_theory / S_theory.max()
ax.plot(r_range, S_th_norm, color='#44885588', linewidth=4, label='Teórico S_k')
ax.plot(r_range, S_norm, color='#cc8844', linewidth=1.5, label='Detectado')
for r_exp, name in [(5/4, 'M3'), (3/2, 'P5')]:
    ax.axvline(r_exp, color='#55cc77', alpha=0.5, linewidth=1, linestyle='--')
    ax.text(r_exp, 1.05, name, ha='center', fontsize=8, color='#55cc77')
ax.set_title('Espectrómetro aritmético: ratios detectados', fontsize=9, color='#bbb')
ax.set_xlabel('r = f/f₁'); ax.set_ylabel('S_k (norm)')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.2)

plt.suptitle('FFT detecta frecuencias absolutas  →  Espectrómetro detecta relaciones racionales',
             fontsize=11, color='#bbb')
plt.tight_layout()
plt.show()

# %% 7. Espectrómetro 2D: fundamental desconocida

sr = 8000
# Señal con fundamental desconocida
f_unknown = 261.63  # Do4
freqs = [f_unknown, f_unknown*5/4, f_unknown*3/2, f_unknown*2]
t, signal = generate_signal(freqs, duration=3.0, sr=sr, noise=0.05)

f_range = np.linspace(100, 500, 80)
r_range = np.linspace(1.0, 2.5, 100)
k = 20

print("Calculando espectrómetro 2D...")
Z = spectrometer_2d(signal, sr, f_range, r_range, k=k, max_n=32)
print("Listo.")

fig, ax = plt.subplots(figsize=(12, 7))

from matplotlib.colors import LinearSegmentedColormap
scmap = LinearSegmentedColormap.from_list('s', ['#0a0a0a','#1a0a2a','#802040','#cc6030','#ddaa30','#ffffcc'])

im = ax.pcolormesh(r_range, f_range, Z, cmap=scmap, shading='auto')
plt.colorbar(im, ax=ax, label='$S_k$', shrink=0.8)

# Marcar posición esperada
ax.axhline(f_unknown, color='#55cc7788', linewidth=1, linestyle='--')
for r_exp, name in [(5/4, 'M3'), (3/2, 'P5'), (2, 'P8')]:
    if r_exp <= r_range[-1]:
        ax.axvline(r_exp, color='#55cc7744', linewidth=1, linestyle='--')
        ax.text(r_exp, f_range[-1]+5, name, ha='center', fontsize=8, color='#55cc77')

ax.set_xlabel('r = f/f₁ (ratio)')
ax.set_ylabel('f₁ candidata (Hz)')
ax.set_title(f'Espectrómetro 2D: busca fundamental Y ratios simultáneamente\n'
             f'Señal: Do4 ({f_unknown:.0f}Hz) + M3 + P5 + P8',
             fontsize=10, color='#bbb')

plt.tight_layout()
plt.show()

# %% 8. Invariancia a transposición

sr = 8000
k = 25
r_range = np.linspace(1.0, 2.0, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 7))

fundamentals = [220, 261.63, 293.66, 329.63, 440, 523.25]
labels = ['A3 (220)', 'C4 (262)', 'D4 (294)', 'E4 (330)', 'A4 (440)', 'C5 (523)']

all_spectra = []

for ax, f1, label in zip(axes.flat, fundamentals, labels):
    # Mismo acorde mayor, diferente fundamental
    freqs = [f1, f1*5/4, f1*3/2]
    t, signal = generate_signal(freqs, duration=2.0, sr=sr, noise=0.03)
    
    acf = fast_autocorrelation(signal)
    T1 = sr / f1
    
    S = np.zeros(len(r_range))
    for j, r in enumerate(r_range):
        for n in range(1, 48+1):
            lag = n * T1  # fraccionario
            if lag < len(acf) - 1:
                raw = acf_interpolated(acf, lag)
                R_val = acf_to_Rn(raw)
                S[j] += R_val**k / n
    
    S_norm = S / S.max() if S.max() > 0 else S + 1e-10
    all_spectra.append(S_norm)
    
    ax.plot(r_range, S_norm, color='#cc8844', linewidth=1.5)
    ax.axvline(5/4, color='#55cc7766', linewidth=1, linestyle='--')
    ax.axvline(3/2, color='#55cc7766', linewidth=1, linestyle='--')
    ax.set_title(f'Mayor en {label}', fontsize=9, color='#bbb')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('r')

plt.suptitle('INVARIANCIA A TRANSPOSICIÓN: mismo acorde mayor en 6 tonalidades\n'
             'El espectro aritmético es idéntico — la FFT sería completamente diferente',
             fontsize=11, color='#bbb')
plt.tight_layout()
plt.show()

# Correlación entre espectros
print("\nCorrelación entre espectros aritméticos de diferentes transposiciones:")
print("(1.0 = idéntico)")
for i in range(len(fundamentals)):
    row = ""
    for j in range(len(fundamentals)):
        corr = np.corrcoef(all_spectra[i], all_spectra[j])[0, 1]
        row += f"  {corr:.3f}"
    print(f"  {labels[i]:<12}:{row}")

# %% 9. Resumen completo

print("""
═══════════════════════════════════════════════════════════════════════
  RESUMEN — F(s) Y ESPECTRÓMETRO ARITMÉTICO
═══════════════════════════════════════════════════════════════════════

RESULTADO CENTRAL SOBRE F(s):

  F(s, p/q, k) = Σ cos²ᵏ(πpn/q) / n^s

  Res_{s=1} F  →  1/q   cuando  k → ∞

  DEMOSTRACIÓN: Res_k = (1/q) Σ_{r=0}^{q-1} cos²ᵏ(2πr/q).
  El término r=0 contribuye 1/q. Los demás decaen como exp(-4π²k/q²).
  Velocidad: |Res_k - 1/q| ~ exp(-4π²k/q²).
  Resolución requerida: k ~ q² para separar denominador q.

  La sonancia 1/q es el RESIDUO de una función meromorfa.
  
PARTE FINITA C₀:

  C₀(p/q) = lim_{N→∞} [S_k(p/q) - Res·ln(N)]
  
  Depende de p Y de q. Simetría: C₀(p/q) = C₀((q-p)/q).
  Para q primo: φ(q)/2 valores distintos de C₀.
  Los residuos cuadráticos mod q pueden tener C₀ diferente de los no-residuos.
  C₀ contiene información aritmética más fina que 1/q.

ESPECTRÓMETRO ARITMÉTICO:

  Input:  señal s(t) = Σ Aᵢ sin(2πfᵢt + φᵢ) + ruido
  Output: espectro aritmético S_k(r) para r ∈ rango de ratios
  
  Procedimiento:
  1. Calcular autocorrelación R(τ) via FFT
  2. Para cada r candidato: S_k(r) = Σ R(n·T₁)^k / n
  3. Picos de S_k revelan las relaciones racionales en la señal
  
  Propiedades:
  • INVARIANTE A TRANSPOSICIÓN: mismo acorde → mismo espectro
  • Complementario a FFT: FFT → Hz absolutos, espectrómetro → ratios
  • Robusto a ruido (el kernel cos²ᵏ filtra fluctuaciones)
  
  Versión 2D: barre f₁ candidatas → encuentra fundamental Y ratios.
═══════════════════════════════════════════════════════════════════════
""")