#%%

import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Parámetros
# -----------------------
fs = 11025
T_obs = 0.5
t = np.arange(0, T_obs, 1/fs)
f1 = 440.0
ratios = np.linspace(1.0, 2.0, 120)

# -----------------------
# Autocorrelación FFT
# -----------------------
def autocorr_fft(x):
    x = x - np.mean(x)
    n = len(x)
    X = np.fft.rfft(x, n=2*n)
    S = X * np.conj(X)
    R = np.fft.irfft(S)
    R = R[:n]
    return R / np.max(R)

Tr, entropy, roughness, peaks_n = [], [], [], []

# -----------------------
# Loop principal
# -----------------------
for r in ratios:
    f2 = f1 * r
    x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

    R = autocorr_fft(x)

    # picos simples sin scipy
    peak_idx = np.where((R[1:-1] > R[:-2]) & (R[1:-1] > R[2:]) & (R[1:-1] > 0.2))[0] + 1

    if len(peak_idx) > 1:
        Tr.append(peak_idx[1] / fs)
    else:
        Tr.append(np.nan)

    p = R[R > 0]
    p = p / np.sum(p)
    entropy.append(-np.sum(p * np.log(p)))

    d2 = np.diff(R, n=2)
    roughness.append(np.sum(np.abs(d2)))

    peaks_n.append(len(peak_idx))

# -----------------------
# Plots
# -----------------------
plt.figure()
plt.plot(ratios, Tr)
plt.xlabel("f2 / f1")
plt.ylabel("Tiempo de recurrencia (s)")
plt.title("Tiempo de recurrencia")
plt.show()

plt.figure()
plt.plot(ratios, entropy)
plt.xlabel("f2 / f1")
plt.ylabel("Entropía")
plt.title("Entropía temporal")
plt.show()

plt.figure()
plt.plot(ratios, roughness)
plt.xlabel("f2 / f1")
plt.ylabel("Rugosidad")
plt.title("Rugosidad temporal")
plt.show()

plt.figure()
plt.plot(ratios, peaks_n)
plt.xlabel("f2 / f1")
plt.ylabel("Número de picos")
plt.title("Densidad de recurrencias")
plt.show()
