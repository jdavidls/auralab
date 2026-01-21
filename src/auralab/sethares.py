# %%
"""
Python translation of http://sethares.engr.wisc.edu/comprog.html
"""
import numpy as np


def amp_to_loudness(amp):
    """
    Converts amplitude to loudness based on the reference implementation.
    """
    # Avoid log(0)
    amp = np.asarray(amp)
    amp = np.where(amp == 0, 1e-12, amp)
    dB = 20 * np.log10(amp)
    loudness = 0.0625 * (2 ** (dB / 10))  # 0.0625 is 1/16
    return loudness


def sethares(
    fvec,
    amp,
    model="min",
    D=0.24,
    S=(0.0207, 18.96),
    C=(5, -5),
    A=(-3.51, -5.75),
    use_loudness=False,
):
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.
    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.

    Parameters:
    fvec: list or array of frequencies
    amp: list or array of amplitudes
    model: 'min' or 'product'
    D: float, point of maximum dissonance (default 0.24)
    S: tuple of floats (S1, S2), used to stretch dissonance curve (default (0.0207, 18.96))
    C: tuple of floats (C1, C2), coefficients for the exponential terms (default (5, -5))
    A: tuple of floats (A1, A2), exponents for the exponential terms (default (-3.51, -5.75))
    use_loudness: bool, if True, converts amplitudes to loudness before calculation (default False)
    """
    # Sort by frequency
    sort_idx = np.argsort(fvec)
    am_sorted = np.asarray(amp)[sort_idx]
    fr_sorted = np.asarray(fvec)[sort_idx]

    if use_loudness:
        am_sorted = amp_to_loudness(am_sorted)

    # Unpack constants
    S1, S2 = S
    C1, C2 = C
    A1, A2 = A

    # Generate all combinations of frequency components
    idx = np.transpose(np.triu_indices(len(fr_sorted), 1))
    fr_pairs = fr_sorted[idx]
    am_pairs = am_sorted[idx]

    Fmin = fr_pairs[:, 0]
    s = D / (S1 * Fmin + S2)  # Renamed local S to s to avoid conflict with parameter S
    Fdif = fr_pairs[:, 1] - fr_pairs[:, 0]

    if model == "min":
        a = np.amin(am_pairs, axis=1)
    elif model == "product":
        a = np.prod(am_pairs, axis=1)  # Older model
    else:
        raise ValueError('model should be "min" or "product"')

    SFdif = s * Fdif
    d_val = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))

    return d_val


if __name__ == "__main__":
    from numpy import array, linspace, empty, concatenate
    import matplotlib.pyplot as plt
    from librosa import cqt_frequencies, note_to_hz

    # freqs = cqt_frequencies(88, fmin=note_to_hz("A0"))

    """
    Reproduce Sethares Figure 3
    http://sethares.engr.wisc.edu/consemi.html#anchor15619672
    """
    freq = 500 * array([1, 2, 3, 4, 5, 6])
    amp = 0.88 ** array([0, 1, 2, 3, 4, 5])
    r_low = 1
    alpharange = 2.3
    method = "product"

    #    # Davide Verotta Figure 4 example
    #    freq = 261.63 * array([1, 2, 3, 4, 5, 6])
    #    amp = 1 / array([1, 2, 3, 4, 5, 6])
    #    r_low = 1
    #    alpharange = 2.0
    #    method = 'product'

    n = 3000
    diss = empty(n)
    a = concatenate((amp, amp))
    for i, alpha in enumerate(linspace(r_low, alpharange, n)):
        f = concatenate((freq, alpha * freq))
        d = sethares(f, a, method)
        diss[i] = d

    plt.figure(figsize=(7, 3))
    plt.plot(linspace(r_low, alpharange, len(diss)), diss)
    plt.xscale("log")
    plt.xlim(r_low, alpharange)

    plt.xlabel("frequency ratio")
    plt.ylabel("sensory dissonance")

    intervals = [(1, 1), (6, 5), (5, 4), (4, 3), (3, 2), (5, 3), (2, 1)]

    for n, d in intervals:
        plt.axvline(n / d, color="silver")

    plt.yticks([])
    plt.minorticks_off()
    plt.xticks(
        [n / d for n, d in intervals], ["{}/{}".format(n, d) for n, d in intervals]
    )
    plt.tight_layout()
    plt.show()
