# %%
"""
Implementation of Sethares' dissonance algorithm.
Reference: https://sethares.engr.wisc.edu/comprog.html
Based on the C++ implementation: https://sethares.engr.wisc.edu/files/disscurve.cpp
"""
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple

# Constants from Sethares' C++ implementation
DSTAR = 0.24  # Point of maximum dissonance
S1 = 0.0207  # Scaling factor 1
S2 = 18.96  # Scaling factor 2
C1 = 5.0  # Parameter for P&L curve
C2 = -5.0  # Parameter for P&L curve
A1 = -3.51  # Exponent 1
A2 = -5.75  # Exponent 2


def dissmeasure(fvec: np.ndarray, amp: np.ndarray, model: str = "min") -> float:
    """
    Calculates the dissonance of a sound with a given spectrum.
    Matches the functionality of the MATLAB `dissmeasure` function described by Sethares.

    Args:
        fvec: Array of frequencies.
        amp: Array of amplitudes.
        model: 'min' (default) or 'product'.
               The C++ reference uses the 'min' model.

    Returns:
        Total sensory dissonance value.
    """
    fvec = np.asarray(fvec)
    amp = np.asarray(amp)

    # Sort by frequency (optional but good practice)
    idx = np.argsort(fvec)
    fvec = fvec[idx]
    amp = amp[idx]

    # Create pairs (i, j) with i < j
    # We use broadcasting to generate all pairs, then filter for i < j
    # This corresponds to the double loop in the algorithm summing over all pairs.

    # f_i: (N, 1), f_j: (1, N)
    f_i = fvec[:, np.newaxis]
    f_j = fvec[np.newaxis, :]

    a_i = amp[:, np.newaxis]
    a_j = amp[np.newaxis, :]

    # We only care about pairs where f_i < f_j
    # In the C++ code, it loops i and j fully for cross-dissonance.
    # For self-dissonance (single spectrum), we sum over unique pairs.
    # The standard definition is sum over i < j.

    # Calculate dissonance for all pairs
    # f_i (N,1) and f_j (1,N) broadcast to (N,N)
    d_matrix = _calculate_pair_dissonance_vectorized(f_i, a_i, f_j, a_j, model)

    # Sum only unique pairs (i < j)
    mask = np.triu(np.ones((len(fvec), len(fvec)), dtype=bool), k=1)
    return np.sum(d_matrix[mask])


def _calculate_pair_dissonance_vectorized(f1, a1, f2, a2, model="min"):
    """
    Helper to calculate dissonance for arrays of pairs.
    f1, f2, a1, a2 must have the same shape.
    Assumes f1 < f2 is not guaranteed, so we calculate fmin/fmax.
    """
    fmin = np.minimum(f1, f2)
    fmax = np.maximum(f1, f2)

    if model == "min":
        a_term = np.minimum(a1, a2)
    elif model == "product":
        a_term = a1 * a2
    else:
        raise ValueError("Unknown model")

    s = DSTAR / (S1 * fmin + S2)
    fdif = fmax - fmin
    arg1 = A1 * s * fdif
    arg2 = A2 * s * fdif

    dnew = a_term * (C1 * np.exp(arg1) + C2 * np.exp(arg2))
    return dnew


@dataclass(frozen=True)
class Partials:
    """
    Represents a partials as a set of partials.
    """

    freq: np.ndarray
    amp: np.ndarray

    # def __post_init__(self):
    #     self.freq = np.asarray(self.freq)
    #     self.amp = np.asarray(self.amp)

    @classmethod
    def from_harmonics(cls, n_harmonics: int, decay_exponent: float = 1.0):
        k = np.arange(1, n_harmonics + 1)
        freq = k.astype(float)
        amp = 1.0 / (k**decay_exponent)
        return cls(freq, amp)

    @classmethod
    def flat(cls, n_harmonics: int):
        k = np.arange(1, n_harmonics + 1)
        freq = k.astype(float)
        amp = np.ones(n_harmonics)
        return cls(freq, amp)


def calculate_dissonance_curve(
    sampling_freqs: np.ndarray,
    sounding_notes: np.ndarray,
    partials: Partials,
    model: str = "min",
) -> np.ndarray:
    """
    Calculates the dissonance curve.

    Args:
        sampling_freqs: Array of frequencies to test (the x-axis).
        sounding_notes: Array of frequencies of the fixed notes (context).
        partials: The partials definition to apply to all notes.
        model: 'min' or 'product'.

    Returns:
        Array of dissonance values.
    """
    sampling_freqs = np.asarray(sampling_freqs)
    sounding_notes = np.asarray(sounding_notes)

    if len(sounding_notes) == 0:
        return np.zeros_like(sampling_freqs)

    # Pre-calculate context partials
    # Shape: (NumSounding, NumPartials)
    context_freqs = sounding_notes[:, np.newaxis] * partials.freq[np.newaxis, :]
    context_amps = (
        np.ones_like(sounding_notes)[:, np.newaxis] * partials.amp[np.newaxis, :]
    )

    context_freqs = context_freqs.flatten()
    context_amps = context_amps.flatten()

    dissonance_values = np.zeros_like(sampling_freqs)

    # We can vectorize this loop partially or fully.
    # Fully vectorizing might be memory intensive if sampling_freqs is large.
    # Let's do a loop for clarity and memory safety, or use the previous vectorized approach if efficient.
    # The user asked for "numpy friendly".

    # Let's use the efficient approach:
    # Total Dissonance = D(Context) + D(Sample) + D(Context, Sample)

    # 1. D(Context) - Constant

    d_context = dissmeasure(context_freqs, context_amps, model)

    # 2. D(Sample) - Varies with f
    # Sample partials: f * partials.freq
    # We need to calculate this for each f in sampling_freqs.
    # D(f) = sum_{i<j} d(f*r_i, f*r_j)
    #      = sum_{i<j} min(a_i, a_j) * (C1*exp(...) + C2*exp(...))
    # Note that s depends on f_min = f * min(r_i, r_j).
    # So D(Sample) is NOT constant, it depends on f.

    # We can compute D(Sample) for all f at once.
    # Pairs of partials in partials:
    t_freq = partials.freq
    t_amp = partials.amp

    # Indices of unique pairs in partials
    idx = np.triu_indices(len(t_freq), k=1)
    r1 = t_freq[idx[0]]
    r2 = t_freq[idx[1]]
    ta1 = t_amp[idx[0]]
    ta2 = t_amp[idx[1]]

    # For each sampling freq f, partials are f*r1, f*r2
    # We can broadcast over sampling_freqs
    # f_sample: (K,)
    # r1, r2: (P_pairs,)
    # f1_grid: (K, P_pairs)
    f1_grid = sampling_freqs[:, np.newaxis] * r1[np.newaxis, :]
    f2_grid = sampling_freqs[:, np.newaxis] * r2[np.newaxis, :]

    # Amplitudes are constant for the partials
    # ta1, ta2: (P_pairs,) -> (1, P_pairs)
    ta1_grid = ta1[np.newaxis, :]
    ta2_grid = ta2[np.newaxis, :]

    # Broadcast amplitudes to match f1_grid shape if needed (though calculate_pair handles broadcasting)
    d_sample = np.sum(
        _calculate_pair_dissonance_vectorized(
            f1_grid, ta1_grid, f2_grid, ta2_grid, model
        ),
        axis=1,
    )

    # 3. D(Context, Sample)
    # Cross dissonance between Context (M partials) and Sample (P partials)
    # Context: context_freqs (M,)
    # Sample: sampling_freqs (K,) x partials.freq (P,)

    # We need pairs between all context partials and all sample partials.
    # c_f: (M,)
    # s_f: (K, P)

    # Expand to (K, M, P)
    # c_f_grid: (1, M, 1)
    # s_f_grid: (K, 1, P)

    c_f_grid = context_freqs[np.newaxis, :, np.newaxis]
    s_f_grid = (
        sampling_freqs[:, np.newaxis, np.newaxis]
        * partials.freq[np.newaxis, np.newaxis, :]
    )

    c_a_grid = context_amps[np.newaxis, :, np.newaxis]
    s_a_grid = (
        np.ones((len(sampling_freqs), 1, 1)) * partials.amp[np.newaxis, np.newaxis, :]
    )

    d_cross = np.sum(
        _calculate_pair_dissonance_vectorized(
            c_f_grid, c_a_grid, s_f_grid, s_a_grid, model
        ),
        axis=(1, 2),
    )
    # print(d_context)
    # print(d_sample.shape)
    # print(d_cross.shape)
    return d_cross

    return d_context + d_sample + d_cross
