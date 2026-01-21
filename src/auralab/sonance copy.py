# %%
import torch
import torch.fft


def psd(x, n_fft, dim, window=True):
    """
    Calculate the Power Spectral Density (PSD) of a signal using FFT.

    Args:
        x (torch.Tensor): Input signal.
        n_fft (int): Number of FFT points.
        dim (int): Dimension along which to perform the FFT.

    Returns:
        torch.Tensor: Power Spectral Density of the input signal.
    """
    if window:
        window = torch.hann_window(n_fft, device=x.device)
        x = x * window

    # FFT
    X = torch.fft.rfft(x, n=n_fft, dim=dim)

    # Power Spectral Density
    return X * torch.conj(X)


def acorr(x, n_fft, dim, norm=True, window=True):
    """
    Calculate the Autocorrelation of a signal using FFT.

    Args:
        x (torch.Tensor): Input signal.
        n_fft (int): Number of FFT points.
        dim (int): Dimension along which to perform the FFT.
        norm (bool): Whether to normalize the autocorrelation.
        window (bool): Whether to apply windowing.

    Returns:
        torch.Tensor: Autocorrelation of the input signal.
    """
    # Power Spectral Density
    S = psd(x, n_fft, dim, window)

    # Inverse FFT to get Autocorrelation
    corr = torch.fft.irfft(S, n=n_fft, dim=dim)

    # Normalize
    # corr[:, 0] is the energy (lag 0)
    # We want to normalize so lag 0 is 1.0
    # Add epsilon to avoid division by zero

    if norm:
        energy = corr[:, 0].unsqueeze(1)
        corr = corr / (energy + 1e-8)

    return corr


def sonance_corr_matrix(ratios, f_base=440.0, sr=44100, duration=0.1, device="cpu"):
    """
    Calculate the autocorrelation matrix for the sum of reference and variable sine waves.

    Args:
        ratios (list or torch.Tensor): List of frequency ratios (f2/f1).
        f_base (float): Base frequency for the reference sine wave.
        sr (int): Sampling rate.
        duration (float): Duration of the signals in seconds.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The normalized autocorrelation matrix.
    """
    ratios = torch.as_tensor(ratios, dtype=torch.float32, device=device)

    # Time vector zero centered
    # We want to center the signal in the middle of the window
    # So we need to shift the time vector by half the duration
    n_samples = int(sr * duration)
    t = torch.linspace(
        -duration / 2, duration / 2, n_samples, device=device, dtype=torch.float32
    )

    # Reference sine wave (f_base)
    # Shape: (1, n_samples)
    s1 = torch.sin(2 * torch.pi * f_base * t).unsqueeze(0)

    # Variable sine waves (f_base * ratios)
    # Shape: (n_ratios, n_samples)
    # ratios.unsqueeze(1) makes it (n_ratios, 1)
    s2 = torch.sin(2 * torch.pi * f_base * ratios.unsqueeze(1) * t)

    # Sum of signals
    x = s1 + s2

    # apply windowing
    window = torch.hann_window(n_samples, device=device)
    x = x * window

    # Autocorrelation using FFT
    # R_xx = IFFT(|FFT(x)|^2)

    # FFT
    n_fft = 2 * n_samples
    X = torch.fft.rfft(x, n=n_fft, dim=1)

    # Power Spectral Density
    S = X * torch.conj(X)

    # Inverse FFT to get Autocorrelation
    corr = torch.fft.irfft(S, n=n_fft, dim=1)

    # Normalize
    # corr[:, 0] is the energy (lag 0)
    # We want to normalize so lag 0 is 1.0
    # Add epsilon to avoid division by zero
    energy = corr[:, 0].unsqueeze(1)
    corr = corr / (energy + 1e-8)

    return corr


def measure_sonance_peaks(corr, correction=True):
    """
    Measure sonance from the correlation matrix using peak detection.

    Args:
        corr (torch.Tensor): Normalized autocorrelation matrix.
        correction (bool): Whether to use parabolic interpolation for peak correction.

    Returns:
        torch.Tensor: Sonance values (max peaks).
    """
    device = corr.device
    n_samples = corr.shape[1] // 2  # Approximate valid range from padding

    # Mask main lobe
    mask_len = 50  # Heuristic
    if mask_len < corr.shape[1]:
        corr = corr.clone()
        corr[:, :mask_len] = 0.0

    # Valid range (first half)
    # We used n_samples in previous implementation, let's stick to it.
    # But corr length is 2*n_samples.
    # The valid linear correlation part is [0, n_samples].
    valid_corr = corr[:, :n_samples]

    # Max peak
    values, indices = torch.max(valid_corr, dim=1)

    # Parabolic interpolation for sub-sample accuracy
    if correction:
        # Ensure we can gather neighbors
        n_cols = valid_corr.shape[1]
        can_interpolate = (indices > 0) & (indices < n_cols - 1)

        # Use safe indices for gathering (clamped) to avoid out of bounds
        safe_indices = torch.clamp(indices, min=1, max=n_cols - 2)

        batch_indices = torch.arange(len(values), device=device)

        y1 = valid_corr[batch_indices, safe_indices - 1]
        y2 = valid_corr[batch_indices, safe_indices]
        y3 = valid_corr[batch_indices, safe_indices + 1]

        # Parabolic peak offset
        # d = (y3 - y1) / (2 * (2 * y2 - y1 - y3))
        denom = 2 * (2 * y2 - y1 - y3)
        # Avoid division by zero (flat peak)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        d = (y3 - y1) / denom

        # Interpolated peak value
        peak_val = y2 + d * (y3 - y1) / 4

        # Use interpolated value where possible, otherwise raw max
        kernel_values = torch.where(can_interpolate, peak_val, values)
    else:
        kernel_values = values

    return kernel_values


# %%


def find_peaks(x: torch.Tensor, dim: int = -1, threshold=1e-2) -> torch.Tensor:
    """
    Detecta picos a lo largo de un eje dado en un tensor de PyTorch.
    Un pico es un valor estrictamente mayor que sus vecinos inmediatos en 'dim'.

    x   : tensor N-dimensional
    dim : eje a lo largo del cual buscar picos (por defecto -1)
    threshold : valor mínimo absoluto que debe superar el pico (opcional)
    """
    if x.ndim == 0:
        raise ValueError("Se necesita al menos 1 dimensión")

    dim = dim % x.ndim  # normaliza ejes negativos

    # Si el tamaño en ese eje es < 3, no hay picos posibles
    if x.shape[dim] < 3:
        return torch.zeros_like(x, dtype=torch.bool)

    # Construimos slices para centro, izquierda y derecha
    slc_center = [slice(None)] * x.ndim
    slc_left   = [slice(None)] * x.ndim
    slc_right  = [slice(None)] * x.ndim

    slc_center[dim] = slice(1, -1)
    slc_left[dim]   = slice(0, -2)
    slc_right[dim]  = slice(2, None)

    center = x[tuple(slc_center)]
    left   = x[tuple(slc_left)]
    right  = x[tuple(slc_right)]

    inner_mask = (center > left) & (center > right)

    # Evita considerar regiones planas en cero como picos válidos
    zero_region = (center == 0) & (left == 0) & (right == 0)
    inner_mask &= ~zero_region

    if threshold is not None:
        if not torch.is_tensor(threshold):
            threshold = torch.tensor(float(threshold), dtype=x.dtype, device=x.device)
        else:
            threshold = threshold.to(dtype=x.dtype, device=x.device)

        inner_mask &= center.abs() >= threshold

    # Creamos máscara completa (bordes siempre False)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[tuple(slc_center)] = inner_mask

    return mask

def generate_sonance_kernel(
    ratios,
    f_base=440.0,
    sr=48000,
    duration=1.0,
    correction=False,
    method="peaks",
    decay_weight=0.5,
    device="cpu",
):
    """
    Generate the Sonance kernel for a given list of frequency ratios using PyTorch.
    Wrapper around sonance_corr_matrix and measurement functions.

    Args:
        ratios (list or torch.Tensor): List of frequency ratios (f2/f1).
        f_base (float): Base frequency for the reference sine wave.
        sr (int): Sampling rate.
        duration (float): Duration of the signals in seconds.
        correction (bool): Whether to use parabolic interpolation (only for 'peaks' method).
        method (str): Measurement method: 'peaks' or 'weighted_sum'.
        decay_weight (float): Decay weight for 'weighted_sum' method.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The generated kernel values corresponding to the ratios.
    """
    corr = sonance_corr_matrix(ratios, f_base, sr, duration, device)

    if method == "peaks":
        return measure_sonance_peaks(corr, correction=correction)

    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Data Generation ---
    print("Generating Sonance Data...")
    f_base = 440.0
    sr = 44100
    duration = 0.1
    device = "cpu"  # Use CPU for simplicity in app, or check CUDA if needed

    # Define domain in Cents
    cents = np.linspace(-2400, 2400, 4000)
    ratios = 2 ** (cents / 1200)

    corr = sonance_corr_matrix(
        ratios,
        f_base=f_base,
        sr=sr,
        duration=duration,
        device=device,
    )

    corr = corr[:, :4410]


    

    plt.imshow(corr**2, aspect="auto")#, extent=[-2400, 2400, -2400, 2400])
    plt.colorbar()
    plt.show()

    #%%

    mask = find_peaks(corr, dim=-1)
    

    #%%

    plt.plot((corr**2).sum(dim=-1))

# %%
