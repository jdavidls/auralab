# Sonance

```python
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

```