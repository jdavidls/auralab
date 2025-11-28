
import torch
import numpy as np
from nnAudio import Spectrogram
from typing import Optional, Union

class TorchCQT:
    """
    A wrapper around nnAudio.Spectrogram.CQT to provide a convenient interface
    similar to librosa, but with PyTorch tensors.
    """
    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        fmin: float = 32.7,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        device: str = 'cpu'
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.device = device
        
        self.cqt_layer = Spectrogram.CQT(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            output_format="Complex",
            verbose=False
        ).to(device)

    def __call__(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute the CQT of the input audio.
        
        Args:
            audio: Input audio waveform. Can be numpy array or torch tensor.
                   Shape should be (batch, samples) or (samples,).
        
        Returns:
            torch.Tensor: Complex CQT spectrogram.
                          Shape: (batch, n_bins, time_frames, 2) where last dim is (real, imag)
                          or (n_bins, time_frames, 2) if input was 1D.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            is_1d = True
        else:
            is_1d = False
            
        audio = audio.to(self.device)
        
        # nnAudio CQT returns (batch, n_bins, time_frames, 2) for Complex output
        # The 2 channels are Real and Imaginary parts
        cqt_complex = self.cqt_layer(audio)
        
        if is_1d:
            cqt_complex = cqt_complex.squeeze(0)
            
        return cqt_complex

def cqt(
    y: Union[np.ndarray, torch.Tensor],
    sr: int = 44100,
    hop_length: int = 512,
    fmin: float = 32.7,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Functional interface for computing CQT using PyTorch.
    """
    processor = TorchCQT(
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        device=device
    )
    return processor(y)
