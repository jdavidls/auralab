#%%
"""Spectral processing utilities built on top of librosa."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _build_two_octave_kernel(bins_per_octave: int) -> np.ndarray:
	"""Create a complex kernel spanning two octaves and boosting the 5th of the 2nd octave."""

	kernel_width = 2 * bins_per_octave + 1
	kernel = np.zeros(kernel_width, dtype=np.complex128)
	kernel[bins_per_octave + 1] = 1.0 + 0j
	#kernel[bins_per_octave+1 +7*2] = 2.0 + 0j
	return kernel


def _spectral_convolution(cqt_matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""Convolve complex CQT bins along the frequency axis with a complex kernel."""

	return np.apply_along_axis(
		lambda column: np.convolve(column, kernel, mode="same"), axis=0, arr=cqt_matrix
	)


def process_guitar_cqt(
	input_path: str | Path = "data/guitar.mp3",
	output_path: str | Path = "data/guitar.out.mp3",
) -> Path:
	"""Apply spectral convolution on the CQT of an audio file and save the ICQT result."""

	input_path = Path(input_path)
	output_path = Path(output_path)

	y, sr = librosa.load(input_path.as_posix(), sr=44100, mono=True)

	hop_length = 128
	bins_per_octave = 24
	n_bins = bins_per_octave * 10
	fmin = librosa.note_to_hz("C0")

	cqt_matrix = librosa.cqt(
		y,
		sr=sr,
		hop_length=hop_length,
		fmin=fmin,
		n_bins=n_bins,
		bins_per_octave=bins_per_octave,
	)

	kernel = _build_two_octave_kernel(bins_per_octave)
	processed_cqt = _spectral_convolution(cqt_matrix, kernel)

	reconstructed = librosa.icqt(
		processed_cqt,
		sr=sr,
		hop_length=hop_length,
		fmin=fmin,
		bins_per_octave=bins_per_octave,
		length=len(y),
	)

	#reconstructed /= np.max(np.abs(reconstructed)) + 1e-10
	sf.write(output_path.as_posix(), reconstructed, sr)

	return output_path


if __name__ == "__main__":
	process_guitar_cqt()

