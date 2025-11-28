# %%
"""Spectral processing utilities built on top of librosa."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

def plot_cqt(data: np.ndarray, sr: int, hop_length: int, bpo: int, audio_path: Path):
	import librosa.display
	import matplotlib.pyplot as plt

	C_db = librosa.amplitude_to_db(np.abs(data), ref=np.max)

	fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

	img = librosa.display.specshow(
		C_db,
		sr=sr,
		hop_length=hop_length,
		x_axis="time",
		y_axis="cqt_note",
		bins_per_octave=bpo,
		ax=ax,
	)

	ax.set(title=f"CQT of {audio_path.name}")
	fig.colorbar(img, ax=ax, label="dB")
	plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    audio_path = project_root / "data" / "guitar.mp3"
    output_path = project_root / "data" / "guitar.out.mp3"
    y, sr = librosa.load(audio_path, sr=48000, mono=True)
    hop_length = 128
    bpo = 48
    bph = 4 # bins per halfstep (semitone)
    n_bins = bpo * 9

    kernel_width = 2 * bpo + 1
    kernel = np.zeros(kernel_width, dtype=np.complex128)
    kernel[bpo] = 0.5 * np.e** (1j * 0)  # center identity
    kernel[bpo + 7 * bph] = np.e** (1j * np.pi / 2)  # perfect fifth above
    #kernel[-1] = np.e** (1j * 0)  # perfect octave above

    C = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bpo)

    D = np.apply_along_axis(
        lambda column: np.convolve(column, kernel, mode="same"),
        axis=0,
        arr=C,
    )

    plot_cqt(C, sr, hop_length, bpo, audio_path)
    plot_cqt(D, sr, hop_length, bpo, audio_path)


    z = librosa.icqt(D, sr=sr, hop_length=hop_length, bins_per_octave=bpo)

    sf.write(output_path.as_posix(), z, sr)

