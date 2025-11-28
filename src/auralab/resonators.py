#%%
"""Simuladores FDTD y herramientas para estudiar simpatia."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np




@dataclass
class StringFDTDResonator:
	"""Simulador FDTD de cuerda 1D con extremos fijos."""

	length_m: float
	tension_n: float
	linear_density: float
	sample_rate: float = 48_000.0
	grid_points: int = 200
	damping: float = 1e-3

	def __post_init__(self) -> None:
		if self.length_m <= 0.0:
			raise ValueError("length_m must be positive")
		if self.tension_n <= 0.0:
			raise ValueError("tension_n must be positive")
		if self.linear_density <= 0.0:
			raise ValueError("linear_density must be positive")
		if self.sample_rate <= 0.0:
			raise ValueError("sample_rate must be positive")
		if self.grid_points < 3:
			raise ValueError("grid_points must be >= 3")
		if self.damping < 0.0:
			raise ValueError("damping must be non-negative")

		self._wave_speed = np.sqrt(self.tension_n / self.linear_density)
		self._dx = self.length_m / (self.grid_points - 1)
		required_fs = (self._wave_speed / self._dx) * 1.02
		if self.sample_rate <= required_fs:
			self.sample_rate = required_fs  # Asegura dt estable segun Courant
		self._dt = 1.0 / self.sample_rate
		self._lambda_sq = (self._wave_speed * self._dt / self._dx) ** 2

	def _position_to_index(self, position: float) -> int:
		pos = float(np.clip(position, 0.0, 1.0))
		return int(round(pos * (self.grid_points - 1)))

	def simulate_pluck(
		self,
		duration: float,
		*,
		pluck_position: float = 0.2,
		observation_position: float = 0.8,
		amplitude: float = 1.0,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Pluck triangular inicial y devuelve desplazamiento observado."""

		if duration <= 0.0:
			raise ValueError("duration must be positive")
		steps = int(duration * self.sample_rate)
		if steps < 2:
			raise ValueError("duration * sample_rate must be >= 2 samples")

		pluck_idx = self._position_to_index(pluck_position)
		obs_idx = self._position_to_index(observation_position)

		y_prev = np.zeros(self.grid_points)
		y_curr = np.zeros(self.grid_points)

		grid = np.linspace(0.0, 1.0, self.grid_points)
		for idx, pos in enumerate(grid):
			if idx <= pluck_idx:
				y_curr[idx] = amplitude * (pos / (grid[pluck_idx] or 1e-9))
			else:
				y_curr[idx] = amplitude * (
					(1.0 - pos) / (1.0 - grid[pluck_idx] if grid[pluck_idx] < 1.0 else 1e-9)
				)

		y_prev[:] = y_curr
		response = np.zeros(steps)
		loss = max(0.0, 1.0 - self.damping)

		for n in range(steps):
			response[n] = y_curr[obs_idx]
			y_next = np.zeros_like(y_curr)
			lap = self._lambda_sq
			for i in range(1, self.grid_points - 1):
				laplacian = y_curr[i + 1] - 2.0 * y_curr[i] + y_curr[i - 1]
				y_next[i] = 2.0 * y_curr[i] - y_prev[i] + lap * laplacian
				y_next[i] *= loss

			y_next[0] = 0.0
			y_next[-1] = 0.0

			y_prev, y_curr = y_curr, y_next

		time = np.arange(steps) * self._dt
		return time, response

	def simulate_sympathetic_drive(
		self,
		duration: float,
		*,
		drive_frequency: float,
		drive_position: float = 0.2,
		observation_position: float = 0.8,
		amplitude: float = 1.0,
		fade_in: float = 0.02,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Excita la cuerda con una fuerza sinusoidal local y devuelve la respuesta."""

		if duration <= 0.0:
			raise ValueError("duration must be positive")
		if drive_frequency <= 0.0:
			raise ValueError("drive_frequency must be positive")

		steps = int(duration * self.sample_rate)
		if steps < 2:
			raise ValueError("duration * sample_rate must be >= 2 samples")

		drive_idx = self._position_to_index(drive_position)
		obs_idx = self._position_to_index(observation_position)

		y_prev = np.zeros(self.grid_points)
		y_curr = np.zeros(self.grid_points)
		response = np.zeros(steps)
		loss = max(0.0, 1.0 - self.damping)

		t = np.arange(steps) * self._dt
		fade_samples = max(1, int(fade_in * self.sample_rate))
		envelope = np.clip(t / (fade_samples * self._dt), 0.0, 1.0)
		drive = amplitude * envelope * np.sin(2.0 * np.pi * drive_frequency * t)
		force_gain = self._dt**2

		for n in range(steps):
			response[n] = y_curr[obs_idx]
			y_next = np.zeros_like(y_curr)
			lap = self._lambda_sq
			for i in range(1, self.grid_points - 1):
				laplacian = y_curr[i + 1] - 2.0 * y_curr[i] + y_curr[i - 1]
				y_next[i] = 2.0 * y_curr[i] - y_prev[i] + lap * laplacian
				y_next[i] *= loss

			y_next[drive_idx] += drive[n] * force_gain
			y_next[0] = 0.0
			y_next[-1] = 0.0

			y_prev, y_curr = y_curr, y_next

		time = t
		return time, response


@dataclass
class TubeFDTDResonator:
	"""Simulador FDTD 1D de tubo de aire."""

	length_m: float
	sound_speed: float
	sample_rate: float = 48_000.0
	grid_points: int = 200
	damping: float = 2e-3
	boundary_left: Literal["open", "closed"] = "open"
	boundary_right: Literal["open", "closed"] = "closed"

	def __post_init__(self) -> None:
		if self.length_m <= 0.0:
			raise ValueError("length_m must be positive")
		if self.sound_speed <= 0.0:
			raise ValueError("sound_speed must be positive")
		if self.sample_rate <= 0.0:
			raise ValueError("sample_rate must be positive")
		if self.grid_points < 3:
			raise ValueError("grid_points must be >= 3")
		if self.damping < 0.0:
			raise ValueError("damping must be non-negative")
		if self.boundary_left not in {"open", "closed"}:
			raise ValueError("boundary_left must be 'open' or 'closed'")
		if self.boundary_right not in {"open", "closed"}:
			raise ValueError("boundary_right must be 'open' or 'closed'")

		self._dx = self.length_m / (self.grid_points - 1)
		required_fs = (self.sound_speed / self._dx) * 1.02
		if self.sample_rate <= required_fs:
			self.sample_rate = required_fs  # Corrige dt para cumplir Courant
		self._dt = 1.0 / self.sample_rate
		self._lambda_sq = (self.sound_speed * self._dt / self._dx) ** 2

	def _position_to_index(self, position: float) -> int:
		pos = float(np.clip(position, 0.0, 1.0))
		return int(round(pos * (self.grid_points - 1)))

	def _apply_boundary(self, array: np.ndarray) -> None:
		if self.boundary_left == "open":
			array[0] = 0.0
		else:
			array[0] = array[1]
		if self.boundary_right == "open":
			array[-1] = 0.0
		else:
			array[-1] = array[-2]

	def simulate_impulse(
		self,
		duration: float,
		*,
		drive_position: float = 0.1,
		observation_position: float = 0.9,
		amplitude: float = 1.0,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Inyecta un impulso de presion y devuelve la respuesta local."""

		if duration <= 0.0:
			raise ValueError("duration must be positive")
		steps = int(duration * self.sample_rate)
		if steps < 2:
			raise ValueError("duration * sample_rate must be >= 2 samples")

		drive_idx = self._position_to_index(drive_position)
		obs_idx = self._position_to_index(observation_position)

		p_prev = np.zeros(self.grid_points)
		p_curr = np.zeros(self.grid_points)
		response = np.zeros(steps)
		loss = max(0.0, 1.0 - self.damping)

		for n in range(steps):
			if n == 0:
				p_curr[drive_idx] += amplitude
			response[n] = p_curr[obs_idx]
			p_next = np.zeros_like(p_curr)
			for i in range(1, self.grid_points - 1):
				laplacian = p_curr[i + 1] - 2.0 * p_curr[i] + p_curr[i - 1]
				p_next[i] = 2.0 * p_curr[i] - p_prev[i] + self._lambda_sq * laplacian
				p_next[i] *= loss

			self._apply_boundary(p_next)
			p_prev, p_curr = p_curr, p_next

		time = np.arange(steps) * self._dt
		return time, response

	def simulate_sympathetic_drive(
		self,
		duration: float,
		*,
		drive_frequency: float,
		drive_position: float = 0.1,
		observation_position: float = 0.9,
		amplitude: float = 1.0,
		fade_in: float = 0.02,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Excita el tubo con una presion sinusoidal local y mide la respuesta."""

		if duration <= 0.0:
			raise ValueError("duration must be positive")
		if drive_frequency <= 0.0:
			raise ValueError("drive_frequency must be positive")

		steps = int(duration * self.sample_rate)
		if steps < 2:
			raise ValueError("duration * sample_rate must be >= 2 samples")

		drive_idx = self._position_to_index(drive_position)
		obs_idx = self._position_to_index(observation_position)

		p_prev = np.zeros(self.grid_points)
		p_curr = np.zeros(self.grid_points)
		response = np.zeros(steps)
		loss = max(0.0, 1.0 - self.damping)

		t = np.arange(steps) * self._dt
		fade_samples = max(1, int(fade_in * self.sample_rate))
		envelope = np.clip(np.arange(steps) / fade_samples, 0.0, 1.0)
		drive = amplitude * envelope * np.sin(2.0 * np.pi * drive_frequency * t)
		force_gain = self._dt**2

		for n in range(steps):
			response[n] = p_curr[obs_idx]
			p_next = np.zeros_like(p_curr)
			for i in range(1, self.grid_points - 1):
				laplacian = p_curr[i + 1] - 2.0 * p_curr[i] + p_curr[i - 1]
				p_next[i] = 2.0 * p_curr[i] - p_prev[i] + self._lambda_sq * laplacian
				p_next[i] *= loss

			p_next[drive_idx] += drive[n] * force_gain
			self._apply_boundary(p_next)
			p_prev, p_curr = p_curr, p_next

		time = t
		return time, response




def sweep_sympathetic_response(
	resonator: StringFDTDResonator | TubeFDTDResonator,
	drive_frequencies: np.ndarray | list[float],
	*,
	duration: float,
	amplitude: float = 1.0,
	drive_position: float | None = None,
	observation_position: float | None = None,
	fade_in: float = 0.02,
	steady_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
	"""Evalua respuesta RMS ante excitaciones sinusoidales (simpatia)."""

	if not hasattr(resonator, "simulate_sympathetic_drive"):
		raise TypeError("resonator must expose simulate_sympathetic_drive")
	if duration <= 0.0:
		raise ValueError("duration must be positive")
	if steady_fraction <= 0.0 or steady_fraction >= 1.0:
		raise ValueError("steady_fraction must be in (0, 1)")

	frequencies = np.asarray(drive_frequencies, dtype=float)
	if np.any(frequencies <= 0.0):
		raise ValueError("drive_frequencies must be positive")

	if drive_position is None:
		drive_position = 0.2 if isinstance(resonator, StringFDTDResonator) else 0.1
	if observation_position is None:
		observation_position = 0.8 if isinstance(resonator, StringFDTDResonator) else 0.9

	responses = np.empty_like(frequencies)

	for idx, freq in enumerate(frequencies):
		time, signal = resonator.simulate_sympathetic_drive(
			duration,
			drive_frequency=freq,
			drive_position=drive_position,
			observation_position=observation_position,
			amplitude=amplitude,
			fade_in=fade_in,
		)
		start_idx = int(steady_fraction * signal.size)
		steady = signal[start_idx:]
		responses[idx] = np.sqrt(np.mean(steady**2)) if steady.size else 0.0

	return frequencies, responses


def plot_sympathetic_sweep(
	resonator: StringFDTDResonator | TubeFDTDResonator,
	drive_frequencies: np.ndarray | list[float],
	*,
	duration: float,
	amplitude: float = 1.0,
	drive_position: float | None = None,
	observation_position: float | None = None,
	fade_in: float = 0.02,
	steady_fraction: float = 0.5,
):
	"""Grafica la curva de simpatia (RMS vs frecuencia de excitacion)."""

	import matplotlib.pyplot as plt

	freqs, rms = sweep_sympathetic_response(
		resonator,
		drive_frequencies,
		duration=duration,
		amplitude=amplitude,
		drive_position=drive_position,
		observation_position=observation_position,
		fade_in=fade_in,
		steady_fraction=steady_fraction,
	)

	fig, ax = plt.subplots(figsize=(7, 3))
	ax.plot(freqs, rms, marker="o")
	ax.set_xlabel("frecuencia de excitacion [Hz]")
	ax.set_ylabel("RMS respuesta simpatica")
	ax.set_title("Barrido de simpatia")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	return fig


def save_response_wav(
	path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
	signal: np.ndarray,
	sample_rate: float,
	*,
	normalize: bool = True,
) -> None:
	"""Guarda un arreglo 1D en formato WAV mono 16 bits."""

	import wave
	from pathlib import Path

	data = np.asarray(signal, dtype=np.float64)
	if data.ndim != 1:
		raise ValueError("signal must be 1-D")
	if sample_rate <= 0.0:
		raise ValueError("sample_rate must be positive")

	if normalize:
		peak = np.max(np.abs(data))
		if peak > 0.0:
			data = data / peak

	int_data = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
	path_obj = Path(path)
	path_obj.parent.mkdir(parents=True, exist_ok=True)
	with wave.open(str(path_obj), "wb") as wav_file:
		wav_file.setnchannels(1)
		wav_file.setsampwidth(2)
		wav_file.setframerate(int(round(sample_rate)))
		wav_file.writeframes(int_data.tobytes())


__all__ = [
	"StringFDTDResonator",
	"TubeFDTDResonator",
	"sweep_sympathetic_response",
	"plot_sympathetic_sweep",
	"save_response_wav",
]


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	string = StringFDTDResonator(
		length_m=0.35,
		tension_n=440.0,
		linear_density=0.01,
		sample_rate=48_000.0,
		grid_points=401,
		damping=1e-4,
	)
	t_string, disp = string.simulate_pluck(
		duration=1.0,
		pluck_position=0.2,
		observation_position=0.8,
		amplitude=0.5,
	)
	fig_string, ax_string = plt.subplots(figsize=(7, 3))
	ax_string.plot(t_string * 1e3, disp)
	ax_string.set_xlabel("tiempo [ms]")
	ax_string.set_ylabel("desplazamiento [u.a.]")
	ax_string.set_title("Respuesta FDTD de cuerda pulsada")
	ax_string.grid(True, alpha=0.3)

	save_response_wav("string_pluck.wav", disp, string.sample_rate)

	tube = TubeFDTDResonator(
		length_m=0.85,
		sound_speed=343.0,
		sample_rate=48_000.0,
		grid_points=241,
		damping=2e-3,
		boundary_left="open",
		boundary_right="closed",
	)
	t_tube, pressure = tube.simulate_impulse(
		duration=0.5,
		drive_position=0.1,
		observation_position=0.9,
		amplitude=1.0,
	)
	fig_tube, ax_tube = plt.subplots(figsize=(7, 3))
	ax_tube.plot(t_tube * 1e3, pressure)
	ax_tube.set_xlabel("tiempo [ms]")
	ax_tube.set_ylabel("presion relativa [u.a.]")
	ax_tube.set_title("Respuesta FDTD de tubo (abierto-cerrado)")
	ax_tube.grid(True, alpha=0.3)
	save_response_wav("tube_impulse.wav", pressure, tube.sample_rate)
    #%%
	string_freqs = np.linspace(50.0, 600.0, 40)
	fig_string_sym = plot_sympathetic_sweep(
		string,
		string_freqs,
		duration=0.4,
		amplitude=0.2,
	)

	tube_freqs = np.linspace(50.0, 600.0, 40)
	fig_tube_sym = plot_sympathetic_sweep(
		tube,
		tube_freqs,
		duration=0.4,
		amplitude=0.2,
	)

	plt.show()

