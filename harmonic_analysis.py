"""
harmonic_analysis.py
====================
FFT and STFT based harmonic analysis for power systems.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import get_window, stft


class HarmonicAnalyzer:
    """Harmonic analysis using FFT for static spectra and STFT for tracking."""

    def __init__(self, fs: float = 10_000, f0: float = 50.0, window: str = "hann"):
        self.fs = fs
        self.f0 = f0
        self.window = window

    def compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the one-sided FFT magnitude spectrum."""
        n_samples = len(signal)
        win = get_window(self.window, n_samples)
        coherent_gain = np.sum(win) / n_samples
        spectrum = np.fft.rfft(signal * win) / (n_samples * coherent_gain)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)
        magnitude = np.abs(spectrum)
        magnitude[1:-1] *= 2
        print(freqs)
        return freqs, magnitude

    def compute_fft_complex(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return complex FFT coefficients for phase extraction."""
        n_samples = len(signal)
        win = get_window(self.window, n_samples)
        coherent_gain = np.sum(win) / n_samples
        spectrum = np.fft.rfft(signal * win) / (n_samples * coherent_gain)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)
        return freqs, spectrum

    def extract_harmonics(
        self,
        freqs: np.ndarray,
        magnitude: np.ndarray,
        n_harmonics: int = 10,
        search_bandwidth: float = 3.0,
    ) -> Dict[int, Tuple[float, float]]:
        """Extract harmonic peaks around integer multiples of the fundamental."""
        harmonics = {}
        for order in range(1, n_harmonics + 1):
            target = order * self.f0
            if target > self.fs / 2:
                break
            mask = (freqs >= target - search_bandwidth) & (freqs <= target + search_bandwidth)
            if not np.any(mask):
                continue
            idx = np.argmax(magnitude[mask])
            harmonics[order] = (float(freqs[mask][idx]), float(magnitude[mask][idx]))
        return harmonics

    def extract_harmonic_phases(
        self,
        signal: np.ndarray,
        n_harmonics: int = 10,
        search_bandwidth: float = 3.0,
    ) -> Dict[int, Tuple[float, float, float]]:
        """Return harmonic frequency, magnitude, and phase."""
        freqs, spectrum = self.compute_fft_complex(signal)
        result = {}
        for order in range(1, n_harmonics + 1):
            target = order * self.f0
            if target > self.fs / 2:
                break
            mask = (freqs >= target - search_bandwidth) & (freqs <= target + search_bandwidth)
            if not np.any(mask):
                continue
            idx = np.argmax(np.abs(spectrum[mask]))
            coeff = spectrum[mask][idx]
            result[order] = (
                float(freqs[mask][idx]),
                float(np.abs(coeff)) * 2.0,
                float(np.angle(coeff)),
            )
        return result

    def total_harmonic_distortion(self, harmonics: Dict[int, Tuple[float, float]]) -> float:
        """IEEE 519 THD referenced to the fundamental magnitude."""
        if 1 not in harmonics or harmonics[1][1] == 0:
            return float("nan")
        fundamental = harmonics[1][1]
        harmonic_power = sum(mag ** 2 for order, (_, mag) in harmonics.items() if order != 1)
        return float(np.sqrt(harmonic_power) / fundamental)

    def total_demand_distortion(
        self,
        harmonics: Dict[int, Tuple[float, float]],
        i_load_peak: float,
    ) -> float:
        """TDD referenced to the peak load current."""
        harmonic_power = sum(mag ** 2 for order, (_, mag) in harmonics.items() if order != 1)
        return float(np.sqrt(harmonic_power) / i_load_peak)

    def individual_harmonic_distortion(
        self,
        harmonics: Dict[int, Tuple[float, float]],
    ) -> Dict[int, float]:
        """Return IHD values for each harmonic order."""
        fundamental = harmonics.get(1, (0.0, 1.0))[1]
        return {
            order: mag / fundamental
            for order, (_, mag) in harmonics.items()
            if order != 1
        }

    def stft_spectrogram(
        self,
        signal: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an STFT power spectrogram."""
        if nperseg is None:
            nperseg = max(128, int(2 * self.fs / self.f0))
        if noverlap is None:
            noverlap = int(0.75 * nperseg)

        freqs, times, zxx = stft(
            signal,
            fs=self.fs,
            window=self.window,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
        )
        return times, freqs, np.abs(zxx) ** 2

    def short_time_harmonic_rms(
        self,
        signal: np.ndarray,
        window_len: Optional[int] = None,
        hop: Optional[int] = None,
        n_harmonics: int = 7,
    ) -> Dict[int, np.ndarray]:
        """Track harmonic amplitudes over time using overlapping FFT frames."""
        if window_len is None:
            window_len = int(self.fs / self.f0)
        if hop is None:
            hop = window_len // 2

        n_frames = (len(signal) - window_len) // hop + 1
        tracks = {order: [] for order in range(1, n_harmonics + 1)}

        for idx in range(n_frames):
            segment = signal[idx * hop : idx * hop + window_len]
            freqs, mag = self.compute_fft(segment)
            harmonics = self.extract_harmonics(freqs, mag, n_harmonics=n_harmonics)
            for order in range(1, n_harmonics + 1):
                tracks[order].append(harmonics.get(order, (0.0, 0.0))[1])

        return {order: np.asarray(values) for order, values in tracks.items()}

    def power_spectral_density(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Welch PSD estimate."""
        from scipy.signal import welch

        return welch(signal, fs=self.fs, nperseg=min(1024, len(signal) // 4))

    def dominant_frequency(self, signal: np.ndarray) -> float:
        """Return the strongest frequency component."""
        freqs, mag = self.compute_fft(signal)
        return float(freqs[np.argmax(mag)])
