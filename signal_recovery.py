"""
signal_recovery.py
===================
Signal recovery for heavily distorted or corrupted power system waveforms.

The main reconstruction path uses STFT analysis and synthesis.
"""

from typing import Dict, Optional

import numpy as np
from scipy.signal import istft, savgol_filter, stft


class SignalRecovery:
    """Recover clean power system signals from corrupted measurements."""

    def __init__(self, fs: float = 10_000, f0: float = 50.0):
        self.fs = fs
        self.f0 = f0

    def recover(self, corrupted: np.ndarray, method: str = "stft", **kwargs) -> np.ndarray:
        methods = {
            "stft": self._recover_stft,
            "harmonic_fit": self._recover_harmonic_fit,
            "savgol": self._recover_savgol,
            "omp": self._recover_omp,
            "iterative_stft": self._recover_iterative_stft,
        }
        if method not in methods:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(methods)}")
        return methods[method](corrupted, **kwargs)

    def _recover_stft(
        self,
        corrupted: np.ndarray,
        nperseg: int = 256,
        noverlap: int = 192,
        noise_quantile: float = 0.35,
        attenuation: float = 0.10,
    ) -> np.ndarray:
        """Recover a waveform through STFT shrinkage and inverse STFT."""
        freqs, times, zxx = stft(
            corrupted,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
        )
        del times

        magnitude = np.abs(zxx)
        phase = np.angle(zxx)
        noise_floor = np.quantile(magnitude, noise_quantile, axis=1, keepdims=True)
        cleaned_mag = np.maximum(magnitude - 1.1 * noise_floor, 0.0)
        cleaned_mag[magnitude < 1.7 * noise_floor] *= attenuation

        harmonic_mask = np.zeros_like(freqs, dtype=bool)
        for order in range(1, 16):
            harmonic_mask |= np.abs(freqs - order * self.f0) <= 8.0
        broadband_mask = freqs <= min(20 * self.f0, 0.45 * self.fs)
        keep_mask = harmonic_mask | broadband_mask
        cleaned_mag[~keep_mask, :] *= attenuation

        _, recovered = istft(
            cleaned_mag * np.exp(1j * phase),
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=None,
        )
        return recovered[: len(corrupted)]

    def _recover_harmonic_fit(
        self,
        corrupted: np.ndarray,
        n_harmonics: int = 10,
        refine_frequency: bool = True,
    ) -> np.ndarray:
        """Fit a harmonic model using least squares."""
        n_samples = len(corrupted)
        time = np.arange(n_samples) / self.fs
        fundamental = self._estimate_f0(corrupted) if refine_frequency else self.f0

        columns = [np.ones(n_samples)]
        for order in range(1, n_harmonics + 1):
            columns.append(np.cos(2 * np.pi * order * fundamental * time))
            columns.append(np.sin(2 * np.pi * order * fundamental * time))

        design = np.column_stack(columns)
        coeffs, _, _, _ = np.linalg.lstsq(design, corrupted, rcond=None)
        return design @ coeffs

    def _estimate_f0(self, signal: np.ndarray, search_range: float = 5.0) -> float:
        n_samples = len(signal)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)
        magnitude = np.abs(np.fft.rfft(signal))
        mask = (freqs >= self.f0 - search_range) & (freqs <= self.f0 + search_range)
        if not np.any(mask):
            return self.f0
        return float(freqs[mask][np.argmax(magnitude[mask])])

    def _recover_savgol(
        self,
        corrupted: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: int = 3,
    ) -> np.ndarray:
        """Savitzky-Golay smoothing baseline."""
        if window_length is None:
            window_length = int(self.fs / self.f0 / 4)
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(window_length, polyorder + 2)
        return savgol_filter(corrupted, window_length=window_length, polyorder=polyorder)

    def _recover_omp(
        self,
        corrupted: np.ndarray,
        n_nonzero: int = 30,
        n_harmonics: int = 50,
    ) -> np.ndarray:
        """Sparse sinusoidal reconstruction via OMP."""
        n_samples = len(corrupted)
        time = np.arange(n_samples) / self.fs

        freqs = [order * self.f0 for order in range(1, n_harmonics + 1)]
        freqs.extend(np.linspace(self.f0 * 0.5, n_harmonics * self.f0, 200))
        freqs = sorted(set(freqs))

        atoms = []
        for freq in freqs:
            atoms.append(np.cos(2 * np.pi * freq * time))
            atoms.append(np.sin(2 * np.pi * freq * time))

        dictionary = np.column_stack(atoms)
        dictionary /= np.linalg.norm(dictionary, axis=0) + 1e-12

        residual = corrupted.copy()
        selected = []
        selected_cols = []
        for _ in range(n_nonzero):
            projections = dictionary.T @ residual
            idx = int(np.argmax(np.abs(projections)))
            selected.append(idx)
            selected_cols.append(dictionary[:, idx])
            design = np.column_stack(selected_cols)
            coeffs, _, _, _ = np.linalg.lstsq(design, corrupted, rcond=None)
            residual = corrupted - design @ coeffs

        return np.column_stack(selected_cols) @ coeffs

    def _recover_iterative_stft(
        self,
        corrupted: np.ndarray,
        n_iter: int = 20,
        attenuation: float = 0.10,
        nperseg: int = 256,
        noverlap: int = 192,
    ) -> np.ndarray:
        """Repeated STFT shrinkage with residual feedback."""
        recovered = corrupted.copy()
        for _ in range(n_iter):
            recovered = recovered + 0.5 * (corrupted - recovered)
            recovered = self._recover_stft(
                recovered,
                nperseg=nperseg,
                noverlap=noverlap,
                attenuation=attenuation,
            )
        return recovered

    def snr(self, reference: np.ndarray, recovered: np.ndarray) -> float:
        """Recovered-signal SNR in dB."""
        noise = reference - recovered[: len(reference)]
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-30:
            return float("inf")
        return float(10 * np.log10(signal_power / noise_power))

    def rmse(self, reference: np.ndarray, recovered: np.ndarray) -> float:
        """Root mean squared error."""
        diff = reference - recovered[: len(reference)]
        return float(np.sqrt(np.mean(diff ** 2)))

    def pearson_r(self, reference: np.ndarray, recovered: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        n_samples = min(len(reference), len(recovered))
        return float(np.corrcoef(reference[:n_samples], recovered[:n_samples])[0, 1])

    def compare_methods(self, reference: np.ndarray, corrupted: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Benchmark the available recovery methods."""
        results = {}
        for method in ["stft", "harmonic_fit", "savgol", "omp"]:
            try:
                recovered = self.recover(corrupted, method=method)
                results[method] = {
                    "snr_db": self.snr(reference, recovered),
                    "rmse": self.rmse(reference, recovered),
                    "pearson_r": self.pearson_r(reference, recovered),
                }
            except Exception as exc:
                results[method] = {"error": str(exc)}
        return results
