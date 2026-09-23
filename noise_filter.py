"""
noise_filter.py
================
Noise detection and STFT-based filtering for power system signals.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, firwin, iirnotch, istft, sosfiltfilt, stft, welch


class NoiseDetectorFilter:
    """Detect noise statistics and apply practical DSP filters."""

    def __init__(self, fs: float = 10_000, f0: float = 50.0):
        self.fs = fs
        self.f0 = f0

    def detect_noise(self, signal: np.ndarray, t: np.ndarray) -> Dict[str, object]:
        """Estimate SNR, noise floor, and dominant noise band."""
        del t
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(1024, len(signal) // 4))
        df = freqs[1] - freqs[0]

        signal_power = 0.0
        n_harmonics = int(self.fs / (2 * self.f0))
        for order in range(1, n_harmonics + 1):
            target = order * self.f0
            if target > self.fs / 2:
                break
            mask = np.abs(freqs - target) <= 5.0
            signal_power += np.sum(psd[mask]) * df

        total_power = np.sum(psd) * df
        noise_power = max(total_power - signal_power, 1e-20)
        signal_power = max(signal_power, 1e-20)
        snr_db = 10 * np.log10(signal_power / noise_power)

        harmonic_mask = np.zeros(len(freqs), dtype=bool)
        for order in range(1, n_harmonics + 1):
            target = order * self.f0
            if target > self.fs / 2:
                break
            harmonic_mask |= np.abs(freqs - target) <= 5.0

        noise_psd = psd.copy()
        noise_psd[harmonic_mask] = 0.0
        bands = {
            "sub-harmonic (<50 Hz)": (0, 50),
            "low (50-500 Hz)": (50, 500),
            "mid (500-2000 Hz)": (500, 2000),
            "high (2-5 kHz)": (2000, 5000),
        }
        band_powers = {}
        for name, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs < fhi)
            band_powers[name] = float(np.sum(noise_psd[mask]) * df)

        psd_pos = psd + 1e-30
        spectral_flatness = float(np.exp(np.mean(np.log(psd_pos))) / np.mean(psd_pos))

        return {
            "snr_db": float(snr_db),
            "noise_rms": float(self._mad_noise_estimate(signal)),
            "signal_rms": float(np.sqrt(np.mean(signal ** 2))),
            "dominant_band": max(band_powers, key=band_powers.get),
            "band_powers": band_powers,
            "spectral_flatness": spectral_flatness,
            "noise_floor_db": float(10 * np.log10(noise_power / len(freqs) + 1e-20)),
        }

    def _mad_noise_estimate(self, signal: np.ndarray) -> float:
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return float(mad / 0.6745)

    def butterworth_lowpass(self, signal: np.ndarray, cutoff: float = 1500.0, order: int = 6) -> np.ndarray:
        sos = butter(order, cutoff / (self.fs / 2), btype="low", output="sos")
        return sosfiltfilt(sos, signal)

    def butterworth_bandpass(self, signal: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
        sos = butter(order, [low / (self.fs / 2), high / (self.fs / 2)], btype="band", output="sos")
        return sosfiltfilt(sos, signal)

    def notch_filter(self, signal: np.ndarray, notch_freq: float, q_factor: float = 30.0) -> np.ndarray:
        b_coef, a_coef = iirnotch(notch_freq / (self.fs / 2), q_factor)
        return filtfilt(b_coef, a_coef, signal)

    def fir_lowpass(
        self,
        signal: np.ndarray,
        cutoff: float = 1500.0,
        num_taps: int = 101,
        window: str = "hamming",
    ) -> np.ndarray:
        coeffs = firwin(num_taps, cutoff / (self.fs / 2), window=window)
        padded = np.pad(signal, (num_taps - 1, num_taps - 1), mode="reflect")
        filtered = np.convolve(padded, coeffs, mode="valid")
        offset = (len(filtered) - len(signal)) // 2
        return filtered[offset : offset + len(signal)]

    def stft_denoise(
        self,
        signal: np.ndarray,
        nperseg: int = 256,
        noverlap: int = 192,
        attenuation: float = 0.15,
        noise_quantile: float = 0.35,
    ) -> np.ndarray:
        """Apply STFT spectral gating and overlap-add reconstruction."""
        freqs, times, zxx = stft(
            signal,
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
        cleaned_mag = np.maximum(magnitude - noise_floor, 0.0)
        cleaned_mag[magnitude < 1.5 * noise_floor] *= attenuation

        keep_band = freqs <= min(15 * self.f0, 0.45 * self.fs)
        cleaned_mag[~keep_band, :] *= attenuation

        _, filtered = istft(
            cleaned_mag * np.exp(1j * phase),
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=None,
        )
        return filtered[: len(signal)]

    def lms_filter(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        mu: float = 0.01,
        filter_length: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Least-mean-squares adaptive noise cancellation."""
        n_samples = len(signal)
        weights = np.zeros(filter_length)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)

        for idx in range(filter_length, n_samples):
            x_vec = reference[idx : idx - filter_length : -1]
            y_val = np.dot(weights, x_vec)
            err = signal[idx] - y_val
            weights += 2 * mu * err * x_vec
            output[idx] = y_val
            error[idx] = err

        return output, error

    def kalman_filter(
        self,
        signal: np.ndarray,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ) -> np.ndarray:
        """Scalar Kalman smoother for noisy signals."""
        n_samples = len(signal)
        x_hat = np.zeros(n_samples)
        cov = np.zeros(n_samples)
        x_hat[0] = signal[0]
        cov[0] = 1.0

        for idx in range(1, n_samples):
            x_pred = x_hat[idx - 1]
            cov_pred = cov[idx - 1] + process_noise
            gain = cov_pred / (cov_pred + measurement_noise)
            x_hat[idx] = x_pred + gain * (signal[idx] - x_pred)
            cov[idx] = (1 - gain) * cov_pred

        return x_hat
