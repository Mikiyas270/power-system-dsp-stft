"""
stft_analysis.py
================
Short-Time Fourier Transform (STFT) engine for power system DSP.

Replaces wavelet-based analysis with STFT methods for:
  - Time-frequency spectrogram
  - Per-harmonic RMS tracking over time
  - Harmonic phase tracking
  - Instantaneous frequency estimation
  - STFT-based noise floor estimation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import get_window, stft, istft


class STFTAnalyzer:
    """
    STFT-based time-frequency analysis for power system signals.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    f0 : float
        Fundamental frequency (Hz).
    nperseg : int
        STFT window length in samples.
    noverlap : int
        Number of overlapping samples between consecutive windows.
    window : str
        Window function ('hann', 'blackman', 'flattop', …).
    """

    def __init__(
        self,
        fs: float = 10_000,
        f0: float = 50.0,
        nperseg: int = 512,
        noverlap: Optional[int] = None,
        window: str = "hann",
    ):
        self.fs = fs
        self.f0 = f0
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg * 3 // 4
        self.window = window

    # Core STFT

    def compute_stft(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the STFT of a signal.

        Returns
        -------
        freqs : (n_freqs,)   – frequency bins (Hz)
        times : (n_frames,)  – time centres (s)
        Zxx   : (n_freqs, n_frames) complex STFT matrix
        """
        freqs, times, Zxx = stft(
            signal,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        return freqs, times, Zxx

    def spectrogram(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the STFT power spectrogram (magnitude squared).

        Returns
        -------
        freqs, times, power_dB  – power in dB re 1 (V²/Hz)
        """
        freqs, times, Zxx = self.compute_stft(signal)
        power = np.abs(Zxx) ** 2
        power_dB = 10 * np.log10(power + 1e-20)
        return freqs, times, power_dB

    def reconstruct(self, Zxx: np.ndarray) -> np.ndarray:
        """
        Reconstruct time-domain signal from (modified) STFT.

        Parameters
        ----------
        Zxx : complex STFT matrix from compute_stft()

        Returns
        -------
        signal : np.ndarray  – reconstructed time-domain signal
        """
        _, signal = istft(
            Zxx,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        return signal

    # Harmonic Tracking

    def harmonic_rms_tracks(
        self,
        signal: np.ndarray,
        n_harmonics: int = 10,
        search_bandwidth: float = 20.0,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Track per-harmonic RMS amplitude over time.

        Parameters
        ----------
        n_harmonics : int
            Number of harmonic orders (1 = fundamental).
        search_bandwidth : float
            ±Hz around each harmonic to search for the peak.

        Returns
        -------
        times : (n_frames,)              – time axis
        tracks : {order: rms_array}      – per-harmonic RMS vs time
        """
        freqs, times, Zxx = self.compute_stft(signal)
        mag = np.abs(Zxx)  # shape: (n_freqs, n_frames)
        tracks: Dict[int, np.ndarray] = {}

        for order in range(1, n_harmonics + 1):
            f_target = order * self.f0
            if f_target > self.fs / 2:
                break
            mask = (freqs >= f_target - search_bandwidth) & (
                freqs <= f_target + search_bandwidth
            )
            if not np.any(mask):
                tracks[order] = np.zeros(len(times))
                continue
            # RMS over the search band for each frame
            tracks[order] = np.sqrt(np.mean(mag[mask, :] ** 2, axis=0))

        return times, tracks

    def harmonic_phase_tracks(
        self,
        signal: np.ndarray,
        n_harmonics: int = 7,
        search_bandwidth: float = 3.0,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Track per-harmonic instantaneous phase angle over time.

        Returns
        -------
        times, phase_tracks : {order: phase_rad_array}
        """
        freqs, times, Zxx = self.compute_stft(signal)
        phase_tracks: Dict[int, np.ndarray] = {}

        for order in range(1, n_harmonics + 1):
            f_target = order * self.f0
            if f_target > self.fs / 2:
                break
            mask = (freqs >= f_target - search_bandwidth) & (
                freqs <= f_target + search_bandwidth
            )
            if not np.any(mask):
                phase_tracks[order] = np.zeros(len(times))
                continue
            # Phase of the bin with maximum amplitude
            idx_in_mask = np.argmax(np.abs(Zxx[mask, :]), axis=0)  # per frame
            all_phases = np.angle(Zxx[mask, :])
            phase_tracks[order] = all_phases[idx_in_mask, np.arange(len(times))]

        return times, phase_tracks

    def instantaneous_thd(
        self,
        signal: np.ndarray,
        n_harmonics: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
        times, tracks = self.harmonic_rms_tracks(signal, n_harmonics=n_harmonics)
    
        v1 = tracks.get(1, np.zeros(len(times)))
        
    
        # Avoid division by zero without distorting physics
        v1_safe = np.maximum(v1, 1e-8)
    
        harm_sq = np.zeros(len(times))
        for order, rms in tracks.items():
            if order != 1:
                harm_sq += rms ** 2
    
        thd = np.sqrt(harm_sq) / v1_safe
    
        return times, thd

    # Noise Detection

    def noise_floor(
        self, signal: np.ndarray, harmonic_bandwidth: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the noise floor spectrum by masking harmonic bins.

        Returns
        -------
        freqs : frequency axis
        noise_psd : noise power spectral density (averaged over time)
        """
        freqs, times, Zxx = self.compute_stft(signal)
        power = np.abs(Zxx) ** 2

        # Build harmonic mask
        harmonic_mask = np.zeros(len(freqs), dtype=bool)
        n_harm = int(self.fs / (2 * self.f0))
        for h in range(1, n_harm + 1):
            fh = h * self.f0
            if fh > self.fs / 2:
                break
            harmonic_mask |= np.abs(freqs - fh) <= harmonic_bandwidth

        # Average PSD over time, then zero harmonic bins
        avg_psd = np.mean(power, axis=1)
        noise_psd = avg_psd.copy()
        noise_psd[harmonic_mask] = np.nan
        return freqs, noise_psd

    def snr_stft(self, signal: np.ndarray) -> float:
        """
        Estimate SNR (dB) as signal power (harmonic bins) over noise power.
        """
        freqs, times, Zxx = self.compute_stft(signal)
        power = np.mean(np.abs(Zxx) ** 2, axis=1)   # avg over frames
        df = freqs[1] - freqs[0]

        harmonic_mask = np.zeros(len(freqs), dtype=bool)
        n_harm = int(self.fs / (2 * self.f0))
        for h in range(1, n_harm + 1):
            fh = h * self.f0
            if fh > self.fs / 2:
                break
            harmonic_mask |= np.abs(freqs - fh) <= 5.0

        sig_power = np.sum(power[harmonic_mask]) * df
        noise_power = np.sum(power[~harmonic_mask]) * df
        if noise_power < 1e-30:
            return float("inf")
        return float(10 * np.log10(sig_power / noise_power))

    # STFT-Based Filtering

    def stft_lowpass(
        self, signal: np.ndarray, cutoff: float = 1500.0
    ) -> np.ndarray:
        """
        STFT-domain low-pass filter: zero all bins above cutoff Hz.
        """
        freqs, times, Zxx = self.compute_stft(signal)
        Zxx_filtered = Zxx.copy()
        Zxx_filtered[freqs > cutoff, :] = 0.0
        return self.reconstruct(Zxx_filtered)[: len(signal)]

    def stft_harmonic_filter(
        self,
        signal: np.ndarray,
        keep_harmonics: List[int],
        bandwidth: float = 5.0,
    ) -> np.ndarray:
        """
        Keep only specified harmonic bins; zero everything else.
        Useful for recovering a clean fundamental or a subset of harmonics.

        Parameters
        ----------
        keep_harmonics : list of harmonic orders to retain, e.g. [1, 3, 5]
        bandwidth : Hz on each side of the harmonic bin to keep
        """
        freqs, times, Zxx = self.compute_stft(signal)
        mask = np.zeros(len(freqs), dtype=bool)
        for order in keep_harmonics:
            fh = order * self.f0
            mask |= np.abs(freqs - fh) <= bandwidth

        Zxx_filtered = np.zeros_like(Zxx)
        Zxx_filtered[mask, :] = Zxx[mask, :]
        return self.reconstruct(Zxx_filtered)[: len(signal)]

    def stft_denoise(
        self,
        signal: np.ndarray,
        threshold_db: Optional[float] = None,
        sigma_factor: float = 3.0,
    ) -> np.ndarray:
        """
        Soft-threshold denoising in the STFT domain (Wiener-like masking).

        If threshold_db is None, it is estimated automatically as
        σ_noise_floor × sigma_factor.

        Parameters
        ----------
        threshold_db : float or None
            Fixed magnitude threshold (linear amplitude units). If None,
            estimated from the noise floor.
        sigma_factor : float
            Multiplier on the estimated noise σ for automatic thresholding.
        """
        freqs, times, Zxx = self.compute_stft(signal)
        mag = np.abs(Zxx)

        if threshold_db is None:
            # Estimate noise level from non-harmonic bins
            harmonic_mask = np.zeros(len(freqs), dtype=bool)
            n_harm = int(self.fs / (2 * self.f0))
            for h in range(1, n_harm + 1):
                fh = h * self.f0
                if fh > self.fs / 2:
                    break
                harmonic_mask |= np.abs(freqs - fh) <= 5.0

            noise_mag = mag[~harmonic_mask, :]
            sigma = np.median(noise_mag) / 0.6745
            threshold = sigma_factor * sigma
        else:
            threshold = threshold_db

        # Soft threshold on magnitude
        gain = np.maximum(mag - threshold, 0.0) / (mag + 1e-30)
        Zxx_denoised = Zxx * gain
        return self.reconstruct(Zxx_denoised)[: len(signal)]

    # Signal Recovery via STFT

    def stft_recover(
        self,
        corrupted: np.ndarray,
        n_harmonics: int = 10,
        bandwidth: float = 8.0,
    ) -> np.ndarray:
        """
        Recover the power system signal by retaining only energy near
        expected harmonic frequencies in the STFT domain.
        Equivalent to a comb-filter in the time-frequency plane.

        Parameters
        ----------
        n_harmonics : int
            Number of harmonic orders to retain.
        bandwidth : float
            Frequency band width (±Hz) around each harmonic.
        """
        keep = list(range(1, n_harmonics + 1))
        return self.stft_harmonic_filter(corrupted, keep_harmonics=keep,
                                         bandwidth=bandwidth)

    # Event Detection

    def stft_event_energy(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-frame total STFT energy for RMS-based event detection.

        Returns
        -------
        times : (n_frames,)
        energy : (n_frames,)  – total spectral energy per frame
        """
        freqs, times, Zxx = self.compute_stft(signal)
        energy = np.sum(np.abs(Zxx) ** 2, axis=0)
        return times, energy

    def stft_transient_detection(
        self,
        signal: np.ndarray,
        high_band_hz: Tuple[float, float] = (1000.0, 5000.0),
        threshold_sigma: float = 5.0,
    ) -> List[Dict]:
        """
        Detect transients as frames with anomalously high energy in a
        specified frequency band.

        Parameters
        ----------
        high_band_hz : (f_low, f_high)  – frequency band to monitor
        threshold_sigma : float
            Frames exceeding mean + threshold_sigma × std are flagged.

        Returns
        -------
        List of dicts: {time, magnitude, type}
        """
        freqs, times, Zxx = self.compute_stft(signal)
        band_mask = (freqs >= high_band_hz[0]) & (freqs <= high_band_hz[1])
        band_energy = np.sum(np.abs(Zxx[band_mask, :]) ** 2, axis=0)

        mean_e = np.mean(band_energy)
        std_e = np.std(band_energy)
        threshold = mean_e + threshold_sigma * std_e

        events = []
        for i, e in enumerate(band_energy):
            if e > threshold:
                events.append({
                    "time": float(times[i]),
                    "magnitude": float(e / (std_e + 1e-12)),
                    "type": "transient",
                })
        return events

    def stft_sag_swell_detection(
        self,
        signal: np.ndarray,
        fundamental_band: float = 5.0,
        sag_threshold: float = 0.90,
        swell_threshold: float = 1.10,
    ) -> List[Dict]:
        """
        Detect voltage sag/swell events by monitoring the fundamental
        component amplitude per STFT frame.

        Returns
        -------
        List of dicts: {type, time, duration, magnitude}
        """
        freqs, times, Zxx = self.compute_stft(signal)
        f_mask = np.abs(freqs - self.f0) <= fundamental_band
        fund_amp = 2.0 * np.mean(np.abs(Zxx[f_mask, :]), axis=0)  # 2× for one-sided

        # Normalise to median (nominal amplitude)
        nominal = np.median(fund_amp)
        pu = fund_amp / (nominal + 1e-12)

        events = []
        in_ev = False
        ev_type = None
        ev_start_idx = 0

        for i, v in enumerate(pu):
            if v < sag_threshold:
                cur_type = "sag"
            elif v > swell_threshold:
                cur_type = "swell"
            else:
                cur_type = None

            if cur_type and not in_ev:
                in_ev = True
                ev_type = cur_type
                ev_start_idx = i
            elif in_ev and (cur_type != ev_type or i == len(pu) - 1):
                t_start = times[ev_start_idx]
                t_end = times[i - 1]
                events.append({
                    "type": ev_type,
                    "time": float(t_start),
                    "start": float(t_start),
                    "end": float(t_end),
                    "duration": float(t_end - t_start),
                    "magnitude": float(np.mean(pu[ev_start_idx:i])),
                    "detector": "stft_fundamental_tracking",
                })
                in_ev = bool(cur_type)
                ev_type = cur_type
                ev_start_idx = i

        return events
