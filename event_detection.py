"""
event_detection.py
===================
Power quality event detection using RMS and STFT features.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, stft


class EventDetector:
    """Detect and classify power quality events."""

    IEEE1159 = {
        "interruption": (0.00, 0.10),
        "sag": (0.10, 0.90),
        "normal": (0.90, 1.10),
        "swell": (1.10, 1.80),
        "overvoltage": (1.10, 9.99),
    }

    def __init__(self, fs: float = 10_000, f0: float = 50.0):
        self.fs = fs
        self.f0 = f0
        self._cycle_samples = int(fs / f0)

    def detect(
        self,
        signal: np.ndarray,
        t: np.ndarray,
        nominal_rms: Optional[float] = None,
    ) -> List[Dict]:
        """Run all event detectors and merge their results."""
        if nominal_rms is None:
            nominal_rms = self._estimate_nominal_rms(signal)

        events = []
        events.extend(self._detect_rms_events(signal, t, nominal_rms))
        events.extend(self._detect_transients_stft(signal, t))
        events.extend(self._detect_harmonic_bursts(signal, t))
        events = self._merge_events(events)
        events.sort(key=lambda ev: ev["time"])
        return events

    def _detect_rms_events(self, signal: np.ndarray, t: np.ndarray, nominal_rms: float) -> List[Dict]:
        rms_trace, t_rms = self._sliding_rms(signal, t)
        pu = rms_trace / (nominal_rms + 1e-12)

        events = []
        in_event = False
        start_idx = 0
        event_type = "normal"

        for idx, value in enumerate(pu):
            current_type = self._classify_rms(value)
            if not in_event and current_type != "normal":
                in_event = True
                start_idx = idx
                event_type = current_type
            elif in_event and (current_type != event_type or idx == len(pu) - 1):
                end_idx = idx
                t_start = t_rms[start_idx]
                t_end = t_rms[min(end_idx, len(t_rms) - 1)]
                events.append(
                    {
                        "type": event_type,
                        "start": float(t_start),
                        "end": float(t_end),
                        "time": float(t_start),
                        "duration": float(t_end - t_start),
                        "magnitude": float(np.mean(pu[start_idx:end_idx])),
                        "detector": "rms_sliding_window",
                    }
                )
                in_event = current_type != "normal"
                start_idx = idx
                event_type = current_type
        return events

    def _sliding_rms(self, signal: np.ndarray, t: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        n_frames = (len(signal) - self._cycle_samples) // stride
        rms_vals = np.zeros(n_frames)
        t_vals = np.zeros(n_frames)
        for idx in range(n_frames):
            seg = signal[idx * stride : idx * stride + self._cycle_samples]
            rms_vals[idx] = np.sqrt(np.mean(seg ** 2))
            t_vals[idx] = t[idx * stride + self._cycle_samples // 2]
        return rms_vals, t_vals

    def _classify_rms(self, pu: float) -> str:
        for label, (low, high) in self.IEEE1159.items():
            if low <= pu < high:
                return label
        return "normal"

    def _detect_transients_stft(
        self,
        signal: np.ndarray,
        t: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        threshold_sigma: float = 3.5,
    ) -> List[Dict]:
        """Detect transients from short bursts of high-frequency STFT energy."""
        if nperseg is None:
            nperseg = max(128, 2 * self._cycle_samples)
        if noverlap is None:
            noverlap = int(0.75 * nperseg)

        freqs, times, zxx = stft(
            signal,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
        )
        high_band = (freqs >= 8 * self.f0) & (freqs <= min(0.4 * self.fs, 2500))
        if not np.any(high_band):
            return []

        hf_energy = np.mean(np.abs(zxx[high_band, :]) ** 2, axis=0)
        threshold = np.median(hf_energy) + threshold_sigma * np.std(hf_energy)
        peaks, properties = find_peaks(hf_energy, height=threshold, distance=max(1, len(times) // 40))

        events = []
        for peak_idx, height in zip(peaks, properties["peak_heights"]):
            time_peak = float(times[peak_idx])
            start = max(float(t[0]), time_peak - nperseg / (2 * self.fs))
            end = min(float(t[-1]), time_peak + nperseg / (2 * self.fs))
            events.append(
                {
                    "type": "transient",
                    "start": start,
                    "end": end,
                    "time": time_peak,
                    "duration": float(end - start),
                    "magnitude": float(height / (np.median(hf_energy) + 1e-12)),
                    "detector": "stft_high_frequency_energy",
                }
            )
        return events

    def _detect_harmonic_bursts(
        self,
        signal: np.ndarray,
        t: np.ndarray,
        window_len: Optional[int] = None,
        thd_threshold: float = 0.20,
    ) -> List[Dict]:
        """Detect intervals where short-time THD exceeds a threshold."""
        if window_len is None:
            window_len = self._cycle_samples
        hop = window_len // 2
        n_frames = (len(signal) - window_len) // hop

        events = []
        thd_trace = []
        t_trace = []
        for idx in range(n_frames):
            seg = signal[idx * hop : idx * hop + window_len]
            freqs = np.fft.rfftfreq(window_len, d=1.0 / self.fs)
            mag = np.abs(np.fft.rfft(seg)) * 2 / window_len

            def pick(target: float) -> float:
                mask = np.abs(freqs - target) < 3
                return float(np.max(mag[mask])) if np.any(mask) else 0.0

            fundamental = pick(self.f0)
            harmonic_rms = np.sqrt(
                sum(
                    pick(order * self.f0) ** 2
                    for order in range(2, 15)
                    if order * self.f0 < self.fs / 2
                )
            )
            thd_trace.append(harmonic_rms / (fundamental + 1e-12))
            t_trace.append(t[idx * hop + window_len // 2])

        thd_trace = np.asarray(thd_trace)
        t_trace = np.asarray(t_trace)
        above = thd_trace > thd_threshold
        in_event = False

        for idx, flag in enumerate(above):
            if flag and not in_event:
                event_start = t_trace[idx]
                event_values = [thd_trace[idx]]
                in_event = True
            elif flag and in_event:
                event_values.append(thd_trace[idx])
            elif not flag and in_event:
                event_end = t_trace[idx - 1]
                events.append(
                    {
                        "type": "harmonic_burst",
                        "start": float(event_start),
                        "end": float(event_end),
                        "time": float(event_start),
                        "duration": float(event_end - event_start),
                        "magnitude": float(np.mean(event_values)),
                        "detector": "stft_thd",
                    }
                )
                in_event = False
        return events

    def _estimate_nominal_rms(self, signal: np.ndarray) -> float:
        n_cycles = len(signal) // self._cycle_samples
        rms_vals = []
        for idx in range(n_cycles):
            seg = signal[idx * self._cycle_samples : (idx + 1) * self._cycle_samples]
            rms_vals.append(np.sqrt(np.mean(seg ** 2)))
        return float(np.median(rms_vals)) if rms_vals else 1.0

    def _merge_events(self, events: List[Dict], gap_threshold: float = 0.002) -> List[Dict]:
        if not events:
            return []
        merged = [sorted(events, key=lambda ev: ev["time"])[0]]
        for event in sorted(events, key=lambda ev: ev["time"])[1:]:
            last = merged[-1]
            if event["type"] == last["type"] and event["start"] - last["end"] < gap_threshold:
                last["end"] = max(last["end"], event["end"])
                last["duration"] = last["end"] - last["start"]
                last["magnitude"] = 0.5 * (last["magnitude"] + event["magnitude"])
            else:
                merged.append(event)
        return merged

    def rms_profile(self, signal: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the RMS profile for plotting."""
        return self._sliding_rms(signal, t)

    def stft_event_map(
        self,
        signal: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an STFT energy map for plotting."""
        if nperseg is None:
            nperseg = max(128, 2 * self._cycle_samples)
        if noverlap is None:
            noverlap = int(0.75 * nperseg)

        freqs, times, zxx = stft(
            signal,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
        )
        return times, freqs, np.abs(zxx) ** 2
