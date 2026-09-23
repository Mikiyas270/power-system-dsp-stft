"""Synthetic voltage/current waveforms used by the examples."""

from typing import Dict, List, Optional

import numpy as np


class PowerSystemSignal:
    """
    Synthetic power-system signal generator.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration : float
        Signal duration in seconds.
    f0 : float
        Fundamental frequency in Hz (50 or 60).
    amplitude : float
        Fundamental peak amplitude (p.u.).
    """

    def __init__(
        self,
        fs: float = 10_000,
        duration: float = 0.2,
        f0: float = 50.0,
        amplitude: float = 1.0,
    ):
        self.fs = fs
        self.duration = duration
        self.f0 = f0
        self.amplitude = amplitude
        self.dt = 1.0 / fs
        self.t = np.arange(0, duration, self.dt)

    # Signal generation

    def generate(
        self,
        harmonics: Optional[Dict[int, float]] = None,
        noise_level: float = 0.02,
        events: Optional[List[dict]] = None,
        phase_deg: float = 0.0,
    ) -> tuple:
        """
        Build a composite power-system signal.

        Parameters
        ----------
        harmonics : dict  {order: relative_magnitude}
            e.g. {1: 1.0, 3: 0.15, 5: 0.10}
        noise_level : float
            Std-dev of additive white Gaussian noise (relative to amplitude).
        events : list of dicts
            Each dict must have 'type', 'start', 'end', 'magnitude'.
        phase_deg : float
            Initial phase of fundamental in degrees.

        Returns
        -------
        t : np.ndarray  – time vector (s)
        signal : np.ndarray – composite signal
        """
        if harmonics is None:
            harmonics = {1: 1.0, 3: 0.10, 5: 0.06}

        t = self.t.copy()
        signal = np.zeros_like(t)
        phase_rad = np.deg2rad(phase_deg)

        # Harmonic synthesis
        for order, rel_mag in harmonics.items():
            signal += (
                self.amplitude
                * rel_mag
                * np.sin(2 * np.pi * order * self.f0 * t + phase_rad)
            )

        # Additive white Gaussian noise
        if noise_level > 0:
            signal += np.random.normal(0, noise_level * self.amplitude, len(t))

        # Power quality events
        if events:
            for ev in events:
                signal = self._apply_event(signal, t, ev)

        return t, signal

    def generate_three_phase(self, **kwargs) -> tuple:
        """Generate balanced three-phase signals (a, b, c)."""
        t, va = self.generate(phase_deg=0, **kwargs)
        _, vb = self.generate(phase_deg=-120, **kwargs)
        _, vc = self.generate(phase_deg=120, **kwargs)
        return t, va, vb, vc

    # Event injection

    def _apply_event(self, signal: np.ndarray, t: np.ndarray, ev: dict) -> np.ndarray:
        """Inject a single power quality event into the signal."""
        mask = (t >= ev["start"]) & (t <= ev["end"])
        ev_type = ev.get("type", "sag").lower()
        mag = ev.get("magnitude", 0.5)

        if ev_type == "sag":
            signal[mask] *= (1.0 - mag)

        elif ev_type == "swell":
            signal[mask] *= (1.0 + mag)

        elif ev_type == "interruption":
            signal[mask] = 0.0

        elif ev_type == "transient":
            # High-frequency damped transient injected at start
            idx_start = np.searchsorted(t, ev["start"])
            n = np.sum(mask)
            t_ev = np.arange(n) / self.fs
            f_trans = ev.get("transient_freq", 1000.0)  # Hz
            decay = ev.get("decay", 500.0)              # 1/s
            transient = (
                mag
                * self.amplitude
                * np.exp(-decay * t_ev)
                * np.sin(2 * np.pi * f_trans * t_ev)
            )
            signal[idx_start : idx_start + n] += transient

        elif ev_type == "flicker":
            # Amplitude modulation to simulate flicker
            f_flicker = ev.get("flicker_freq", 10.0)
            modulation = 1.0 + mag * np.sin(2 * np.pi * f_flicker * t[mask])
            signal[mask] *= modulation

        elif ev_type == "harmonic_injection":
            # Inject a specific harmonic during the event window
            order = ev.get("order", 5)
            signal[mask] += (
                mag
                * self.amplitude
                * np.sin(2 * np.pi * order * self.f0 * t[mask])
            )

        return signal

    # Small calculation helpers

    def rms(self, signal: np.ndarray) -> float:
        """Compute true RMS of a signal."""
        return float(np.sqrt(np.mean(signal**2)))

    def thd_theoretical(self, harmonics: Dict[int, float]) -> float:
        """
        Compute theoretical THD from harmonic magnitudes.
        THD = sqrt(sum of squares of harmonics 2..N) / fundamental
        """
        fund = harmonics.get(1, 1.0)
        others = [v for k, v in harmonics.items() if k != 1]
        return np.sqrt(sum(v**2 for v in others)) / fund
