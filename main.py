"""Run the complete power-system DSP demonstration."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from signal_generator import PowerSystemSignal
from harmonic_analysis import HarmonicAnalyzer
from stft_analysis import STFTAnalyzer
from noise_filter import NoiseDetectorFilter
from signal_recovery import SignalRecovery
from event_detection import EventDetector
from stft_visualizer import STFTVisualizer


def main():
    print("=" * 62)
    print("  Power System DSP Analysis Suite  [STFT Edition]")
    print("=" * 62)

    fs = 10_000
    duration = 0.4
    f0 = 50
    nperseg = 512
    noverlap = 496

    print("\n[1] Generating power system signal ...")
    signal_source = PowerSystemSignal(fs=fs, duration=duration, f0=f0)
    t, signal = signal_source.generate(
        harmonics={1: 1.0, 3: 0.15, 5: 0.10, 7: 0.06, 9: 0.03},
        noise_level=0.04,
        events=[
            {"type": "sag",       "start": 0.10, "end": 0.18, "magnitude": 0.40},
            {"type": "transient", "start": 0.22, "end": 0.23,
             "magnitude": 1.30, "transient_freq": 1500.0, "decay": 700.0},
            {"type": "swell",     "start": 0.28, "end": 0.34, "magnitude": 0.40},
        ],
    )
    print(f"   Length : {len(signal)} samples  |  fs={fs} Hz  |  f0={f0} Hz")

    print("\n[2] Running FFT Harmonic Analysis ...")
    harmonic_analyzer = HarmonicAnalyzer(fs=fs, f0=f0)
    fft_freqs, fft_mag = harmonic_analyzer.compute_fft(signal)
    harmonics = harmonic_analyzer.extract_harmonics(fft_freqs, fft_mag, n_harmonics=10)
    thd = harmonic_analyzer.total_harmonic_distortion(harmonics)
    print(f"   THD = {thd*100:.2f} %")
    for order, (freq, mag) in harmonics.items():
        print(f"   H{order:>2}: {freq:6.1f} Hz  ->  {mag:.4f} p.u.")

    print("\n[3] STFT Analysis ...")
    stft_analyzer = STFTAnalyzer(fs=fs, f0=f0, nperseg=nperseg, noverlap=noverlap)
    stft_freqs, stft_times, stft_power_db = stft_analyzer.spectrogram(signal)
    harm_times, harm_tracks = stft_analyzer.harmonic_rms_tracks(signal, n_harmonics=9)
    _, inst_thd = stft_analyzer.instantaneous_thd(signal, n_harmonics=9)
    snr_stft = stft_analyzer.snr_stft(signal)
    print(f"   STFT frames : {len(stft_times)}  |  freq bins : {len(stft_freqs)}")
    print(f"   Instantaneous THD: mean={np.mean(inst_thd)*100:.1f}%  peak={np.max(inst_thd)*100:.1f}%")
    print(f"   STFT SNR estimate : {snr_stft:.1f} dB")

    print("\n[4] Noise Detection & STFT Filtering ...")
    noise_filter = NoiseDetectorFilter(fs=fs)
    noise_stats = noise_filter.detect_noise(signal, t)
    print(f"   Estimated SNR  : {noise_stats['snr_db']:.1f} dB")
    print(f"   Noise RMS      : {noise_stats['noise_rms']:.5f}")
    filtered_stft_lp = stft_analyzer.stft_lowpass(signal, cutoff=1500.0)
    filtered_stft_denoise = stft_analyzer.stft_denoise(signal, sigma_factor=3.0)
    filtered_butter = noise_filter.butterworth_lowpass(signal, cutoff=1500)
    print("   Filters applied: STFT low-pass + STFT soft-threshold + Butterworth")

    print("\n[5] Signal Recovery via STFT ...")
    corrupted = signal + np.random.normal(0, 0.30, len(signal))
    recovered_stft = stft_analyzer.stft_recover(corrupted, n_harmonics=9, bandwidth=20.0)
    recovery = SignalRecovery(fs=fs)
    recovery_snr = recovery.snr(signal, recovered_stft)
    recovered_harmfit = recovery.recover(corrupted, method="harmonic_fit", n_harmonics=9)
    snr_harmfit = recovery.snr(signal, recovered_harmfit)
    print(f"   Recovery SNR (STFT comb): {recovery_snr:.1f} dB")
    print(f"   Recovery SNR (harm. fit): {snr_harmfit:.1f} dB")

    print("\n[6] STFT Event Detection ...")
    sag_swell_events = stft_analyzer.stft_sag_swell_detection(
        signal, fundamental_band=25, sag_threshold=0.9, swell_threshold=1.1
    )
    sag_swell_events = sag_swell_events[1:-1]
    transient_events = stft_analyzer.stft_transient_detection(
        signal, high_band_hz=(800.0, 4000.0), threshold_sigma=5.0
    )
    all_stft_events = sorted(sag_swell_events + transient_events, key=lambda e: e["time"])
    event_detector = EventDetector(fs=fs, f0=f0)
    trad_events = event_detector.detect(signal, t)
    print(f"   STFT events  : {len(all_stft_events)}")
    for ev in all_stft_events:
        print(f"   -> {ev['type']:15s}  t={ev['time']:.4f}s  mag={ev.get('magnitude', 0):.3f}")

    print("\n[7] Generating STFT plots ...")
    viz = STFTVisualizer()
    viz.plot_all(
        t=t, signal=signal,
        filtered_stft_lp=filtered_stft_lp,
        filtered_stft_denoise=filtered_stft_denoise,
        filtered_butter=filtered_butter,
        recovered_stft=recovered_stft,
        recovered_harmfit=recovered_harmfit,
        corrupted=corrupted,
        fft_freqs=fft_freqs, fft_mag=fft_mag,
        harmonics=harmonics, thd=thd,
        stft_freqs=stft_freqs, stft_times=stft_times, stft_power_db=stft_power_db,
        harm_times=harm_times, harm_tracks=harm_tracks, inst_thd=inst_thd,
        events=all_stft_events, trad_events=trad_events,
        noise_stats=noise_stats, fs=fs, f0=f0, output_dir="output",
    )
    print("   Plots saved to  ./output/")
    print("\n✔  Analysis complete.\n")

    return {
        "thd": thd, "snr_stft": snr_stft,
        "noise_stats": noise_stats, "stft_events": all_stft_events,
        "harmonics": harmonics, "recovery_snr_stft": recovery_snr,
        "recovery_snr_harmfit": snr_harmfit,
    }


if __name__ == "__main__":
    main()
