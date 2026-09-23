"""
stft_visualizer.py
==================
Publication-quality STFT plots for the power system DSP suite.

Figures produced
----------------
1.  Raw signal + injected events overlay
2.  FFT spectrum + harmonic bar markers
3.  STFT power spectrogram (time-frequency heatmap)
4.  Per-harmonic RMS tracks from STFT
5.  Noise filtering comparison (STFT LP, STFT denoise, Butterworth)
6.  Signal recovery (corrupted → STFT comb → harmonic fit)
7.  THD bar chart + instantaneous THD trace
8.  STFT event detection overlay
9.  Master dashboard (all key panels)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Tuple


# ── Dark theme palette ──────────────────────────────────────────────
C = {
    "raw":       "#4FC3F7",
    "butter":    "#FF7043",
    "stft_lp":   "#66BB6A",
    "stft_den":  "#AB47BC",
    "recovered": "#FFA726",
    "harmfit":   "#EC407A",
    "corrupted": "#78909C",
    "reference": "#ECEFF1",
    "harmonic":  "#FFD54F",
    "event_sag": "#EF5350",
    "event_sw":  "#FF9800",
    "event_tr":  "#CE93D8",
    "thd_line":  "#F44336",
    "grid":      "#21262D",
}

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       C["grid"],
    "grid.linestyle":   "--",
    "grid.alpha":       0.55,
    "legend.facecolor": "#161B22",
    "legend.edgecolor": "#30363D",
    "font.family":      "monospace",
    "font.size":        9,
})

EVENT_COLORS = {
    "sag": C["event_sag"],
    "swell": C["event_sw"],
    "transient": C["event_tr"],
    "interruption": "#F44336",
    "harmonic_burst": "#FF9800",
}


class STFTVisualizer:

    def plot_all(
        self,
        t, signal,
        filtered_stft_lp, filtered_stft_denoise, filtered_butter,
        recovered_stft, recovered_harmfit, corrupted,
        fft_freqs, fft_mag,
        harmonics, thd,
        stft_freqs, stft_times, stft_power_db,
        harm_times, harm_tracks, inst_thd,
        events, trad_events,
        noise_stats, fs, f0, output_dir="output",
    ):
        os.makedirs(output_dir, exist_ok=True)

        self.fig1_signal(t, signal, trad_events, output_dir)
        self.fig2_fft(fft_freqs, fft_mag, harmonics, f0, output_dir)
        self.fig3_stft_spectrogram(stft_freqs, stft_times, stft_power_db, f0, events, output_dir)
        self.fig4_harmonic_tracks(harm_times, harm_tracks, output_dir)
        self.fig5_filtering(t, signal, filtered_stft_lp, filtered_stft_denoise, filtered_butter, output_dir)
        self.fig6_recovery(t, signal, corrupted, recovered_stft, recovered_harmfit, output_dir)
        self.fig7_thd(harmonics, thd, harm_times, inst_thd, output_dir)
        self.fig8_event_detection(stft_freqs, stft_times, stft_power_db, events, t, signal, output_dir)
        self.fig9_dashboard(
            t, signal, filtered_stft_denoise, recovered_stft,
            fft_freqs, fft_mag, harmonics, thd,
            stft_freqs, stft_times, stft_power_db,
            harm_times, harm_tracks, inst_thd,
            events, noise_stats, fs, f0, output_dir,
        )

    # Fig 1 – Raw signal + events
    def fig1_signal(self, t, signal, events, output_dir):
        fig, ax = plt.subplots(figsize=(13, 4))
        ms = t * 1000
        ax.plot(ms, signal, color=C["raw"], lw=0.7, label="Signal")
        for ev in events:
            col = EVENT_COLORS.get(ev["type"], "#FF9800")
            ax.axvspan(ev.get("start", ev["time"]) * 1000,
                       ev.get("end", ev["time"]) * 1000,
                       alpha=0.18, color=col)
            mid = (ev.get("start", ev["time"]) + ev.get("end", ev["time"])) / 2 * 1000
            ax.text(mid, ax.get_ylim()[1] * 0.88,
                    ev["type"].replace("_", "\n"),
                    ha="center", fontsize=7, color=col)
        ax.set(xlabel="Time (ms)", ylabel="Amplitude (p.u.)",
               title="Power System Signal with PQ Events")
        ax.grid(True)
        ax.legend()
        self._save(fig, output_dir, "fig1_signal_events.png")

    # Fig 2 – FFT spectrum
    def fig2_fft(self, freqs, mag, harmonics, f0, output_dir):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

        ax = axes[0]
        ax.semilogy(freqs, mag + 1e-10, color=C["raw"], lw=0.7)
        for order, (fh, mh) in harmonics.items():
            ax.axvline(fh, color=C["harmonic"], lw=0.8, ls="--", alpha=0.7)
            ax.text(fh + 8, mh * 2, f"H{order}", fontsize=7, color=C["harmonic"])
        ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (log)",
               title="FFT Spectrum – Full Range")
        ax.set_xlim(0, min(5000, freqs[-1]))
        ax.grid(True)

        ax2 = axes[1]
        zoom = freqs <= (max(harmonics) + 1) * f0 + 50
        ax2.bar(freqs[zoom], mag[zoom], width=3, color=C["raw"], alpha=0.6)
        for order, (fh, mh) in harmonics.items():
            ax2.bar(fh, mh, width=7, color=C["harmonic"], zorder=3,
                    label=f"H{order}" if order <= 5 else "")
        ax2.set(xlabel="Frequency (Hz)", ylabel="Magnitude (linear)",
               title="Harmonic Content (zoom)")
        ax2.grid(True)

        fig.suptitle("FFT Harmonic Analysis", fontsize=12)
        plt.tight_layout()
        self._save(fig, output_dir, "fig2_fft_harmonics.png")

    # Fig 3 – STFT spectrogram
    def fig3_stft_spectrogram(self, freqs, times, power_db, f0, events, output_dir):
        fig, ax = plt.subplots(figsize=(13, 5))
        t_ms = times * 1000
        f_plot = freqs[freqs <= 2000]
        p_plot = power_db[freqs <= 2000, :]

        im = ax.pcolormesh(t_ms, f_plot, p_plot, shading="gouraud",
                           cmap="magma", vmin=np.percentile(p_plot, 10),
                           vmax=np.percentile(p_plot, 99))
        plt.colorbar(im, ax=ax, label="Power (dB)")

        # Harmonic lines
        for h in range(1, 20):
            fh = h * f0
            if fh < 2000:
                ax.axhline(fh, color="white", lw=0.5, ls="--", alpha=0.35)
                ax.text(t_ms[-1] * 1.002, fh, f"H{h}", fontsize=6,
                        va="center", color="white", alpha=0.6)

        # Event markers
        for ev in events:
            col = EVENT_COLORS.get(ev["type"], "cyan")
            ax.axvline(ev["time"] * 1000, color=col, lw=1.0, ls=":", alpha=0.8)

        ax.set(xlabel="Time (ms)", ylabel="Frequency (Hz)",
               title="STFT Power Spectrogram (time-frequency)")
        ax.set_ylim(0, 2000)
        self._save(fig, output_dir, "fig3_stft_spectrogram.png")

    # Fig 4 – Per-harmonic RMS tracks
    def fig4_harmonic_tracks(self, times, tracks, output_dir):
        orders = sorted(tracks.keys())
        n = len(orders)
        palette = plt.cm.tab10(np.linspace(0, 1, n))

        fig, axes = plt.subplots(n, 1, figsize=(13, 1.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        t_ms = times * 1000

        for ax, order, col in zip(axes, orders, palette):
            ax.plot(t_ms, tracks[order], color=col, lw=0.9)
            ax.fill_between(t_ms, 0, tracks[order], color=col, alpha=0.3)
            ax.set_ylabel(f"H{order}", fontsize=8)
            ax.grid(True)
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle("Per-Harmonic RMS Amplitude Tracks (STFT)", fontsize=12)
        plt.tight_layout()
        self._save(fig, output_dir, "fig4_harmonic_tracks.png")

    # Fig 5 – Filtering comparison
    def fig5_filtering(self, t, raw, filt_lp, filt_denoise, filt_butter, output_dir):
        ms = t * 1000
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

        axes[0].plot(ms, raw,          color=C["raw"],      lw=0.6);  axes[0].set_title("Raw Signal")
        axes[1].plot(ms, filt_lp,      color=C["stft_lp"],  lw=0.8);
        axes[1].plot(ms, raw,          color=C["raw"],       lw=0.3, alpha=0.3)
        axes[1].set_title("STFT Low-Pass (cutoff 1500 Hz)")
        axes[2].plot(ms, filt_denoise, color=C["stft_den"], lw=0.8)
        axes[2].plot(ms, raw,          color=C["raw"],       lw=0.3, alpha=0.3)
        axes[2].set_title("STFT Soft-Threshold Denoising")
        axes[3].plot(ms, filt_butter,  color=C["butter"],   lw=0.8)
        axes[3].plot(ms, raw,          color=C["raw"],       lw=0.3, alpha=0.3)
        axes[3].set_title("Butterworth Low-Pass (reference)")
        axes[3].set_xlabel("Time (ms)")

        for ax in axes:
            ax.set_ylabel("Amplitude")
            ax.grid(True)

        fig.suptitle("Noise Filtering Comparison", fontsize=12)
        plt.tight_layout()
        self._save(fig, output_dir, "fig5_filtering.png")

    # Fig 6 – Signal Recovery
    def fig6_recovery(self, t, reference, corrupted, recovered_stft, recovered_hf, output_dir):
        ms = t * 1000
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

        axes[0].plot(ms, corrupted,         color=C["corrupted"],  lw=0.5, label="Corrupted")
        axes[0].plot(ms, reference,          color=C["reference"],  lw=1.0, ls="--", alpha=0.7, label="Reference")
        axes[0].set_title("Corrupted Signal vs Reference")

        axes[1].plot(ms, recovered_stft[:len(t)], color=C["recovered"], lw=0.8, label="STFT Comb Recovery")
        axes[1].plot(ms, reference,               color=C["reference"],  lw=1.0, ls="--", alpha=0.7, label="Reference")
        axes[1].set_title("STFT Comb Filter Recovery")

        axes[2].plot(ms, recovered_hf[:len(t)], color=C["harmfit"], lw=0.8, label="Harmonic Fit Recovery")
        axes[2].plot(ms, reference,              color=C["reference"], lw=1.0, ls="--", alpha=0.7, label="Reference")
        axes[2].set_title("Harmonic Least-Squares Recovery")
        axes[2].set_xlabel("Time (ms)")

        for ax in axes:
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=7)
            ax.grid(True)

        fig.suptitle("Signal Recovery", fontsize=12)
        plt.tight_layout()
        self._save(fig, output_dir, "fig6_recovery.png")

    # Fig 7 – THD bar + instantaneous THD
    def fig7_thd(self, harmonics, thd, harm_times, inst_thd, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Bar chart
        ax = axes[0]
        orders = sorted(harmonics.keys())
        v1 = harmonics.get(1, (0, 1.0))[1]
        ihd = [harmonics[o][1] / v1 * 100 for o in orders]
        bar_colors = [C["harmonic"] if o != 1 else C["raw"] for o in orders]
        bars = ax.bar([f"H{o}" for o in orders], ihd, color=bar_colors,
                      edgecolor="#30363D", linewidth=0.5)
        ax.axhline(thd * 100, color=C["thd_line"], lw=1.5, ls="--",
                   label=f"THD = {thd*100:.2f}%")
        for bar, val in zip(bars, ihd):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", fontsize=7)
        ax.set(xlabel="Harmonic Order", ylabel="IHD (%)",
               title="Harmonic Distortion (IEEE 519)")
        ax.legend()
        ax.grid(True, axis="y")

        # Instantaneous THD
        ax2 = axes[1]
        t_ms = harm_times * 1000
        ax2.plot(t_ms, inst_thd * 100, color=C["stft_lp"], lw=0.9)
        ax2.axhline(thd * 100, color=C["thd_line"], lw=1.2, ls="--",
                    label=f"Global THD = {thd*100:.1f}%")
        ax2.fill_between(t_ms, 0, inst_thd * 100, alpha=0.25, color=C["stft_lp"])
        ax2.set(xlabel="Time (ms)", ylabel="THD (%)",
               title="Instantaneous THD over Time (STFT)")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle("Total Harmonic Distortion Analysis", fontsize=12)
        plt.tight_layout()
        self._save(fig, output_dir, "fig7_thd.png")

    # Fig 8 – STFT event detection
    def fig8_event_detection(self, freqs, times, power_db, events, t, signal, output_dir):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        t_ms = times * 1000

        # STFT spectrogram (bottom)
        f_plot = freqs[freqs <= 3000]
        p_plot = power_db[freqs <= 3000, :]
        axes[1].pcolormesh(t_ms, f_plot, p_plot, shading="gouraud",
                           cmap="inferno",
                           vmin=np.percentile(p_plot, 5),
                           vmax=np.percentile(p_plot, 98))
        axes[1].set(ylabel="Frequency (Hz)", xlabel="Time (ms)",
                   title="STFT Spectrogram with Event Markers")

        # Signal + event spans (top)
        sig_ms = t * 1000
        axes[0].plot(sig_ms, signal, color=C["raw"], lw=0.7)
        axes[0].set(ylabel="Amplitude", title="Signal + STFT-Detected Events")

        for ev in events:
            col = EVENT_COLORS.get(ev["type"], "cyan")
            t_ev_ms = ev["time"] * 1000
            axes[0].axvline(t_ev_ms, color=col, lw=1.2, ls="--", alpha=0.85)
            axes[1].axvline(t_ev_ms, color=col, lw=1.2, ls="--", alpha=0.85)
            axes[0].text(t_ev_ms, axes[0].get_ylim()[1] * 0.85,
                         ev["type"][:10], ha="center", fontsize=6.5, color=col,
                         rotation=70)

        for ax in axes:
            ax.grid(True, alpha=0.4)

        fig.suptitle("STFT-Based Event Detection", fontsize=12)
        plt.tight_layout()
        self._save(fig, output_dir, "fig8_event_detection.png")

    # Fig 9 – Master Dashboard
    def fig9_dashboard(
        self,
        t, signal, filtered, recovered,
        fft_freqs, fft_mag,
        harmonics, thd,
        stft_freqs, stft_times, stft_power_db,
        harm_times, harm_tracks, inst_thd,
        events, noise_stats, fs, f0, output_dir,
    ):
        fig = plt.figure(figsize=(20, 13))
        fig.patch.set_facecolor("#0D1117")
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)
        ms = t * 1000

        ax_sig   = fig.add_subplot(gs[0, :2])
        ax_fft   = fig.add_subplot(gs[0, 2])
        ax_stft  = fig.add_subplot(gs[1, :2])
        ax_thd   = fig.add_subplot(gs[1, 2])
        ax_rec   = fig.add_subplot(gs[2, :2])
        ax_info  = fig.add_subplot(gs[2, 2])

        # Signal
        ax_sig.plot(ms, signal, color=C["raw"], lw=0.7)
        for ev in events:
            col = EVENT_COLORS.get(ev["type"], "cyan")
            ax_sig.axvline(ev["time"] * 1000, color=col, lw=1.0, ls="--", alpha=0.8)
        ax_sig.set(title="Signal + Events", ylabel="Amplitude")
        ax_sig.grid(True)

        # FFT
        zm = fft_freqs <= 1500
        ax_fft.semilogy(fft_freqs[zm], fft_mag[zm] + 1e-10, color=C["raw"], lw=0.7)
        for _, (fh, _) in harmonics.items():
            if fh <= 1500:
                ax_fft.axvline(fh, color=C["harmonic"], lw=0.7, ls="--", alpha=0.8)
        ax_fft.set(title="FFT Spectrum", xlabel="Hz")
        ax_fft.grid(True)

        # STFT spectrogram
        f_zm = stft_freqs <= 2000
        t_ms_stft = stft_times * 1000
        ax_stft.pcolormesh(
            t_ms_stft, stft_freqs[f_zm],
            stft_power_db[f_zm, :],
            shading="gouraud", cmap="magma",
            vmin=np.percentile(stft_power_db[f_zm, :], 10),
            vmax=np.percentile(stft_power_db[f_zm, :], 99),
        )
        for h in range(1, 15):
            fh = h * f0
            if fh < 2000:
                ax_stft.axhline(fh, color="white", lw=0.3, ls="--", alpha=0.3)
        ax_stft.set(title="STFT Spectrogram", ylabel="Hz", xlabel="Time (ms)")

        # Instantaneous THD
        ax_thd.plot(harm_times * 1000, inst_thd * 100, color=C["stft_lp"], lw=0.9)
        ax_thd.axhline(thd * 100, color=C["thd_line"], lw=1.2, ls="--",
                       label=f"Global {thd*100:.1f}%")
        ax_thd.set(title="Instantaneous THD (%)", xlabel="ms")
        ax_thd.legend(fontsize=7)
        ax_thd.grid(True)

        # Recovery
        ax_rec.plot(ms, signal,                  color=C["reference"],  lw=1.0,  ls="--", alpha=0.7, label="Reference")
        ax_rec.plot(ms, recovered[:len(t)],       color=C["recovered"],  lw=0.8,  label="STFT Recovery")
        ax_rec.set(title="Signal Recovery", ylabel="Amplitude", xlabel="Time (ms)")
        ax_rec.legend(fontsize=7)
        ax_rec.grid(True)

        # Info panel
        ax_info.axis("off")
        lines = [
            "── Summary ──────────",
            f"fs        : {fs/1000:.1f} kHz",
            f"f0        : {f0} Hz",
            f"THD (FFT) : {thd*100:.2f} %",
            f"THD (inst): {np.mean(inst_thd)*100:.2f} % avg",
            f"SNR       : {noise_stats['snr_db']:.1f} dB",
            f"Noise RMS : {noise_stats['noise_rms']:.4f}",
            f"Events    : {len(events)}",
            "",
            "── STFT Events ──────",
        ]
        for ev in events[:6]:
            lines.append(f"  {ev['type'][:12]:12s} {ev['time']*1000:.0f}ms")
        ax_info.text(
            0.04, 0.96, "\n".join(lines),
            transform=ax_info.transAxes,
            fontsize=8, va="top", fontfamily="monospace", color="#C9D1D9",
            bbox=dict(boxstyle="round", fc="#161B22", ec="#30363D"),
        )

        fig.suptitle("Power System DSP — STFT Analysis Dashboard",
                     fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, output_dir, "fig9_dashboard.png")

    def _save(self, fig, output_dir, filename, dpi=150):
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"   Saved: {path}")
