# Power System DSP Analysis Suite (STFT Edition)

A Python-based digital signal processing project for analyzing power-system signals using the Fast Fourier Transform (FFT) and Short-Time Fourier Transform (STFT).

The project generates synthetic power-system waveforms and demonstrates harmonic analysis, noise detection and filtering, signal recovery, power-quality event detection, and visualization.

## Features

- Synthetic power-system signal generation
- Configurable harmonic content
- Additive white Gaussian noise
- Power-quality event simulation
  - Voltage sag
  - Voltage swell
  - Interruption
  - Transient
  - Flicker
  - Harmonic injection
- FFT-based harmonic analysis
- STFT time-frequency analysis
- Harmonic RMS tracking
- Total Harmonic Distortion (THD) analysis
- Noise-floor and SNR estimation
- STFT-based denoising and filtering
- Butterworth filtering
- Signal recovery using STFT and harmonic fitting
- Sag, swell, transient, and harmonic-burst event detection
- Automatic generation of analysis plots

## Project Structure

```text
power-system-dsp-stft/
├── main.py
├── signal_generator.py
├── harmonic_analysis.py
├── stft_analysis.py
├── noise_filter.py
├── signal_recovery.py
├── event_detection.py
├── stft_visualizer.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Main Files

- `main.py` — Runs the complete DSP analysis workflow.
- `signal_generator.py` — Generates synthetic power-system signals with harmonics, noise, and power-quality events.
- `harmonic_analysis.py` — Performs FFT/STFT harmonic analysis and distortion calculations.
- `stft_analysis.py` — Provides the main STFT analysis, tracking, filtering, recovery, and event-detection methods.
- `noise_filter.py` — Estimates noise characteristics and applies DSP filtering methods.
- `signal_recovery.py` — Reconstructs corrupted power-system waveforms.
- `event_detection.py` — Detects and classifies power-quality events.
- `stft_visualizer.py` — Generates the project plots and analysis dashboard.

## Requirements

The project requires Python and the following packages:

- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

From the project folder, run:

```bash
python main.py
```

The program runs the complete analysis workflow and saves the generated figures in the `output/` directory.

## Analysis Workflow

The main program demonstrates four primary DSP analysis areas:

1. **Harmonic Analysis** — FFT and STFT harmonic tracking
2. **Noise Detection and Filtering** — STFT-domain analysis and filtering
3. **Signal Recovery** — Recovery of corrupted signals using STFT and harmonic fitting
4. **Event Detection** — Detection of sag, swell, transient, and other power-quality events

## Project Team and Responsibilities

The work is divided into three reasonably independent parts so each contributor can own and commit a meaningful section of the project.

| Contributor | Main responsibility | Files |
| --- | --- | --- |
| Person 1 | Signal modelling, harmonic measurements, and waveform recovery | `signal_generator.py`, `harmonic_analysis.py`, `signal_recovery.py` |
| Person 2 | STFT processing and noise/filter design | `stft_analysis.py`, `noise_filter.py` |
| Person 3 | Event detection, plots, and full-program integration | `event_detection.py`, `stft_visualizer.py`, `main.py` |

Replace Person 1/2/3 with the team members' names before submission.

## Collaboration

Each contributor should work on a separate Git branch and commit only the files assigned to them. `TEAM_WORKFLOW.md` contains the exact branch and commit sequence.

## Notes

The `output/` directory is generated when the project is run and is excluded from Git tracking.
