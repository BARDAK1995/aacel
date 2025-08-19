import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.fft import rfft, rfftfreq


# -------------------------
# Data loading and helpers
# -------------------------

def load_accel_data(filename: str, sample_rate_hz: float = 30.0) -> pd.DataFrame:
    """
    Load accelerometer data stored as a semicolon-separated CSV with columns
    `index;x;y;z`, and add a `time` column in seconds.

    Args:
        filename: Path to the data file.
        sample_rate_hz: Sampling frequency used to compute time from index.

    Returns:
        DataFrame with columns: index, x, y, z, time.
    """
    df = pd.read_csv(filename, sep=';')
    if 'index' not in df.columns or not {'x', 'y', 'z'}.issubset(df.columns):
        raise ValueError("Expected columns: 'index;x;y;z' (semicolon-separated).")
    df['time'] = df['index'] / float(sample_rate_hz)
    return df


def compute_total_acceleration_g(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total acceleration magnitude in g from x, y, z components and store
    it as `a_total_g`.

    Args:
        df: DataFrame containing columns x, y, z.

    Returns:
        Same DataFrame with an added `a_total_g` column.
    """
    a_total = np.sqrt(df['x'].to_numpy() ** 2 + df['y'].to_numpy() ** 2 + df['z'].to_numpy() ** 2)
    df = df.copy()
    df['a_total_g'] = a_total
    return df


def get_sample_rate_hz(time_array: np.ndarray) -> float:
    """
    Estimate sampling frequency from a time array.
    """
    dt = np.median(np.diff(time_array))
    if dt <= 0:
        raise ValueError("Non-positive time step encountered.")
    return 1.0 / dt


def select_time_window(df: pd.DataFrame, start_time_s: float, window_s: float) -> pd.DataFrame:
    """
    Slice the DataFrame rows within [start_time_s, start_time_s + window_s].
    """
    end_time_s = start_time_s + window_s
    return df[(df['time'] >= start_time_s) & (df['time'] <= end_time_s)].copy()


# -------------------------
# Filtering and transforms
# -------------------------

def butter_bandpass(lowcut_hz: float, highcut_hz: float, fs_hz: float, order: int = 4):
    nyq = 0.5 * fs_hz
    low = max(lowcut_hz, 1e-6) / nyq
    high = min(highcut_hz, nyq * 0.99) / nyq
    if not (0 < low < high < 1):
        # Fallback to a high-pass if the band is invalid (e.g., too close to Nyquist)
        high = min(0.99, high)
        low = max(1e-3, min(low, 0.49))
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(signal: np.ndarray, fs_hz: float, fmin_hz: float = 0.2, fmax_hz: float = 5.0, order: int = 4) -> np.ndarray:
    """
    Zero-phase band-pass filter to isolate suspension oscillation band.
    """
    # Detrend (remove DC and slow drift) before filtering
    x = signal - np.median(signal)
    b, a = butter_bandpass(fmin_hz, fmax_hz, fs_hz, order=order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xf = filtfilt(b, a, x, method='gust')
    return xf


# -------------------------
# Frequency estimation
# -------------------------

def estimate_dominant_frequency_hz(time_s: np.ndarray,
                                   signal_g: np.ndarray,
                                   fmin_hz: float = 0.2,
                                   fmax_hz: float = 5.0) -> float:
    """
    Estimate dominant oscillation frequency via rFFT peak within [fmin_hz, fmax_hz].

    Args:
        time_s: Time array in seconds.
        signal_g: Signal array (g units suggested).
        fmin_hz, fmax_hz: Frequency band to search.

    Returns:
        Dominant frequency in Hz (np.nan if not enough data).
    """
    if len(time_s) < 8:
        return float('nan')

    fs_hz = get_sample_rate_hz(time_s)
    x = bandpass_filter(signal_g, fs_hz, fmin_hz, fmax_hz)

    # Apply a Hann window to reduce spectral leakage
    n = len(x)
    window = np.hanning(n)
    X = rfft(x * window)
    freqs = rfftfreq(n, d=1.0 / fs_hz)
    mag = np.abs(X)

    # Focus on the band of interest
    band_mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not np.any(band_mask):
        return float('nan')

    idx_band = np.where(band_mask)[0]
    idx_peak_local = idx_band[np.argmax(mag[band_mask])]
    return float(freqs[idx_peak_local])


# -------------------------
# Damping estimation
# -------------------------

@dataclass
class DampingResult:
    zeta: float
    log_decrement: float
    fd_hz: float
    fn_hz: float
    sigma: float  # decay rate (1/s)
    tau: float    # time constant (s)


def estimate_damping_ratio(time_s: np.ndarray,
                           signal_g: np.ndarray,
                           known_fd_hz: float | None = None,
                           fmin_hz: float = 0.2,
                           fmax_hz: float = 5.0,
                           bandwidth_hz: float | None = None) -> DampingResult:
    """
    Estimate damping from the decay of peak amplitudes using log decrement.

    Steps:
      1) Band-pass filter to the expected oscillation band.
      2) Find positive peaks; compute average log decrement between successive peaks.
      3) Convert log decrement to damping ratio (zeta).
      4) Estimate damped frequency from average period between positive peaks.

    Returns:
        DampingResult with zeta, log_decrement, fd_hz, fn_hz, sigma, tau.
        If insufficient peaks, returns NaNs.
    """
    if len(time_s) < 8:
        return DampingResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    fs_hz = get_sample_rate_hz(time_s)

    # If the dominant frequency is provided, use a narrow band around it for precise envelope fitting
    if known_fd_hz is not None and np.isfinite(known_fd_hz) and known_fd_hz > 0:
        # Determine a bandwidth. Default: 50% of fd, but no narrower than 0.3 Hz and no wider than (fmax-fmin)/2
        if bandwidth_hz is None:
            bandwidth_hz = max(0.3, 0.5 * known_fd_hz)
        low = max(fmin_hz, known_fd_hz - bandwidth_hz)
        high = min(fmax_hz, known_fd_hz + bandwidth_hz)
        if low >= high:
            # Fallback to a small symmetric band
            half_bw = max(0.2, min(0.8, 0.5 * known_fd_hz))
            low, high = known_fd_hz - half_bw, known_fd_hz + half_bw
            low = max(1e-3, low)
        x = bandpass_filter(signal_g, fs_hz, low, high)

        # Hilbert envelope for amplitude decay
        analytic = hilbert(x)
        envelope = np.abs(analytic)

        # Avoid edge artifacts: trim 1 period from each edge when possible
        trim_t = 1.0 / known_fd_hz
        mask_center = (time_s >= time_s[0] + trim_t) & (time_s <= time_s[-1] - trim_t)
        if np.sum(mask_center) >= 8:
            t_fit = time_s[mask_center]
            env_fit = envelope[mask_center]
        else:
            t_fit = time_s
            env_fit = envelope

        # Exclude low SNR parts near noise floor for stability
        env_eps = 1e-9
        thr = 0.1 * np.nanmax(env_fit) + env_eps
        good = env_fit > thr
        if np.sum(good) < 8:
            # If too few, relax threshold
            thr = 0.05 * np.nanmax(env_fit) + env_eps
            good = env_fit > thr
        if np.sum(good) < 4:
            # Not enough points to fit reliably
            return DampingResult(np.nan, np.nan, known_fd_hz, np.nan, np.nan, np.nan)

        y = np.log(env_fit[good] + env_eps)
        t = t_fit[good]

        # Linear fit: ln(A) = ln(A0) - sigma * t
        slope, intercept = np.polyfit(t, y, 1)
        sigma = -float(slope)
        sigma = float(np.clip(sigma, 0.0, np.inf))

        omega_d = 2.0 * np.pi * known_fd_hz
        omega_n = float(np.sqrt(omega_d ** 2 + sigma ** 2))
        zeta = sigma / omega_n if omega_n > 0 else np.nan
        fn_hz = omega_n / (2.0 * np.pi)
        # Log decrement for one damped cycle: δ = sigma * T_d
        T_d = 1.0 / known_fd_hz
        delta = sigma * T_d
        tau = 1.0 / sigma if sigma > 0 else np.nan

        return DampingResult(zeta=zeta, log_decrement=delta, fd_hz=known_fd_hz, fn_hz=fn_hz, sigma=sigma, tau=tau)

    # Fallback: unknown frequency -> use peak-based log decrement within provided band
    x = bandpass_filter(signal_g, fs_hz, fmin_hz, fmax_hz)

    # Find positive peaks (same-phase maxima)
    min_distance_s = max(0.3, 0.4 / max(fmin_hz, 1e-3))
    min_distance_samples = max(1, int(min_distance_s * fs_hz))
    prominence = 0.1 * np.std(x) + 1e-6
    peaks, _ = find_peaks(x, distance=min_distance_samples, prominence=prominence)

    if len(peaks) < 3:
        return DampingResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    peak_times = time_s[peaks]
    peak_amps = x[peaks]

    mask_positive = peak_amps > 0
    peak_times = peak_times[mask_positive]
    peak_amps = peak_amps[mask_positive]
    if len(peak_amps) < 3:
        return DampingResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    periods = np.diff(peak_times)
    T_avg = float(np.mean(periods)) if len(periods) > 0 else np.nan
    fd_hz = 1.0 / T_avg if np.isfinite(T_avg) and T_avg > 0 else np.nan

    deltas = np.log(peak_amps[:-1] / peak_amps[1:])
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(deltas) == 0:
        return DampingResult(np.nan, np.nan, fd_hz, np.nan, np.nan, np.nan)

    delta = float(np.mean(deltas))

    zeta = delta / np.sqrt((2.0 * np.pi) ** 2 + delta ** 2)
    if np.isfinite(fd_hz):
        omega_d = 2.0 * np.pi * fd_hz
        omega_n = omega_d / np.sqrt(max(1e-12, 1.0 - zeta ** 2))
        sigma = zeta * omega_n
        tau = 1.0 / sigma if sigma > 0 else np.nan
        fn_hz = omega_n / (2.0 * np.pi)
    else:
        sigma = np.nan
        tau = np.nan
        fn_hz = np.nan

    return DampingResult(zeta=zeta, log_decrement=delta, fd_hz=fd_hz, fn_hz=fn_hz, sigma=sigma, tau=tau)


# -------------------------
# Plotting
# -------------------------

def plot_total_acceleration(df: pd.DataFrame, show: bool = True) -> None:
    """
    Plot total acceleration vs time.
    """
    font_size = 16
    plt.figure(figsize=(12, 5))
    plt.plot(df['time'].to_numpy(), df['a_total_g'].to_numpy(), label='Total acceleration |g|')
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Acceleration (g)', fontsize=font_size)
    plt.title('Total Acceleration (Magnitude)', fontsize=font_size + 2)
    plt.grid(True)
    plt.legend(fontsize=font_size - 2)
    plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
    if show:
        plt.show()


def plot_total_with_window(df: pd.DataFrame, start_time_s: float, window_s: float, show: bool = True) -> None:
    """
    Plot total acceleration with the analysis window highlighted.
    """
    font_size = 16
    t0 = start_time_s
    t1 = start_time_s + window_s
    plt.figure(figsize=(12, 5))
    plt.plot(df['time'].to_numpy(), df['a_total_g'].to_numpy(), label='Total acceleration |g|')
    plt.axvspan(t0, t1, color='orange', alpha=0.2, label=f'Window [{t0:.2f}, {t1:.2f}] s')
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Acceleration (g)', fontsize=font_size)
    plt.title('Total Acceleration with Analysis Window', fontsize=font_size + 2)
    plt.grid(True)
    plt.legend(fontsize=font_size - 2)
    plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
    if show:
        plt.show()


def plot_window_analysis(time_s: np.ndarray,
                         signal_g: np.ndarray,
                         fmin_hz: float,
                         fmax_hz: float,
                         damping: DampingResult | None = None,
                         show: bool = True) -> None:
    """
    Plot the windowed signal and overlay peaks and an exponential envelope if available.
    """
    fs_hz = get_sample_rate_hz(time_s)
    x = bandpass_filter(signal_g, fs_hz, fmin_hz, fmax_hz)

    font_size = 16
    plt.figure(figsize=(12, 5))
    plt.plot(time_s, signal_g, color='tab:gray', alpha=0.4, label='Raw (window)')
    plt.plot(time_s, x, color='tab:blue', label='Bandpassed')

    if damping is not None and np.isfinite(damping.zeta):
        # Overlay envelope using sigma
        t_rel = time_s - time_s[0]
        # Estimate initial amplitude from bandpassed signal at t0
        A0 = np.max(np.abs(x[: max(1, int(0.5 * fs_hz))]))
        if np.isfinite(damping.sigma) and A0 > 0:
            envelope_pos = A0 * np.exp(-damping.sigma * t_rel)
            envelope_neg = -envelope_pos
            plt.plot(time_s, envelope_pos, 'r--', linewidth=1.5, label='Envelope (+)')
            plt.plot(time_s, envelope_neg, 'r--', linewidth=1.5, label='Envelope (-)')

        subtitle = (f"fd≈{damping.fd_hz:.2f} Hz, fn≈{damping.fn_hz:.2f} Hz, "
                    f"ζ≈{damping.zeta:.3f}, δ≈{damping.log_decrement:.3f}, τ≈{damping.tau:.2f} s")
    else:
        subtitle = "Insufficient peaks for damping estimate"

    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Acceleration (g)', fontsize=font_size)
    plt.title(f'Windowed Signal ({fmin_hz:.1f}-{fmax_hz:.1f} Hz). {subtitle}', fontsize=font_size)
    plt.grid(True)
    plt.legend(fontsize=font_size - 2)
    plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
    if show:
        plt.show()


# -------------------------
# High-level convenience
# -------------------------

def analyze_suspension(
    filename: str,
    start_time_s: float,
    window_s: float,
    sample_rate_hz: float = 30.0,
    fmin_hz: float = 0.2,
    fmax_hz: float = 5.0,
) -> dict:
    """
    Convenience function to:
      - Load data and compute total acceleration
      - Plot total acceleration with window highlighted
      - Compute dominant frequency and damping within the window
      - Plot the analyzed window with envelope overlay

    Returns a dict with the key outputs.
    """
    df = load_accel_data(filename, sample_rate_hz)
    df = compute_total_acceleration_g(df)

    # Show overview plots to help choose the window
    plot_total_acceleration(df, show=False)
    plot_total_with_window(df, start_time_s, window_s, show=False)

    # Slice window
    dwin = select_time_window(df, start_time_s, window_s)
    time_s = dwin['time'].to_numpy()
    a_tot = dwin['a_total_g'].to_numpy()

    freq_hz = estimate_dominant_frequency_hz(time_s, a_tot, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
    damping = estimate_damping_ratio(time_s, a_tot, known_fd_hz=freq_hz, fmin_hz=fmin_hz, fmax_hz=fmax_hz)

    plot_window_analysis(time_s, a_tot, fmin_hz=fmin_hz, fmax_hz=fmax_hz, damping=damping, show=False)

    plt.show()

    return {
        'dominant_frequency_hz': freq_hz,
        'damping': damping,
        'window_dataframe': dwin,
    }


# -------------------------
# CLI entry point
# -------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Suspension oscillation analysis from phone accelerometer data.')
    p.add_argument('--file', type=str, required=True, help='Path to CSV data (semicolon-separated: index;x;y;z).')
    p.add_argument('--start', type=float, default=0.0, help='Start time (s) of analysis window.')
    p.add_argument('--window', type=float, default=20.0, help='Duration (s) of analysis window.')
    p.add_argument('--sr', type=float, default=30.0, help='Sample rate (Hz) used to compute time from index.')
    p.add_argument('--fmin', type=float, default=0.2, help='Min frequency (Hz) for bandpass and search.')
    p.add_argument('--fmax', type=float, default=5.0, help='Max frequency (Hz) for bandpass and search.')
    return p


def main():
    args = _build_arg_parser().parse_args()
    result = analyze_suspension(
        filename=args.file,
        start_time_s=args.start,
        window_s=args.window,
        sample_rate_hz=args.sr,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
    )

    damping: DampingResult = result['damping']
    print('\n--- Analysis Results ---')
    print(f"Dominant frequency: {result['dominant_frequency_hz']:.3f} Hz" if np.isfinite(result['dominant_frequency_hz']) else 'Dominant frequency: n/a')
    if np.isfinite(damping.zeta):
        print(f"Damping ratio ζ: {damping.zeta:.4f}")
        print(f"Log decrement δ: {damping.log_decrement:.4f}")
        print(f"Damped frequency f_d: {damping.fd_hz:.3f} Hz")
        print(f"Natural frequency f_n: {damping.fn_hz:.3f} Hz")
        print(f"Decay rate σ: {damping.sigma:.4f} 1/s (τ≈{damping.tau:.2f} s)")
    else:
        print('Damping estimate: n/a (insufficient peaks)')


if __name__ == '__main__':
    main()


