import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Five ECG types
kinds = ['N', 'A', 'V', 'L', 'R']
kind_names = {
    'N': 'Normal (Normal)',
    'A': 'Atrial Premature (APC)',
    'V': 'Ventricular Premature (VPC)',
    'L': 'Left Bundle Branch Block (LBBB)',
    'R': 'Right Bundle Branch Block (RBBB)'
}

def generate_ecg_wave(ecg_type, duration=2.0, fs=360):
    """
    Generate ECG waveforms of different types

    Parameters:
    ecg_type: ECG type ('N', 'A', 'V', 'L', 'R')
    duration: Duration in seconds
    fs: Sampling frequency in Hz

    Returns:
    t: Time axis
    ecg: ECG signal
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    ecg = np.zeros_like(t)

    # Heart rate (BPM)
    if ecg_type == 'N':
        heart_rate = 70
    elif ecg_type == 'A':
        heart_rate = 75  # APC with compensatory pause
    elif ecg_type == 'V':
        heart_rate = 65  # VPC with compensatory pause
    elif ecg_type == 'L':
        heart_rate = 60
    elif ecg_type == 'R':
        heart_rate = 65
    else:
        heart_rate = 70

    # R-R interval
    rr_interval = 60.0 / heart_rate

    # Generate beat sequence
    beat_times = np.arange(0, duration, rr_interval)

    for beat_time in beat_times:
        if ecg_type == 'N':
            # Normal ECG
            ecg += add_normal_beat(t, beat_time, fs)
        elif ecg_type == 'A':
            # Atrial premature contraction - abnormal P wave, shortened PR interval
            if len(beat_times) > 1 and beat_time == beat_times[1]:
                ecg += add_apc_beat(t, beat_time, fs)
            else:
                ecg += add_normal_beat(t, beat_time, fs)
        elif ecg_type == 'V':
            # Ventricular premature contraction - wide bizarre QRS
            if len(beat_times) > 1 and beat_time == beat_times[1]:
                ecg += add_vpc_beat(t, beat_time, fs)
            else:
                ecg += add_normal_beat(t, beat_time, fs)
        elif ecg_type == 'L':
            # Left bundle branch block - wide QRS, M-shaped
            ecg += add_lbbb_beat(t, beat_time, fs)
        elif ecg_type == 'R':
            # Right bundle branch block - wide QRS with R' wave
            ecg += add_rbbb_beat(t, beat_time, fs)

    # Add baseline drift and noise
    baseline_drift = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    noise = 0.02 * np.random.normal(0, 1, len(t))

    ecg = ecg + baseline_drift + noise

    # Apply bandpass filter
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    ecg = signal.filtfilt(b, a, ecg)

    return t, ecg

def add_normal_beat(t, beat_time, fs):
    """Add normal heartbeat"""
    ecg_beat = np.zeros_like(t)
    idx = np.where((t >= beat_time - 0.1) & (t <= beat_time + 0.4))[0]

    if len(idx) > 0:
        t_beat = t[idx] - beat_time

        # P wave
        p_wave = 0.15 * np.exp(-((t_beat + 0.15)**2) / 0.002)

        # QRS complex
        q_wave = -0.1 * np.exp(-((t_beat + 0.02)**2) / 0.0005)
        r_wave = 1.0 * np.exp(-((t_beat)**2) / 0.001)
        s_wave = -0.2 * np.exp(-((t_beat - 0.02)**2) / 0.001)

        # T wave
        t_wave = 0.3 * np.exp(-((t_beat - 0.2)**2) / 0.01)

        ecg_beat[idx] = p_wave + q_wave + r_wave + s_wave + t_wave

    return ecg_beat

def add_apc_beat(t, beat_time, fs):
    """Add atrial premature contraction"""
    ecg_beat = np.zeros_like(t)
    idx = np.where((t >= beat_time - 0.1) & (t <= beat_time + 0.4))[0]

    if len(idx) > 0:
        t_beat = t[idx] - beat_time

        # Abnormal P wave (early, abnormal morphology)
        p_wave = 0.1 * np.exp(-((t_beat + 0.12)**2) / 0.003)

        # Shortened PR interval
        qrs_start = -0.02
        q_wave = -0.05 * np.exp(-((t_beat - qrs_start + 0.02)**2) / 0.0005)
        r_wave = 0.8 * np.exp(-((t_beat - qrs_start)**2) / 0.001)
        s_wave = -0.15 * np.exp(-((t_beat - qrs_start - 0.02)**2) / 0.001)

        # T wave
        t_wave = 0.25 * np.exp(-((t_beat - qrs_start - 0.2)**2) / 0.01)

        ecg_beat[idx] = p_wave + q_wave + r_wave + s_wave + t_wave

    return ecg_beat

def add_vpc_beat(t, beat_time, fs):
    """Add ventricular premature contraction"""
    ecg_beat = np.zeros_like(t)
    idx = np.where((t >= beat_time - 0.1) & (t <= beat_time + 0.4))[0]

    if len(idx) > 0:
        t_beat = t[idx] - beat_time

        # No P wave
        # Wide bizarre QRS
        qrs_width = 0.12  # 120ms
        qrs_amplitude = 1.5
        qrs = qrs_amplitude * np.exp(-((t_beat)**2) / (2 * qrs_width**2))
        qrs = qrs * (1 + 0.5 * np.sin(10 * np.pi * t_beat))  # Add notching

        # Secondary ST-T changes
        st_t = -0.2 * np.exp(-((t_beat - 0.05)**2) / 0.02)

        ecg_beat[idx] = qrs + st_t

    return ecg_beat

def add_lbbb_beat(t, beat_time, fs):
    """Add left bundle branch block"""
    ecg_beat = np.zeros_like(t)
    idx = np.where((t >= beat_time - 0.1) & (t <= beat_time + 0.4))[0]

    if len(idx) > 0:
        t_beat = t[idx] - beat_time

        # Normal P wave
        p_wave = 0.15 * np.exp(-((t_beat + 0.15)**2) / 0.002)

        # Wide QRS, M-shaped
        qrs_width = 0.16  # 160ms
        r1 = 0.5 * np.exp(-((t_beat)**2) / 0.002)
        r2 = 0.3 * np.exp(-((t_beat - 0.08)**2) / 0.003)
        s = -0.3 * np.exp(-((t_beat + 0.04)**2) / 0.002)
        qrs = r1 + s + r2

        # Abnormal ST-T
        st_t = 0.2 * np.exp(-((t_beat - 0.15)**2) / 0.015)

        ecg_beat[idx] = p_wave + qrs + st_t

    return ecg_beat

def add_rbbb_beat(t, beat_time, fs):
    """Add right bundle branch block"""
    ecg_beat = np.zeros_like(t)
    idx = np.where((t >= beat_time - 0.1) & (t <= beat_time + 0.4))[0]

    if len(idx) > 0:
        t_beat = t[idx] - beat_time

        # Normal P wave
        p_wave = 0.15 * np.exp(-((t_beat + 0.15)**2) / 0.002)

        # Wide QRS with R' wave
        qrs_width = 0.14  # 140ms
        r = 0.8 * np.exp(-((t_beat)**2) / 0.001)
        s = -0.4 * np.exp(-((t_beat - 0.02)**2) / 0.001)
        r_prime = 0.3 * np.exp(-((t_beat - 0.08)**2) / 0.002)
        qrs = r + s + r_prime

        # T wave
        t_wave = 0.25 * np.exp(-((t_beat - 0.2)**2) / 0.01)

        ecg_beat[idx] = p_wave + qrs + t_wave

    return ecg_beat

# Create figure
plt.figure(figsize=(15, 10))

# Generate and plot waveforms for each ECG type
for i, kind in enumerate(kinds):
    t, ecg = generate_ecg_wave(kind, duration=3.0, fs=360)

    plt.subplot(5, 1, i + 1)
    plt.plot(t, ecg, 'b-', linewidth=1.5)
    plt.title(f'{kind}: {kind_names[kind]}', fontsize=12)
    plt.ylabel('Amplitude (mV)', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Mark R peaks
    if kind == 'N':
        peaks, _ = signal.find_peaks(ecg, height=0.5, distance=200)
    elif kind == 'A':
        peaks, _ = signal.find_peaks(ecg, height=0.4, distance=180)
    elif kind == 'V':
        peaks, _ = signal.find_peaks(ecg, height=0.8, distance=200)
    elif kind == 'L':
        peaks, _ = signal.find_peaks(ecg, height=0.4, distance=200)
    elif kind == 'R':
        peaks, _ = signal.find_peaks(ecg, height=0.6, distance=200)

    if len(peaks) > 0:
        plt.plot(t[peaks], ecg[peaks], 'ro', markersize=4)

    # Set y-axis range
    plt.ylim([-0.5, 2.0])

    # Only show x-axis label on the last subplot
    if i == len(kinds) - 1:
        plt.xlabel('Time (s)', fontsize=10)

plt.tight_layout()
plt.suptitle('Five Types of ECG Waveforms', fontsize=16, y=1.02)
plt.show()