"""
===============================================================================
Digital Communication Laboratory - Experiment 9
Amplitude Shift Keying (ASK) and Frequency Shift Keying (FSK)
===============================================================================
Programme: B.Tech in Electronics and Communication Engineering
Platform: Python 3 (NumPy, SciPy, Matplotlib, Pandas)
Laboratory Mode: Self-contained first-principles simulation without toolboxes

Description:
  Implements an end-to-end digital passband communication system to demonstrate:
  1. Passband waveform synthesis for Binary ASK (OOK) and Binary FSK (BFSK).
  2. Coherent correlator-based detection for ASK and BFSK.
  3. Non-coherent envelope / quadrature energy detection for ASK and BFSK.
  4. BFSK tone spacing condition for orthogonality (coherent vs non-coherent).
  5. Power Spectral Density (PSD) estimation using Welch's periodogram method.
  6. Decision-statistic distributions and noise margin histograms.
  7. Comprehensive Monte Carlo BER simulation vs. theoretical error curves.
  8. Mandatory syllabus validation: Numerical inner product calculation of
     BFSK basis signals over one bit interval to verify orthogonality.
===============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import scipy.signal as signal
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd

# Set plot styling for publication-quality visuals
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Setup Output Directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. CORE SYSTEM PARAMETERS
# =============================================================================
RB = 1000.0                # Bit rate = 1000 bits/sec
TB = 1.0 / RB              # Bit duration = 1.0 ms
FS = 100000.0              # Sampling frequency = 100 kHz
SPS = int(FS * TB)         # Samples per bit = 100 samples
FC_ASK = 10000.0           # ASK Carrier frequency = 10 kHz (10 cycles per bit)
F1_FSK = 10000.0           # BFSK Mark Frequency ('1') = 10 kHz (10 cycles per bit)
F2_FSK = 12000.0           # BFSK Space Frequency ('0') = 12 kHz (12 cycles per bit, Delta_f = 2 kHz)
AMP = 1.0                  # Transmit peak carrier amplitude (Volts)


# =============================================================================
# 2. SIGNAL GENERATION & MODULATION (FIRST PRINCIPLES)
# =============================================================================

def generate_random_bits(n_bits: int, seed: Optional[int] = 42) -> np.ndarray:
    """Generate independent, equiprobable binary data bits {0, 1}."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits)


def modulate_ask(bits: np.ndarray, tb: float = TB, fc: float = FC_ASK,
                 fs: float = FS, amp: float = AMP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Binary Amplitude Shift Keying (OOK) passband waveform.
    s(t) = A * cos(2*pi*fc*t) for bit 1, and 0 for bit 0.
    """
    n_bits = len(bits)
    sps = int(fs * tb)
    total_samples = n_bits * sps
    t = np.arange(total_samples) / fs

    # Baseband rectangular pulse train
    baseband = np.repeat(bits, sps)

    # Carrier waveform
    carrier = amp * np.cos(2.0 * np.pi * fc * t)

    # Passband modulated signal
    ask_signal = baseband * carrier

    return t, baseband, ask_signal


def modulate_bfsk(bits: np.ndarray, tb: float = TB, f1: float = F1_FSK, f2: float = F2_FSK,
                  fs: float = FS, amp: float = AMP, phase_continuous: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Binary Frequency Shift Keying (BFSK) passband waveform.
    Bit 1 -> Frequency f1
    Bit 0 -> Frequency f2
    """
    n_bits = len(bits)
    sps = int(fs * tb)
    total_samples = n_bits * sps
    t = np.arange(total_samples) / fs
    baseband = np.repeat(bits, sps)

    bfsk_signal = np.zeros(total_samples)
    current_phase = 0.0

    for i, b in enumerate(bits):
        idx_start = i * sps
        idx_end = (i + 1) * sps
        t_segment = t[idx_start:idx_end] - t[idx_start]
        freq = f1 if b == 1 else f2

        if phase_continuous:
            # Continuous phase FSK (CPFSK)
            phase = current_phase + 2.0 * np.pi * freq * t_segment
            bfsk_signal[idx_start:idx_end] = amp * np.cos(phase)
            current_phase = (current_phase + 2.0 * np.pi * freq * tb) % (2.0 * np.pi)
        else:
            # Standard coherent BFSK with fixed phase
            bfsk_signal[idx_start:idx_end] = amp * np.cos(2.0 * np.pi * freq * (t[idx_start:idx_end]))

    return t, baseband, bfsk_signal


# =============================================================================
# 3. CHANNEL MODEL: ADDITIVE WHITE GAUSSIAN NOISE (AWGN)
# =============================================================================

def add_awgn(signal_in: np.ndarray, snr_db: float, signal_power: Optional[float] = None,
             seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Add AWGN to a passband waveform for a specified SNR in dB.
    Returns: (noisy_signal, measured_noise_variance)
    """
    if signal_power is None:
        signal_power = np.mean(signal_in ** 2)

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_variance), size=len(signal_in))
    noisy_signal = signal_in + noise

    return noisy_signal, noise_variance


# =============================================================================
# 4. COHERENT CORRELATOR-BASED RECEIVERS
# =============================================================================

def demodulate_ask_coherent(r_t: np.ndarray, n_bits: int, tb: float = TB, fc: float = FC_ASK,
                            fs: float = FS, amp: float = AMP) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Coherent Correlator Receiver for Binary ASK (OOK).
    Correlates received signal with basis function phi(t) = sqrt(2/Tb) * cos(2*pi*fc*t).
    Decision threshold is gamma = sqrt(Eb) / 2.
    """
    sps = int(fs * tb)
    dt = 1.0 / fs
    basis = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * fc * (np.arange(sps) / fs))

    correlator_outputs = np.zeros(n_bits)
    for i in range(n_bits):
        r_seg = r_t[i * sps:(i + 1) * sps]
        # Discrete numerical integration over [0, Tb]
        correlator_outputs[i] = np.sum(r_seg * basis) * dt

    # Theoretical energy for bit 1: E_s1 = A^2 * Tb / 2.
    # Energy metric in correlator space: sqrt(E_s1) = A * sqrt(Tb / 2).
    # Midpoint optimal threshold for equiprobable bits:
    v_peak = amp * np.sqrt(tb / 2.0)
    threshold = v_peak / 2.0

    detected_bits = (correlator_outputs > threshold).astype(int)
    return detected_bits, correlator_outputs, threshold


def demodulate_bfsk_coherent(r_t: np.ndarray, n_bits: int, tb: float = TB, f1: float = F1_FSK,
                             f2: float = F2_FSK, fs: float = FS) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Coherent Dual-Correlator Receiver for Binary FSK.
    Correlates received signal with phi1(t) and phi2(t).
    Decision statistic D = Z1 - Z2. Decide 1 if D > 0, else 0.
    """
    sps = int(fs * tb)
    dt = 1.0 / fs
    t_bit = np.arange(sps) / fs
    basis1 = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f1 * t_bit)
    basis2 = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f2 * t_bit)

    z1 = np.zeros(n_bits)
    z2 = np.zeros(n_bits)

    for i in range(n_bits):
        r_seg = r_t[i * sps:(i + 1) * sps]
        z1[i] = np.sum(r_seg * basis1) * dt
        z2[i] = np.sum(r_seg * basis2) * dt

    decision_stat = z1 - z2
    detected_bits = (decision_stat > 0.0).astype(int)

    return detected_bits, z1, z2, decision_stat


# =============================================================================
# 5. NON-COHERENT RECEIVERS (ENVELOPE / QUADRATURE DETECTION)
# =============================================================================

def demodulate_ask_noncoherent(r_t: np.ndarray, n_bits: int, tb: float = TB, fc: float = FC_ASK,
                               fs: float = FS, amp: float = AMP) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Non-coherent Quadrature Envelope Detector for Binary ASK.
    Computes R = sqrt(I^2 + Q^2) over each symbol period.
    """
    sps = int(fs * tb)
    dt = 1.0 / fs
    t_bit = np.arange(sps) / fs
    cos_ref = np.cos(2.0 * np.pi * fc * t_bit)
    sin_ref = np.sin(2.0 * np.pi * fc * t_bit)

    envelope_outputs = np.zeros(n_bits)
    for i in range(n_bits):
        r_seg = r_t[i * sps:(i + 1) * sps]
        i_comp = (2.0 / tb) * np.sum(r_seg * cos_ref) * dt
        q_comp = (2.0 / tb) * np.sum(r_seg * sin_ref) * dt
        envelope_outputs[i] = np.sqrt(i_comp ** 2 + q_comp ** 2)

    # Threshold chosen at half the peak envelope amplitude A
    threshold = amp / 2.0
    detected_bits = (envelope_outputs > threshold).astype(int)

    return detected_bits, envelope_outputs, threshold


def demodulate_bfsk_noncoherent(r_t: np.ndarray, n_bits: int, tb: float = TB, f1: float = F1_FSK,
                                f2: float = F2_FSK, fs: float = FS) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-coherent Dual Quadrature Envelope Detector for Binary FSK.
    Branch 1 Envelope: R1 = sqrt(I1^2 + Q1^2)
    Branch 2 Envelope: R2 = sqrt(I2^2 + Q2^2)
    Decide 1 if R1 > R2, else 0.
    """
    sps = int(fs * tb)
    dt = 1.0 / fs
    t_bit = np.arange(sps) / fs

    cos1 = np.cos(2.0 * np.pi * f1 * t_bit)
    sin1 = np.sin(2.0 * np.pi * f1 * t_bit)
    cos2 = np.cos(2.0 * np.pi * f2 * t_bit)
    sin2 = np.sin(2.0 * np.pi * f2 * t_bit)

    r1 = np.zeros(n_bits)
    r2 = np.zeros(n_bits)

    for i in range(n_bits):
        r_seg = r_t[i * sps:(i + 1) * sps]
        i1 = (2.0 / tb) * np.sum(r_seg * cos1) * dt
        q1 = (2.0 / tb) * np.sum(r_seg * sin1) * dt
        r1[i] = np.sqrt(i1 ** 2 + q1 ** 2)

        i2 = (2.0 / tb) * np.sum(r_seg * cos2) * dt
        q2 = (2.0 / tb) * np.sum(r_seg * sin2) * dt
        r2[i] = np.sqrt(i2 ** 2 + q2 ** 2)

    decision_diff = r1 - r2
    detected_bits = (decision_diff > 0.0).astype(int)

    return detected_bits, r1, r2, decision_diff


# =============================================================================
# 6. MANDATORY SYLLABUS VALIDATION: BFSK BASIS ORTHOGONALITY
# =============================================================================

def validate_bfsk_orthogonality(tb: float = TB, fs: float = FS, f1: float = F1_FSK) -> pd.DataFrame:
    r"""
    Mandatory Syllabus Validation Requirement:
    'Calculate inner products of the BFSK basis signals over one bit interval.'

    Theoretical Formula for Inner Product of basis signals:
      rho = (2 / Tb) * \int_{0}^{Tb} cos(2*pi*f1*t) * cos(2*pi*(f1 + Delta_f)*t + phi) dt
    """
    sps = int(fs * tb)
    # Closed interval [0, Tb] with trapezoidal integration
    t_bit = np.linspace(0.0, tb, sps + 1)

    phi1 = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f1 * t_bit)

    # Test distinct frequency separations
    test_cases = [
        ("Sub-Nyquist / Non-orthogonal", 250.0, 0.25),
        ("Minimum Coherent Orthogonal (1/(2Tb))", 500.0, 0.50),
        ("Intermediate Non-orthogonal", 750.0, 0.75),
        ("Minimum Non-coherent Orthogonal (1/Tb)", 1000.0, 1.00),
        ("Second Coherent Orthogonal (3/(2Tb))", 1500.0, 1.50),
        ("Our Lab Setting (Delta_f = 2 kHz = 2/Tb)", 2000.0, 2.00)
    ]

    records = []
    for desc, delta_f, norm_spacing in test_cases:
        f2 = f1 + delta_f
        # Case A: In-phase (phi = 0)
        phi2_inphase = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f2 * t_bit)
        inner_prod_inphase = float(np.trapezoid(phi1 * phi2_inphase, t_bit))

        # Case B: Quadrature phase (phi = pi / 2)
        phi2_quad = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f2 * t_bit + np.pi / 2.0)
        inner_prod_quad = float(np.trapezoid(phi1 * phi2_quad, t_bit))

        # Max cross-correlation magnitude
        rho_envelope = np.sqrt(inner_prod_inphase ** 2 + inner_prod_quad ** 2)

        is_coherent_ortho = np.isclose(inner_prod_inphase, 0.0, atol=1e-5)
        is_noncoherent_ortho = np.isclose(rho_envelope, 0.0, atol=1e-5)

        records.append({
            "Description": desc,
            "Delta_f (Hz)": delta_f,
            "Delta_f * Tb": norm_spacing,
            "Inner Product (phi=0)": inner_prod_inphase,
            "Inner Product (phi=pi/2)": inner_prod_quad,
            "Envelope Cross-Corr": rho_envelope,
            "Coherent Orthogonal": "YES" if is_coherent_ortho else "NO",
            "Non-Coherent Orthogonal": "YES" if is_noncoherent_ortho else "NO"
        })

    df = pd.DataFrame(records)
    return df


# =============================================================================
# 7. MONTE CARLO BIT ERROR RATE (BER) SIMULATION
# =============================================================================

def simulate_ber_sweep(eb_no_db_range: np.ndarray, n_bits: int = 50000) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo BER simulation for ASK and BFSK across Eb/N0.
    Simulates:
      1. Coherent ASK
      2. Non-coherent ASK
      3. Coherent BFSK
      4. Non-coherent BFSK
    """
    bits = generate_random_bits(n_bits, seed=123)

    ber_ask_coh = np.zeros(len(eb_no_db_range))
    ber_ask_noncoh = np.zeros(len(eb_no_db_range))
    ber_bfsk_coh = np.zeros(len(eb_no_db_range))
    ber_bfsk_noncoh = np.zeros(len(eb_no_db_range))

    # Fast vectorized simulation using discrete-time baseband equivalents
    # For ASK: Bit 1 has amplitude sqrt(2*Eb), Bit 0 has amplitude 0.
    # Average Eb = E_s1 / 2 = 1.0 (normalized)
    # For BFSK: Bit 1 has signal on branch 1, Bit 0 has signal on branch 2, each with energy Eb = 1.0.

    for idx, eb_no_db in enumerate(eb_no_db_range):
        eb_no_lin = 10.0 ** (eb_no_db / 10.0)
        n0 = 1.0 / eb_no_lin
        sigma = np.sqrt(n0 / 2.0)

        rng = np.random.default_rng(2000 + idx)

        # ----------------------------------------------------
        # 1. Coherent ASK:
        # Transmitted correlator metric: s1 = sqrt(2*Eb) for bit 1, 0 for bit 0.
        # Threshold gamma = sqrt(2*Eb) / 2 = sqrt(Eb/2).
        # Noise ~ N(0, sigma^2 = N0/2).
        # ----------------------------------------------------
        s_ask = np.where(bits == 1, np.sqrt(2.0), 0.0)
        noise_ask = rng.normal(0.0, sigma, size=n_bits)
        r_ask = s_ask + noise_ask
        detected_ask_coh = (r_ask > (np.sqrt(2.0) / 2.0)).astype(int)
        ber_ask_coh[idx] = np.mean(detected_ask_coh != bits)

        # ----------------------------------------------------
        # 2. Non-coherent ASK (Envelope Detection):
        # r_I = s + noise_I, r_Q = noise_Q
        # R = sqrt(r_I^2 + r_Q^2), threshold = sqrt(2*Eb) / 2
        # ----------------------------------------------------
        noise_ask_q = rng.normal(0.0, sigma, size=n_bits)
        r_ask_env = np.sqrt(r_ask ** 2 + noise_ask_q ** 2)
        detected_ask_noncoh = (r_ask_env > (np.sqrt(2.0) / 2.0)).astype(int)
        ber_ask_noncoh[idx] = np.mean(detected_ask_noncoh != bits)

        # ----------------------------------------------------
        # 3. Coherent BFSK:
        # Branch 1: sqrt(Eb) if bit 1, else 0.
        # Branch 2: sqrt(Eb) if bit 0, else 0.
        # Noise in each branch ~ N(0, N0/2) independent.
        # ----------------------------------------------------
        s_fsk_1 = np.where(bits == 1, 1.0, 0.0)
        s_fsk_2 = np.where(bits == 0, 1.0, 0.0)
        n_fsk_1 = rng.normal(0.0, sigma, size=n_bits)
        n_fsk_2 = rng.normal(0.0, sigma, size=n_bits)
        z1 = s_fsk_1 + n_fsk_1
        z2 = s_fsk_2 + n_fsk_2
        detected_bfsk_coh = (z1 > z2).astype(int)
        ber_bfsk_coh[idx] = np.mean(detected_bfsk_coh != bits)

        # ----------------------------------------------------
        # 4. Non-coherent BFSK:
        # Dual envelope detector with Rayleigh / Rician envelopes:
        # R1 = sqrt((s_1 + n_1I)^2 + n_1Q^2)
        # R2 = sqrt((s_2 + n_2I)^2 + n_2Q^2)
        # ----------------------------------------------------
        n_fsk_1q = rng.normal(0.0, sigma, size=n_bits)
        n_fsk_2q = rng.normal(0.0, sigma, size=n_bits)
        r1_env = np.sqrt(z1 ** 2 + n_fsk_1q ** 2)
        r2_env = np.sqrt(z2 ** 2 + n_fsk_2q ** 2)
        detected_bfsk_noncoh = (r1_env > r2_env).astype(int)
        ber_bfsk_noncoh[idx] = np.mean(detected_bfsk_noncoh != bits)

    return {
        "eb_no_db": eb_no_db_range,
        "ask_coh": ber_ask_coh,
        "ask_noncoh": ber_ask_noncoh,
        "bfsk_coh": ber_bfsk_coh,
        "bfsk_noncoh": ber_bfsk_noncoh
    }


def theoretical_ber_curves(eb_no_db_range: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Theoretical BER formulas from digital communications theory:
      1. Coherent ASK (OOK):
         P_b = Q(sqrt(Eb / N0)) = 0.5 * erfc(sqrt(Eb / (2 * N0)))  [for peak Eb_peak = 2*Eb_avg]
         P_b = Q(sqrt(Eb_avg / N0))
      2. Coherent BFSK:
         P_b = Q(sqrt(Eb / N0)) = 0.5 * erfc(sqrt(Eb / (2 * N0)))
      3. Non-coherent BFSK:
         P_b = 0.5 * exp(-Eb / (2 * N0))
      4. Non-coherent ASK (approximate high-SNR envelope bound):
         P_b approx 0.5 * exp(-Eb / (4 * N0))
    """
    eb_no_lin = 10.0 ** (eb_no_db_range / 10.0)

    # Q(x) = 0.5 * erfc(x / sqrt(2))
    # Coherent BFSK: Q(sqrt(Eb/N0))
    ber_bfsk_coh = 0.5 * sp.erfc(np.sqrt(eb_no_lin / 2.0))

    # Coherent ASK with equiprobable bits and optimal threshold:
    # d = sqrt(2*Eb), noise variance = N0/2 => SNR_eff = (d/2)^2 / (N0/2) = (Eb/2) / (N0/2) = Eb/N0
    # P_b = Q(sqrt(Eb / N0))
    ber_ask_coh = 0.5 * sp.erfc(np.sqrt(eb_no_lin / 2.0))

    # Non-coherent BFSK:
    ber_bfsk_noncoh = 0.5 * np.exp(-eb_no_lin / 2.0)

    # Non-coherent ASK:
    ber_ask_noncoh = 0.5 * np.exp(-eb_no_lin / 4.0)

    return {
        "eb_no_db": eb_no_db_range,
        "ask_coh": ber_ask_coh,
        "ask_noncoh": ber_ask_noncoh,
        "bfsk_coh": ber_bfsk_coh,
        "bfsk_noncoh": ber_bfsk_noncoh
    }


# =============================================================================
# 8. VISUALIZATION FUNCTIONS (SAVED TO OUTPUTS/)
# =============================================================================

def plot_passband_waveforms(n_bits: int = 10, filename: str = "1_ask_bfsk_passband_waveforms.png"):
    """Figure 1: Time-domain passband modulated waveforms for ASK and BFSK."""
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])[:n_bits]
    t, baseband, ask_signal = modulate_ask(bits, TB, FC_ASK, FS, AMP)
    _, _, bfsk_signal = modulate_bfsk(bits, TB, F1_FSK, F2_FSK, FS, AMP, phase_continuous=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    # (a) Bit Sequence / Baseband
    axes[0].step(t * 1000, baseband, where="post", color="navy", linewidth=2.0)
    axes[0].set_ylabel("Data Bits")
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].set_title(f"Binary Input Sequence: {list(bits)}")
    for i, b in enumerate(bits):
        axes[0].text((i + 0.5) * TB * 1000, 0.5, f"b={b}", ha="center", va="center",
                     fontsize=12, fontweight="bold", color="darkred")

    # (b) Unmodulated Carrier
    carrier = AMP * np.cos(2.0 * np.pi * FC_ASK * t)
    axes[1].plot(t * 1000, carrier, color="gray", linewidth=1.0)
    axes[1].set_ylabel("Carrier")
    axes[1].set_title(f"Unmodulated Carrier Reference ($f_c = {FC_ASK/1000:.0f}$ kHz)")

    # (c) Binary ASK (OOK)
    axes[2].plot(t * 1000, ask_signal, color="crimson", linewidth=1.2)
    axes[2].set_ylabel("ASK $s(t)$")
    axes[2].set_title("Binary ASK (OOK) Passband Waveform: $s_1(t) = A\\cos(2\\pi f_c t), s_0(t) = 0$")

    # (d) Binary FSK (CPFSK)
    axes[3].plot(t * 1000, bfsk_signal, color="forestgreen", linewidth=1.2)
    axes[3].set_ylabel("BFSK $s(t)$")
    axes[3].set_xlabel("Time (milliseconds)")
    axes[3].set_title(f"Binary FSK Waveform: Mark $f_1 = {F1_FSK/1000:.0f}$ kHz ('1'), Space $f_2 = {F2_FSK/1000:.0f}$ kHz ('0')")

    for ax in axes:
        ax.grid(True)
        for i in range(len(bits) + 1):
            ax.axvline(i * TB * 1000, color="black", linestyle=":", alpha=0.6)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_power_spectral_density(n_bits: int = 5000, filename: str = "2_power_spectral_density.png"):
    """Figure 2: Power Spectral Density of ASK and BFSK using Welch's method."""
    bits = generate_random_bits(n_bits, seed=555)
    t, _, ask_signal = modulate_ask(bits, TB, FC_ASK, FS, AMP)
    _, _, bfsk_signal = modulate_bfsk(bits, TB, F1_FSK, F2_FSK, FS, AMP, phase_continuous=True)

    nperseg = 4096
    f_ask, psd_ask = signal.welch(ask_signal, FS, nperseg=nperseg, window="hann")
    f_bfsk, psd_bfsk = signal.welch(bfsk_signal, FS, nperseg=nperseg, window="hann")

    # Convert to dB/Hz normalized to peak
    psd_ask_db = 10.0 * np.log10(psd_ask / np.max(psd_ask) + 1e-12)
    psd_bfsk_db = 10.0 * np.log10(psd_bfsk / np.max(psd_bfsk) + 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ASK PSD
    ax1.plot(f_ask / 1000.0, psd_ask_db, color="crimson", linewidth=1.5, label="BASK (OOK) PSD")
    ax1.axvline(FC_ASK / 1000.0, color="black", linestyle="--", label=f"Carrier $f_c = {FC_ASK/1000:.0f}$ kHz")
    ax1.axvline((FC_ASK - RB) / 1000.0, color="gray", linestyle=":", label="Nulls ($f_c \\pm R_b$)")
    ax1.axvline((FC_ASK + RB) / 1000.0, color="gray", linestyle=":")
    ax1.set_ylabel("Normalized PSD (dB/Hz)")
    ax1.set_title("Power Spectral Density of Binary Amplitude Shift Keying (BASK)")
    ax1.set_ylim(-60, 5)
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # BFSK PSD
    ax2.plot(f_bfsk / 1000.0, psd_bfsk_db, color="forestgreen", linewidth=1.5, label="BFSK PSD")
    ax2.axvline(F1_FSK / 1000.0, color="blue", linestyle="--", label=f"Mark $f_1 = {F1_FSK/1000:.0f}$ kHz")
    ax2.axvline(F2_FSK / 1000.0, color="purple", linestyle="--", label=f"Space $f_2 = {F2_FSK/1000:.0f}$ kHz")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Normalized PSD (dB/Hz)")
    ax2.set_title("Power Spectral Density of Binary Frequency Shift Keying (BFSK)")
    ax2.set_xlim(5, 18)
    ax2.set_ylim(-60, 5)
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_correlator_outputs(n_bits: int = 15, snr_db: float = 12.0, filename: str = "3_correlator_outputs_and_decision.png"):
    """Figure 3: Time-domain correlator outputs and sampling decision instants."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0])[:n_bits]
    t, baseband, ask_clean = modulate_ask(bits, TB, FC_ASK, FS, AMP)
    _, _, bfsk_clean = modulate_bfsk(bits, TB, F1_FSK, F2_FSK, FS, AMP)

    ask_noisy, _ = add_awgn(ask_clean, snr_db, seed=77)
    bfsk_noisy, _ = add_awgn(bfsk_clean, snr_db, seed=78)

    # ASK Demodulation
    detected_ask, ask_corr, ask_thresh = demodulate_ask_coherent(ask_noisy, n_bits, TB, FC_ASK, FS, AMP)
    # BFSK Demodulation
    detected_bfsk, z1, z2, decision_stat = demodulate_bfsk_coherent(bfsk_noisy, n_bits, TB, F1_FSK, F2_FSK, FS)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    bit_indices = np.arange(n_bits) + 0.5

    # (a) Original bits
    axes[0].step(np.arange(n_bits), bits, where="post", color="navy", linewidth=2.0, label="Transmitted Bits")
    axes[0].set_ylabel("Data Bit")
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].set_title(f"Transmitted Binary Sequence (SNR = {snr_db} dB)")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # (b) ASK Correlator Output
    axes[1].stem(bit_indices, ask_corr, linefmt="crimson", markerfmt="ro", basefmt="k-", label="Correlator Output $Z$")
    axes[1].axhline(ask_thresh, color="black", linestyle="--", linewidth=1.8, label=f"Threshold $\\gamma = {ask_thresh:.4f}$")
    axes[1].set_ylabel("ASK Correlator")
    axes[1].set_title("Coherent ASK Correlator Decision Statistic")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # (c) BFSK Differential Correlator Output
    axes[2].stem(bit_indices, decision_stat, linefmt="forestgreen", markerfmt="go", basefmt="k-", label="Differential Metric $D = Z_1 - Z_2$")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.8, label="Zero Decision Threshold")
    axes[2].set_ylabel("BFSK Decision Metric")
    axes[2].set_xlabel("Bit Index")
    axes[2].set_title("Coherent BFSK Dual-Correlator Difference ($Z_1 - Z_2$)")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_decision_histograms(n_bits: int = 5000, snr_db: float = 8.0, filename: str = "4_decision_statistic_histograms.png"):
    """Figure 4: Decision statistic distributions under AWGN for bits 0 and 1."""
    bits = generate_random_bits(n_bits, seed=888)
    _, _, ask_clean = modulate_ask(bits, TB, FC_ASK, FS, AMP)
    _, _, bfsk_clean = modulate_bfsk(bits, TB, F1_FSK, F2_FSK, FS, AMP)

    ask_noisy, _ = add_awgn(ask_clean, snr_db, seed=991)
    bfsk_noisy, _ = add_awgn(bfsk_clean, snr_db, seed=992)

    _, ask_corr, ask_thresh = demodulate_ask_coherent(ask_noisy, n_bits, TB, FC_ASK, FS, AMP)
    _, _, _, bfsk_stat = demodulate_bfsk_coherent(bfsk_noisy, n_bits, TB, F1_FSK, F2_FSK, FS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ASK Histogram
    ax1.hist(ask_corr[bits == 0], bins=50, density=True, alpha=0.6, color="blue", label="Bit '0' ($s_0=0$)")
    ax1.hist(ask_corr[bits == 1], bins=50, density=True, alpha=0.6, color="red", label="Bit '1' ($s_1=A\\cos(\\omega_c t)$)")
    ax1.axvline(ask_thresh, color="black", linestyle="--", linewidth=2.0, label=f"Optimum Threshold $\\gamma = {ask_thresh:.4f}$")
    ax1.set_title(f"Coherent ASK Correlator PDF ($E_b/N_0 = {snr_db}$ dB)")
    ax1.set_xlabel("Correlator Output $Z$")
    ax1.set_ylabel("Probability Density")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # BFSK Histogram
    ax2.hist(bfsk_stat[bits == 0], bins=50, density=True, alpha=0.6, color="purple", label="Bit '0' ($f_2=12$ kHz)")
    ax2.hist(bfsk_stat[bits == 1], bins=50, density=True, alpha=0.6, color="green", label="Bit '1' ($f_1=10$ kHz)")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=2.0, label="Optimum Threshold $\\gamma = 0$")
    ax2.set_title(f"Coherent BFSK Differential Metric PDF ($E_b/N_0 = {snr_db}$ dB)")
    ax2.set_xlabel("Decision Metric $D = Z_1 - Z_2$")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_tone_spacing_orthogonality(filename: str = "5_bfsk_tone_spacing_orthogonality.png"):
    """Figure 5: Tone spacing condition for BFSK orthogonality (coherent vs non-coherent)."""
    tb = TB
    fs = 200000.0  # High sampling rate for fine resolution
    sps = int(fs * tb)
    dt = 1.0 / fs
    t_bit = np.arange(sps) * dt
    f1 = 10000.0

    delta_f_array = np.linspace(0.0, 3000.0, 600)  # 0 to 3 * Rb
    norm_spacing = delta_f_array * tb

    rho_inphase = np.zeros_like(delta_f_array)
    rho_quad = np.zeros_like(delta_f_array)
    rho_envelope = np.zeros_like(delta_f_array)

    phi1 = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f1 * t_bit)

    for i, df in enumerate(delta_f_array):
        f2 = f1 + df
        # In-phase basis
        phi2_i = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f2 * t_bit)
        # Quadrature basis
        phi2_q = np.sqrt(2.0 / tb) * np.cos(2.0 * np.pi * f2 * t_bit + np.pi / 2.0)

        rho_i = np.sum(phi1 * phi2_i) * dt
        rho_q = np.sum(phi1 * phi2_q) * dt

        rho_inphase[i] = rho_i
        rho_quad[i] = rho_q
        rho_envelope[i] = np.sqrt(rho_i ** 2 + rho_q ** 2)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(norm_spacing, rho_inphase, color="blue", linewidth=2.0, label="Coherent In-Phase: $\\rho(\\Delta f, \\phi=0)$")
    ax.plot(norm_spacing, rho_quad, color="orange", linestyle="--", linewidth=1.5, label="Coherent Quadrature: $\\rho(\\Delta f, \\phi=\\pi/2)$")
    ax.plot(norm_spacing, rho_envelope, color="red", linewidth=2.2, label="Non-Coherent Envelope: $\\sqrt{\\rho_I^2 + \\rho_Q^2}$")

    # Annotate critical nulls
    ax.axvline(0.5, color="green", linestyle=":", linewidth=1.8, label="Coherent Min Spacing $\\Delta f = 1/(2T_b)$ (500 Hz)")
    ax.axvline(1.0, color="purple", linestyle=":", linewidth=1.8, label="Non-Coherent Min Spacing $\\Delta f = 1/T_b$ (1000 Hz)")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8)

    ax.set_title("BFSK Cross-Correlation vs Normalized Tone Spacing (Orthogonality Condition)", fontsize=13)
    ax.set_xlabel("Normalized Frequency Separation $\\Delta f \\cdot T_b = \\Delta f / R_b$", fontsize=12)
    ax.set_ylabel("Basis Cross-Correlation Coefficient $\\rho$", fontsize=12)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.35, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_ber_performance(filename: str = "6_ber_performance_comparison.png"):
    """Figure 6: Publication-quality semi-log Bit Error Rate (BER) curves."""
    eb_no_db = np.linspace(0.0, 14.0, 15)

    print("Running Monte Carlo BER simulation (100,000 bits per point)...")
    sim_results = simulate_ber_sweep(eb_no_db, n_bits=100000)
    theory_results = theoretical_ber_curves(np.linspace(0.0, 14.0, 100))

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Coherent ASK
    ax.semilogy(theory_results["eb_no_db"], theory_results["ask_coh"], "r-", linewidth=1.8, label="Coherent ASK (Theory)")
    ax.semilogy(sim_results["eb_no_db"], sim_results["ask_coh"], "ro", markersize=6, fillstyle="none", markeredgewidth=1.8, label="Coherent ASK (Sim)")

    # Coherent BFSK
    ax.semilogy(theory_results["eb_no_db"], theory_results["bfsk_coh"], "b-", linewidth=1.8, label="Coherent BFSK (Theory)")
    ax.semilogy(sim_results["eb_no_db"], sim_results["bfsk_coh"], "bs", markersize=6, fillstyle="none", markeredgewidth=1.8, label="Coherent BFSK (Sim)")

    # Non-coherent BFSK
    ax.semilogy(theory_results["eb_no_db"], theory_results["bfsk_noncoh"], "g-", linewidth=1.8, label="Non-Coherent BFSK (Theory)")
    ax.semilogy(sim_results["eb_no_db"], sim_results["bfsk_noncoh"], "g^", markersize=6, fillstyle="none", markeredgewidth=1.8, label="Non-Coherent BFSK (Sim)")

    # Non-coherent ASK
    ax.semilogy(theory_results["eb_no_db"], theory_results["ask_noncoh"], "m--", linewidth=1.5, label="Non-Coherent ASK (Bound)")
    ax.semilogy(sim_results["eb_no_db"], sim_results["ask_noncoh"], "md", markersize=6, fillstyle="none", markeredgewidth=1.5, label="Non-Coherent ASK (Sim)")

    ax.set_title("Theoretical vs Monte Carlo Simulated BER for ASK and BFSK over AWGN", fontsize=13)
    ax.set_xlabel("Energy per Bit to Noise Spectral Density $E_b / N_0$ (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_xlim(0, 14)
    ax.set_ylim(1e-5, 0.6)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_noncoherent_envelope_detector(n_bits: int = 10, filename: str = "7_noncoherent_envelope_detection.png"):
    """Figure 7: Non-coherent envelope detection waveform extraction."""
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])[:n_bits]
    t, baseband, ask_clean = modulate_ask(bits, TB, FC_ASK, FS, AMP)
    ask_noisy, _ = add_awgn(ask_clean, snr_db=14.0, seed=123)

    # Compute analytical Hilbert envelope
    analytic_signal = signal.hilbert(ask_noisy)
    envelope = np.abs(analytic_signal)

    # Low-pass filter the envelope
    b_lpf, a_lpf = signal.butter(4, 2.0 * RB / FS, btype="low")
    smooth_envelope = signal.filtfilt(b_lpf, a_lpf, envelope)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    # (a) Input noisy passband
    axes[0].plot(t * 1000, ask_noisy, color="crimson", linewidth=1.0)
    axes[0].set_ylabel("Noisy ASK (14 dB)")
    axes[0].set_title("Received Passband ASK Waveform with Additive White Gaussian Noise")
    axes[0].grid(True)

    # (b) Extracted Envelope
    axes[1].plot(t * 1000, envelope, color="gray", alpha=0.5, label="Raw Rectified Envelope")
    axes[1].plot(t * 1000, smooth_envelope, color="darkorange", linewidth=2.0, label="Post-LPF Smoothed Envelope")
    axes[1].axhline(AMP / 2.0, color="black", linestyle="--", linewidth=1.5, label=f"Detection Threshold ({AMP/2.0} V)")
    axes[1].set_ylabel("Envelope (V)")
    axes[1].set_title("Non-Coherent Envelope Extraction via Diode/LPF Analog")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # (c) Reconstructed Data Bits
    reconstructed_bits = (smooth_envelope > (AMP / 2.0)).astype(int)
    axes[2].step(t * 1000, baseband, where="post", color="navy", linewidth=2.0, label="Transmitted Bits")
    axes[2].step(t * 1000, reconstructed_bits, where="post", color="red", linestyle="--", linewidth=1.5, label="Decided Bits")
    axes[2].set_ylabel("Data Bits")
    axes[2].set_xlabel("Time (milliseconds)")
    axes[2].set_title("Transmitted vs Non-Coherently Recovered Digital Bit Stream")
    axes[2].set_ylim(-0.2, 1.2)
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    for ax in axes:
        for i in range(len(bits) + 1):
            ax.axvline(i * TB * 1000, color="black", linestyle=":", alpha=0.5)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_mandatory_validation(df_val: pd.DataFrame, filename: str = "8_mandatory_orthogonality_validation.png"):
    """Figure 8: Graphical display of the mandatory basis inner product validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart of inner products at key spacings
    labels = [f"$\\Delta f={r['Delta_f (Hz)']:.0f}$Hz\n({r['Delta_f * Tb']:.2f}$R_b$)" for _, r in df_val.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, df_val["Inner Product (phi=0)"], width, label="In-Phase $(\\phi=0)$", color="royalblue")
    ax1.bar(x + width/2, df_val["Envelope Cross-Corr"], width, label="Envelope $(\\max_\\phi |\\rho|)$", color="crimson")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Inner Product $\\langle \\phi_1, \\phi_2 \\rangle$")
    ax1.set_title("Mandatory Validation: Basis Inner Products over $[0, T_b]$", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Table subplot
    ax2.axis("off")
    table_data = []
    for _, r in df_val.iterrows():
        table_data.append([
            f"{r['Delta_f (Hz)']:.0f} Hz",
            f"{r['Delta_f * Tb']:.2f}",
            f"{r['Inner Product (phi=0)']:+.4f}",
            f"{r['Envelope Cross-Corr']:.4f}",
            r['Coherent Orthogonal'],
            r['Non-Coherent Orthogonal']
        ])
    col_labels = ["Delta_f", "Norm Spacing", "Inner Prod (phi=0)", "Envelope", "Coh Ortho?", "Non-Coh Ortho?"]
    table = ax2.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    ax2.set_title("Numerical Inner Product Verification Summary", fontsize=12, pad=15)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


# =============================================================================
# 9. MAIN EXECUTION DRIVER
# =============================================================================

def main():
    print("=" * 80)
    print("Digital Communication Lab - Experiment 9: ASK and FSK Simulation")
    print("=" * 80)

    # 1. Mandatory Syllabus Validation: Inner product calculation
    print("\n--- 1. Mandatory Syllabus Validation: BFSK Basis Orthogonality ---")
    df_val = validate_bfsk_orthogonality(TB, FS, F1_FSK)
    print(df_val.to_string(index=False))

    # 2. Generate all publication figures
    print("\n--- 2. Generating Publication Visualizations (outputs/) ---")
    plot_passband_waveforms(n_bits=10)
    plot_power_spectral_density(n_bits=5000)
    plot_correlator_outputs(n_bits=15, snr_db=12.0)
    plot_decision_histograms(n_bits=5000, snr_db=8.0)
    plot_tone_spacing_orthogonality()
    plot_noncoherent_envelope_detector(n_bits=10)
    plot_mandatory_validation(df_val)
    plot_ber_performance()

    print("\n" + "=" * 80)
    print("Experiment 9 simulation completed successfully!")
    print(f"All outputs generated in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
