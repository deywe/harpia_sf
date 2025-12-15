# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_strawberryfields_toroidal_3d_v9_EN_analyzer.py
# Purpose: BENCHMARK ANALYZER AND PLOTTER (WIGNER, HISTOGRAM, STABILITY) FROM CSV.
# Author: Gemini AI
# ───────────────────────────────────────────────────────────────

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
import os
import sys

# === Configuration and Variables (Must match the simulator) ===
TARGET_MODE = 0 
TUNNELING_THRESHOLD = 0.05 
LOG_DIR = "analysed_sphy_v9"
os.makedirs(LOG_DIR, exist_ok=True)

# Required CSV column definitions
COV_HEADERS = ["Vqq_0", "Vqp_0", "Vpq_0", "Vpp_0"]
MEANS_HEADERS = ["mu_q_0", "mu_p_0"]
PROXY_MAG_COL = "Proxy_Mag"
SPHY_COHERENCE_COL = "SPHY (%)"
ACCEPTED_COL = "Accepted"
FRAME_COL = "Frame"

# === 1. Wigner Function Plotting (Final State) ===

def plot_wigner_function(cov_target, means_target, fig_filename_wigner, total_frames):
    """Generates the Wigner Function (CV State Visualization) for the last frame's state."""
    if cov_target is None or means_target is None or not any(cov_target):
        print("❌ Error: Quantum state data (Wigner) not available in CSV.")
        return

    # Reconstruct the 2x2 Covariance Matrix
    cov = np.array([[cov_target[0], cov_target[1]], [cov_target[2], cov_target[3]]])
    # Displacement Vector [mu_q, mu_p]
    means = np.array([means_target[0], means_target[1]])
    
    q_lim = max(3.0, np.max(np.abs(means))) + 1.0 
    q_grid = np.linspace(-q_lim, q_lim, 100)
    Q, P = np.meshgrid(q_grid, q_grid)
    coords = np.vstack([Q.flatten(), P.flatten()]).T
    
    try:
        # The Wigner Function for the Gaussian state is modeled by the multivariate PDF.
        wigner_pdf = multivariate_normal.pdf(coords, mean=means, cov=cov)
    except np.linalg.LinAlgError:
        print("⚠️ Linear Algebra Error in Wigner. Singular covariance matrix.")
        return
        
    W = wigner_pdf.reshape(Q.shape)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(Q, P, W, 100, cmap='RdBu_r')
    ax.scatter(means[0], means[1], marker='x', color='black', s=100, label='Center ($\mu_q, \mu_p$)')
    
    ax.set_title(f'Wigner Function of the Target Mode (Final Frame: {total_frames})', fontsize=14)
    ax.set_xlabel('Position Quadrature ($q$)')
    ax.set_ylabel('Momentum Quadrature ($p$)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(contour, label='W(q, p) Amplitude')
    
    plt.savefig(fig_filename_wigner, dpi=300)
    plt.show(block=False) # Display the plot without blocking the rest of the script
    print(f"🖼️ Wigner Function saved: {fig_filename_wigner}")


# === 2. Performance Histogram Plotting ===

def plot_tunneling_histogram(df, threshold, fig_filename_hist, total_frames):
    """Generates the Histogram of the Tunneling Proxy Magnitude (Proxy_Mag)."""
    if df.empty: return

    proxy_data = df[PROXY_MAG_COL].astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(proxy_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue', 
            label='Tunneling Proxy Magnitude ( |$\\Delta \\bar{n}$| )')
    
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Tunneling Threshold ({threshold})')
    
    success_count = (proxy_data >= threshold).sum()
    success_rate = 100 * (success_count / total_frames)
    
    ax.text(0.95, 0.90, f'Total Success: {success_rate:.2f}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
            horizontalalignment='right', fontsize=12, color='darkgreen', weight='bold')

    ax.set_title(f'Performance Distribution over {total_frames} Frames', fontsize=14)
    ax.set_xlabel('Tunneling Proxy Magnitude ( |$\\Delta \\bar{n}$| )')
    ax.set_ylabel('Frequency of Occurrence (Frames)')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    plt.savefig(fig_filename_hist, dpi=300)
    plt.show(block=False)
    print(f"🖼️ Tunneling Histogram saved: {fig_filename_hist}")


# === 3. SPHY Stability Evolution Plotting (2D) ===

def plot_sphy_evolution(df, fig_filename):
    """Generates the 2D SPHY stability graph over time (based on SPHY Coherence)."""
    sphy_evolution_list = df[SPHY_COHERENCE_COL].astype(float).tolist()
    if not sphy_evolution_list: return

    sphy_evolution = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution))
    
    # Reproduces the interpolation and redundancy logic from the simulator
    n_redundancies = 2 
    signals = [interp1d(time_points, np.roll(sphy_evolution, i), kind='cubic') for i in range(n_redundancies)]
    new_time = np.linspace(0, 1, 2000)
    data = [sinal(new_time) + np.random.normal(0, 0.15, len(new_time)) for sinal in signals]
    weights = np.linspace(1, 1.5, n_redundancies)
    tunneling_stability = np.average(data, axis=0, weights=weights)

    stability_mean_2 = np.mean(data[1]) 
    stability_variance_2 = np.var(data[1])

    total_frames = len(sphy_evolution_list)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Graph 1: SPHY Coherence Signal (Amplitude)
    ax1.set_title("SPHY Coherence Evolution (Signal 1: Amplitude)")
    for i in range(n_redundancies):
        ax1.plot(new_time, data[i], alpha=0.3, color='blue')  
    ax1.plot(new_time, tunneling_stability, 'k--', linewidth=2, label="Weighted Average Stability")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.legend()
    ax1.grid()

    # Graph 2: SPHY Coherence Signal (Stability)
    ax2.set_title("SPHY Coherence Evolution (Signal 2: Stability)")
    ax2.plot(new_time, data[1], color='red', alpha=0.7, label='Coherence Signal (2)')
    
    ax2.axhline(stability_mean_2, color='green', linestyle='--', label=f"Mean: {stability_mean_2:.2f}")
    ax2.axhline(stability_mean_2 + np.sqrt(stability_variance_2), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(stability_mean_2 - np.sqrt(stability_variance_2), color='orange', linestyle='--')

    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Quantum Tunneling Analysis (SF CV): {total_frames} Frames (SPHY Stability)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)
    plt.show(block=False)
    print(f"🖼️ 2D Stability Graph saved: {fig_filename}")

# === Main Analysis Function ===

def run_analysis(csv_filepath):
    """Loads the CSV, calculates metrics, and generates the plots."""
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found: {csv_filepath}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print("=" * 60)
    print(f" 🔍 Starting SPHY-CV Analysis for: {os.path.basename(csv_filepath)}")
    print("=" * 60)
    
    # 1. METRICS CALCULATION (Reproducing simulator metrics)
    total_frames = len(df)
    
    # Success Rate (Tunneling)
    accepted_frames = df[df[ACCEPTED_COL] == '✅']
    success_rate = 100 * (len(accepted_frames) / total_frames)
    
    # SPHY Stability
    sphy_data = df[SPHY_COHERENCE_COL].astype(float)
    mean_stability = sphy_data.mean()
    stability_variance = sphy_data.var()
    
    # Quantum Metrics (Last Frame)
    cov_target_flat = df.iloc[-1][COV_HEADERS].values.astype(float)
    purity, squeezing_min, wigner_max = float('nan'), float('nan'), float('nan')
    
    try:
        V = np.array([[cov_target_flat[0], cov_target_flat[1]], [cov_target_flat[2], cov_target_flat[3]]])
        det_2V = np.linalg.det(2 * V)
        purity = 1.0 / np.sqrt(det_2V)
        
        trace_V = np.trace(V)
        det_V = np.linalg.det(V)
        squeezing_min = 0.5 * (trace_V - np.sqrt(trace_V**2 - 4 * det_V))
        
        wigner_max = purity / np.pi
    except Exception:
        pass # Maintains NaN if calculation fails
        

    # 2. PRINTING THE METRICS REPORT
    
    print("      📊 SPHY-CV PERFORMANCE REPORT")
    print("-" * 60)
    print(f"| Total Analyzed Frames: {total_frames:,}")
    print(f"| Success Rate (Tunnel Accepted): {len(accepted_frames)}/{total_frames} | **{success_rate:.2f}%**")
    print("-" * 60)
    print(f"| ⭐ Mean SPHY Stability: {mean_stability:.4f}")
    print(f"| 🌊 Stability Variance: {stability_variance:.6f}")
    print("-" * 60)
    print(f"| ⚛️ Final Purity (μ): {purity:.4f}")
    print(f"| 🔬 Min Squeezing (λ_min): {squeezing_min:.4f}")
    print(f"| 📈 Max Wigner (W_max): {wigner_max:.4f}")
    print("=" * 60)
    
    
    # 3. GENERATING FILENAMES
    base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
    fig_filename_wigner = os.path.join(LOG_DIR, f"{base_name}_WIGNER_ANALYSIS.png")
    fig_filename_hist = os.path.join(LOG_DIR, f"{base_name}_HISTOGRAM_ANALYSIS.png")
    fig_filename_stability = os.path.join(LOG_DIR, f"{base_name}_STABILITY_ANALYSIS.png")


    # 4. PLOTTING AND DISPLAY
    
    # A. WIGNER FUNCTION
    means_target = df.iloc[-1][MEANS_HEADERS].values.astype(float)
    plot_wigner_function(cov_target_flat, means_target, fig_filename_wigner, total_frames)
    
    # B. TUNNELING HISTOGRAM
    plot_tunneling_histogram(df, TUNNELING_THRESHOLD, fig_filename_hist, total_frames)
    
    # C. SPHY STABILITY EVOLUTION
    plot_sphy_evolution(df, fig_filename_stability)

    # Blocks execution ONLY at the end to keep the Matplotlib windows open
    print("\nVisualizing plots... Close Matplotlib windows to finalize.")
    plt.show(block=True) 


if __name__ == "__main__":
    
    csv_file = None
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if csv_file is None:
        print("\n--- SPHY CSV ANALYSER ---")
        csv_file = input("Please, enter the full path or name of the CSV log file: ")
        
    if not csv_file:
        print("❌ Operation cancelled. No file path provided.")
        sys.exit(1)
        
    run_analysis(csv_file)