# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_dualrail_v6_tunnel_EN_analyzer.py
# Purpose: Analyzes the Quantum Tunneling CSV and reproduces the SPHY graph.
# Author: Gemini AI
# ───────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d

# Activates Matplotlib interactive mode
plt.ion() 

def input_csv_path():
    """Prompts for the CSV file path and verifies its existence."""
    while True:
        # Requests the path, removing quotes and spaces
        path = input("\nEnter the full path of the Tunneling Simulation CSV: ").strip().strip('"').strip("'")
        if os.path.exists(path) and path.lower().endswith('.csv'):
            print(f"File found: {path}")
            return path
        print("File not found or invalid. Please try again.")

def analyze_and_plot(filepath):
    """Loads data, calculates metrics, and generates the 3-panel graph."""
    print("Loading data...")
    
    # Reads only the essential columns
    try:
        df = pd.read_csv(
            filepath,
            usecols=['Frame', 'SPHY (%)', 'Accepted'],
            dtype={'Frame': 'int32', 'SPHY (%)': 'float32', 'Accepted': 'category'},
            engine='c'
        )
    except KeyError:
        print("❌ Error: The CSV does not contain the expected columns: 'Frame', 'SPHY (%)', and 'Accepted'.")
        return None

    sphy_np = df['SPHY (%)'].values
    total_frames = len(sphy_np)
    
    if total_frames == 0:
        print("❌ The CSV file is empty.")
        return None

    print("Calculating metrics and reconstructing the SPHY curve...")

    # === SPHY Curve Reconstruction (Must replicate the generator logic) ===
    x_vals = np.linspace(0, 1, len(sphy_np))
    signals = [interp1d(x_vals, np.roll(sphy_np, i), kind='cubic') for i in range(2)]
    new_x = np.linspace(0, 1, 2000)
    
    # NOTE: Random noise will NOT be reproduced here, only the main average curve.
    # To be faithful to the look of the generation plot, we will use the interpolated outputs.
    signal_outputs = [sig(new_x) for sig in signals] 
    
    weights = np.linspace(1, 1.5, len(signal_outputs))
    final_curve = np.average(signal_outputs, axis=0, weights=weights)

    # === Stability Metrics Calculation ===
    stability_mean = final_curve.mean()
    stability_var = final_curve.var()
    coherence_gain = final_curve[-1] - 90.0 # Assumed initial point

    stats = {
        "Total Frames": total_frames,
        "Mean Stability Index": stability_mean,
        "Stability Variance Index": stability_var,
        "Total Coherence Gain (%)": coherence_gain
    }
    
    # === Graph Generation (3 Panels) ===
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    # Ax1: SPHY Evolution - Quantum Tunnel
    ax1.plot(new_x, final_curve, 'k--', lw=2, label="Stable SPHY (Reconstructed)")
    for i, sig in enumerate(signal_outputs):
        ax1.plot(new_x, sig, alpha=0.3, color='tab:blue' if i == 0 else 'tab:red', 
                 label=f"Signal {i+1} (Interp.)" if total_frames < 2000 else None)
    ax1.set_title("SPHY Evolution - Quantum Tunneling (Analysis)")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Phase Stability")
    ax1.legend()
    ax1.grid(alpha=0.5)

    # Ax2: Phase Variation and Stability
    ax2.plot(new_x, final_curve, 'k-', lw=1.5, label="Mean Stability")
    ax2.axhline(stability_mean, color='green', linestyle='--', label=f"Mean: {stability_mean:.6f}")
    ax2.axhline(stability_mean + np.sqrt(stability_var), color='orange', linestyle='--', label="± Variance (Standard Deviation)")
    ax2.axhline(stability_mean - np.sqrt(stability_var), color='orange', linestyle='--')
    ax2.set_title("Phase Variation and Stability (SPHY Control)")
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Stable Amplitude")
    ax2.legend()
    ax2.grid(alpha=0.5)

    # Ax3: Coherence Evolution per Frame (Raw Data) and Histogram
    frames_idx = df['Frame'].values
    ax3.plot(frames_idx, sphy_np, '-', ms=3, lw=1, color='tab:purple', label='Raw Coherence (frames)')
    ax3.set_title('Coherence Evolution per Frame')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('SPHY (%)')
    ax3.grid(alpha=0.4)

    # Inserts a small histogram as an inset in ax3
    left, bottom, width, height = 0.70, 0.08, 0.20, 0.18
    ax_hist = fig.add_axes([left, bottom, width, height])
    ax_hist.hist(sphy_np, bins=20, color='gray', alpha=0.8)
    ax_hist.set_title('SPHY Histogram', fontsize=9)
    ax_hist.tick_params(labelsize=8)


    fig.suptitle(f"SPHY Quantum Tunneling Benchmark Analysis ({total_frames} Frames)", fontsize=16)
    fig.subplots_adjust(top=0.92, hspace=0.45)
    
    # Saves the graph
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = filepath.replace(".csv", f"_ANALYZED_PLOT_{now}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graph saved: {plot_path}")

    return stats, plot_path, fig

def print_report(stats, plot_path):
    """Prints the metrics report to the screen."""
    print("\n" + "═" * 70)
    print(" " * 15 + "SPHY BENCHMARK REPORT - QUANTUM TUNNELING")
    print("═" * 70)
    print(f"Total Frames Analyzed: {stats['Total Frames']:,}")
    print("-" * 70)
    print("SPHY Stability Metrics (Reconstructed Curve):")
    print(f"   Mean Stability Index (Mean): {stats['Mean Stability Index']:.6f}")
    print(f"   Stability Variance Index (Variance): {stats['Stability Variance Index']:.6f}")
    print(f"   Total Coherence Gain (Net Gain): {stats['Total Coherence Gain (%)']:+.4f}%")
    print("-" * 70)
    print(f"Analysis Graph (3 Panels): {os.path.basename(plot_path)}")
    print("═" * 70)

# === Main Execution

if __name__ == "__main__":
    print("Starting Benchmark Analysis with Interactive Viewer")
    csv_path = input_csv_path()
    
    # 1. Analyzes and generates the graph
    result = analyze_and_plot(csv_path)
    
    if result:
        stats, plot_path, fig_handle = result
        
        # 2. Prints the report
        print_report(stats, plot_path)
        
        # 3. Displays the interactive viewer
        print("\nOpening Matplotlib interactive viewer (zoom, pan, save)...")
        plt.show(block=True) 
        print("\nWindow closed. Program terminated.")
    else:
        print("Analysis aborted due to data error.")