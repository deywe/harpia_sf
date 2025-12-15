# file sphy_sf_qaoa_v02.3_PLOTS_analyzer_spatial3d_eng.py
# ...existing code...
# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_qaoa_v02.3_PLOTS_analyzer_spatial3d_eng.py
# Purpose: Benchmark Generator (QBench SPHY QAOA w/ P) for Tunneling Logs
# Author: Gemini & deywe@QLZ — Patched Version with Spatial 3D Plot
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re
from mpl_toolkits.mplot3d import Axes3D # Required for the 3D plot

# Define the log directory
LOG_DIR = "logs_harpia_sphy_cirq"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === MAIN ANALYSIS FUNCTION ===
# ====================================================================

def analyze_qaoa_tunnel_log():
    """
    Prompts for the CSV file path (LOG_CSV), loads the data, and generates the benchmark report,
    focused on comparing entropy modes 1 and 2. Now includes a 3D Spatial Plot visualizing 
    P = [Px,Py,Pz] and Pdot = [Vx,Vy,Vz] extracted from psi0_noise.
    """
    # 1. Prompt for file path
    default_path_pattern = os.path.join(LOG_DIR, "harpia_tunnel_cirq_batch_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH QAOA TUNNELING ANALYZER: Entropic Modes Comparison ".center(80))
    print("="*80)
    
    file_path = input(f"📁 Enter the full path of the LOG_CSV (e.g., {default_path_pattern}):\n>> ").strip()
    
    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found at: {file_path}")
        sys.exit(1)
        
    # Extract the MODE from the filename for the title
    mode_match = re.search(r"MODE(\d+)", os.path.basename(file_path))
    mode_info = f"MODE {mode_match.group(1)}" if mode_match else "Unknown Mode"

    # 2. Load the Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Error loading the CSV: {e}")
        sys.exit(1)

    # Ensure essential columns exist and perform cleanup
    required_cols = ['status', 'energy', 'f_opt', 'time', 'psi0_noise']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Error: Missing essential columns. Required: {required_cols}")
        sys.exit(1)

    df['f_opt'] = pd.to_numeric(df['f_opt'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['f_opt', 'energy', 'time'], inplace=True)
    
    # === 3. BENCHMARK METRICS CALCULATION ===
    
    total_rounds = len(df)
    accepted_df = df[df['status'].str.contains('Accepted', na=False, case=False)].copy()
    accepted_count = len(accepted_df)
    success_rate = (accepted_count / total_rounds) * 100 if total_rounds > 0 else 0.0

    # 3.1. Optimization Quality (QAOA Energy)
    mean_energy = accepted_df['energy'].mean() if accepted_count > 0 else 0.0
    std_energy = accepted_df['energy'].std() if accepted_count > 1 else 0.0
    
    # 3.2. SPHY/Tunneling Control (f_opt)
    mean_f_opt = accepted_df['f_opt'].mean() if accepted_count > 0 else 0.0
    variance_f_opt = accepted_df['f_opt'].var() if accepted_count > 1 else 0.0
    
    # 3.3. Time and Momentum (time, and P inference)
    mean_valid_time = accepted_df['time'].mean() if accepted_count > 0 else 0.0
    
    # Function to extract Mean Momentum (P proxy)
    def extract_mean_momentum(trace_str):
        try:
            # psi0_noise = "P1;P2;P3;P_dot1;P_dot2;P_dot3"
            parts = [float(x) for x in str(trace_str).split(';')]
            if len(parts) == 6:
                # Assume Momentum/Velocity are the final components
                return np.mean(np.abs(parts[3:])) 
        except:
            return np.nan
            
    # Create column with momentum proxy in the original dataframe (avoids SettingWithCopyWarning)
    df['Momentum_Proxy'] = df['psi0_noise'].apply(extract_mean_momentum)
    # Re-filter accepted data to include the new column
    accepted_df = df[df['status'].str.contains('Accepted', na=False, case=False)].copy() 
    mean_momentum_proxy = accepted_df['Momentum_Proxy'].mean() if len(accepted_df) > 0 else 0.0
    
    # === 4. REPORT GENERATION ===
    
    print("\n" + "="*80)
    print(f" 📈 SPHY QAOA TUNNELING QBENCH REPORT - {mode_info}".center(80))
    print("="*80)
    
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Total Rounds:** {total_rounds:,}")
    print(f" **Accepted Rounds (QAOA Executed):** {accepted_count:,} out of {total_rounds:,}")
    print("---")
    
    # -- A. Success Metrics (SPHY Filter) --
    print(" ## A. Controlled Tunneling Efficiency (SPHY)")
    print(f" * **Controlled Tunneling Rate:** **{success_rate:.2f}%**")
    print(f" * **Mean STDJ (f_opt) (Accepted):** **{mean_f_opt:.6f}**")
    print(f" * **STDJ Variance (f_opt):** {variance_f_opt:.6f} (Trigger Consistency)")
    
    # -- B. Quality Metrics (QAOA) --
    print("\n ## B. Quantum Solution Quality (QAOA)")
    print(f" * **Mean QAOA Energy:** **{mean_energy:.6f}** (Ideal Target: -1.0)")
    print(f" * **Energy Standard Deviation:** {std_energy:.6f} (Optimization Reliability)")
    
    # -- C. Dynamic Metrics (Time and Momentum P) --
    print("\n ## C. Noise Field Dynamics (P and Time)")
    print(f" * **Mean Trigger Time (time):** **{mean_valid_time:.4f}s** (Time to Reach Cancellation Zone)")
    print(f" * **Mean Momentum (P Proxy):** **{mean_momentum_proxy:.6f}** (Noise Derivative Influence)")
    
    print("="*80)
    
    # === 5. Plot Generation (Histograms) ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram 1: QAOA Energy Distribution
    ax[0].hist(accepted_df['energy'], bins=20, color='#00796B', alpha=0.8, edgecolor='black')
    ax[0].axvline(mean_energy, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_energy:.6f}')
    ax[0].axvline(-1.0, color='blue', linestyle='dotted', linewidth=1, label='Ideal (-1.0)')
    ax[0].set_title(f'QAOA Energy Distribution ({mode_info})')
    ax[0].set_xlabel('Energy (Exp. Value)')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Histogram 2: Momentum Distribution (P Proxy)
    ax[1].hist(accepted_df['Momentum_Proxy'], bins=20, color='#E91E63', alpha=0.8, edgecolor='black')
    ax[1].axvline(mean_momentum_proxy, color='darkgreen', linestyle='dashed', linewidth=1, label=f'Mean: {mean_momentum_proxy:.6f}')
    ax[1].set_title(f'Momentum Distribution (P Proxy) - Accepted ({mode_info})')
    ax[1].set_xlabel('Mean Momentum (P Proxy)')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    ax[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the main plot
    log_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir, f"qaoa_tunnel_p_report_{mode_info.replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Graphical report saved to: {graph_filename}")
    plt.show()

    # === 6. PHASE SPATIAL PLOT: ψ(t₀) — SPHY Field Proxy ===
    print("\n🌐 Generating Phase Spatial Plot...")

    # Extract P = [Px, Py, Pz] and Pdot = [Vx, Vy, Vz]
    positions = []
    velocities = []

    for row in accepted_df['psi0_noise']:
        try:
            parts = [float(p) for p in str(row).strip().split(';')]
            if len(parts) == 6:
                P = parts[:3]  # Position/Phase components (P1, P2, P3)
                Pdot = parts[3:] # Momentum/Velocity components (P_dot1, P_dot2, P_dot3)
                positions.append(P)
                velocities.append(Pdot)
        except:
            continue

    positions = np.array(positions)
    velocities = np.array(velocities)

    if positions.shape[0] == 0:
        print("⚠️ No valid vectors collected for the Phase Spatial Plot.")
    else:
        # Define a maximum number of vectors for legible visualization
        max_vectors = 1000
        skip = max(1, positions.shape[0] // max_vectors)  # space out the data

        fig_phase = plt.figure(figsize=(10, 8))
        ax3d = fig_phase.add_subplot(111, projection='3d')

        ax3d.quiver(
            positions[::skip, 0], positions[::skip, 1], positions[::skip, 2],
            velocities[::skip, 0], velocities[::skip, 1], velocities[::skip, 2],
            length=0.5, normalize=True, color='deepskyblue', alpha=0.7
        )

        ax3d.set_title(f'Phase Spatial Plot — SPHY Vectors ({mode_info})')
        ax3d.set_xlabel('P1 (X)')
        ax3d.set_ylabel('P2 (Y)')
        ax3d.set_zlabel('P3 (Z)')
        ax3d.grid(True)

        spatial_plot_filename = os.path.join(log_dir, f"sphy_phase_spatial_plot_{mode_info.replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(spatial_plot_filename, dpi=300)
        print(f"🌀 Spatial Plot saved to: {spatial_plot_filename}")

        plt.show()


if __name__ == "__main__":
    analyze_qaoa_tunnel_log()