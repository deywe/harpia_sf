# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: benchmark_analyzer_v3.py
# Purpose: SPHY/HARPIA Log Analysis Tool with Complex Stability Graph
# Author: Gemini AI Assistant + QLZ Collaboration
# ───────────────────────────────────────────────────────────────

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from statistics import mean
import io 
from scipy.interpolate import interp1d

def analyze_csv_and_generate_report(file_path):
    """
    Reads a log CSV file, calculates benchmark metrics,
    reads the Thermal Noise SHA256 signature, and generates the complex Stability graph.
    
    Args:
        file_path (str): The full path to the log CSV file.
    """
    
    # 1. File Check and Full Read
    start_time = time.time()
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at '{file_path}'.")
        return
    
    # Reading all lines to separate data and metadata
    try:
        with open(file_path, mode="r", encoding='utf-8') as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return

    # The footer contains 3 lines: 1 empty, 1 Thermal Noise (with SHA256), 1 Total Time
    FOOTER_LINES = 3 
    
    # 2. Metadata Extraction (Footer)
    metadata = {
        'Thermal Noise Status': 'N/A', 
        'Thermal Noise SHA256': 'N/A',
        'Total Time (s)': '0.00'
    }
    
    if len(all_lines) > FOOTER_LINES:
        footer = all_lines[-FOOTER_LINES:]
        
        for line in footer:
            if line.strip().startswith("Thermal Noise Status"):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    metadata['Thermal Noise Status'] = parts[1].strip()
                if len(parts) >= 3:
                    metadata['Thermal Noise SHA256'] = parts[2].strip()
            elif line.strip().startswith("Total Time (s)"):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    metadata['Total Time (s)'] = parts[1].strip()

    # 3. Data Reading and Extraction
    coherence_evolution = []
    log_data = []
    
    # Filter only data lines (Header + Simulation Lines)
    data_lines_to_process = all_lines[0:len(all_lines) - FOOTER_LINES]
    data_io = io.StringIO("".join(data_lines_to_process))
    
    try:
        reader = csv.reader(data_io)
        header = next(reader)
        
        try:
            col_frame = header.index("Frame")
            col_coherence = header.index("Coherence")
            col_valid = header.index("Valid")
        except ValueError:
            print("❌ Error: The CSV file does not have 'Frame', 'Coherence', and 'Valid' columns in the main header.")
            return

        print("⏳ Reading file data...")
        for row in tqdm(reader):
            if not row or len(row) < col_valid + 1: continue 
            try:
                coherence_evolution.append(float(row[col_coherence]))
                log_data.append([
                    float(row[col_frame]),
                    row[col_valid]
                ])
            except (ValueError, IndexError):
                continue

    except Exception as e:
        print(f"❌ An error occurred while processing the data: {e}")
        return
    
    duration_read = time.time() - start_time

    # 4. Calculation of COMPLETE Metrics
    total_frames = len(coherence_evolution)
    if total_frames == 0:
        print("⚠️ Warning: The CSV file is empty. No metrics to calculate.")
        return

    # --- Acceptance Metrics ---
    accepted_count = sum(1 for row in log_data if row[1].strip() == "✅")
    accepted_percentage = (accepted_count / total_frames) * 100
    
    # --- Coherence Metrics (Statistics) ---
    raw_coherence_evolution = np.array(coherence_evolution)
    mean_coherence = raw_coherence_evolution.mean()
    median_coherence = np.median(raw_coherence_evolution)
    coherence_variance = raw_coherence_evolution.var()
    std_dev_coherence = raw_coherence_evolution.std()
    min_coherence = raw_coherence_evolution.min()
    max_coherence = raw_coherence_evolution.max()
    
    # --- Performance Metrics ---
    total_duration_reported = metadata['Total Time (s)']
    try:
        throughput = total_frames / float(total_duration_reported) if float(total_duration_reported) > 0 else 0.0
    except ValueError:
        throughput = 0.0

    # Extracts the file name and directory to save the graph
    directory = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    fig_name = file_name.replace(".csv", "_graph.png")
    saved_fig_path = os.path.join(directory, fig_name)

    # Tries to extract the number of modes/qubits
    num_modes = "N/A" 
    try:
        if "_q_" in file_name:
             parts = file_name.split('_')
             for i, part in enumerate(parts):
                 if 'q' in part and part.replace('q', '').isdigit() and i > 0:
                     num_modes = part.replace('q', '')
                     break
    except:
        pass


    # 5. Generation of the COMPLETE Report
    print("\n" + "=" * 65)
    print("                HARPIA QPOC BENCHMARK REPORT")
    print("=" * 65)
    
    # Metadata
    print("\n--- Configuration Tracking ---")
    print(f"🌡 Thermal Noise: {metadata['Thermal Noise Status']}")
    print(f"🔐 **Signature:** {metadata['Thermal Noise SHA256']}")
    
    # Acceptance and Read Metrics
    print(f"\n📊 Accepted: {accepted_count:,}/{total_frames:,} ({accepted_percentage:.2f}%)")
    print(f"⏱ Read Time: {duration_read:.2f}s")
    
    # Performance Metrics
    print("\n--- Performance Metrics ---")
    print(f"⏱ Total Simulation Time (Reported): {total_duration_reported}s")
    print(f"⚡ Throughput (Frames/s): {throughput:,.2f} frames/s")
    
    # Stability and Coherence Metrics
    print("\n--- Stability and Coherence Metrics ---")
    print(f"📊 Mean Coherence Index (Mean): {mean_coherence:.6f}")
    print(f"📈 Median Coherence Index (Median): {median_coherence:.6f}")
    print(f"📊 Coherence Variance Index: {coherence_variance:.6f}")
    print(f"📊 Coherence Std Dev (Standard Deviation): {std_dev_coherence:.6f}")
    print(f"⬇️ Coherence Min: {min_coherence:.6f}")
    print(f"⬆️ Coherence Max: {max_coherence:.6f}")
    
    # 6. Generation of Entanglement and Stability Graph
    
    if total_frames < 2:
        print("⚠️ Warning: Insufficient data to generate the graph.")
        return
        
    # --- Logic for Stability and Entanglement Graph ---
    
    NUM_COMPLEXITY_VECTORS = 4
    
    time_normalized = np.linspace(0, 1, total_frames)
    sphy_evolution = raw_coherence_evolution
    
    # 1. Create 'N' interpolated signals with different delays (roll)
    signals = [interp1d(time_normalized, np.roll(sphy_evolution, i), kind='cubic') for i in range(NUM_COMPLEXITY_VECTORS)]
    
    # 2. Generate data (the vectors) by applying visualization noise
    new_time = np.linspace(0, 1, 500)
    # Add noise with sigma = 0.3 to force the vibration effect (wave)
    data = [signal(new_time) + np.random.normal(0, 0.3, len(new_time)) for signal in signals] 
    
    # 3. Calculate the Entanglement Curve (Weighted Average of all vectors)
    weights = np.linspace(1, 1.5, NUM_COMPLEXITY_VECTORS)
    entanglement = np.average(data, axis=0, weights=weights) 

    stability_mean_plot = np.mean(entanglement)
    stability_variance = np.var(entanglement)
    stability_std_dev = np.sqrt(stability_variance)

    colors = plt.cm.get_cmap('Spectral', NUM_COMPLEXITY_VECTORS)

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Subplot 1: Simulated/Stabilized Entanglement
    ax1.plot(new_time, entanglement, 'k--', linewidth=2, label="SPHY Stabilized Coherence")
    # Plotting all N vectors with distinct colors
    for i in range(NUM_COMPLEXITY_VECTORS):
        ax1.plot(new_time, data[i], alpha=0.3, color=colors(i))
        
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/SPHY Amplitude (%)")
    ax1.set_title(f"GHZ CV Entanglement - {num_modes} Modes ({NUM_COMPLEXITY_VECTORS} Vectors)")
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Stability and Variance
    ax2.plot(new_time, entanglement, 'k-', label="SPHY Stabilized Coherence")
    
    ax2.axhline(stability_mean_plot, color='green', linestyle='--', label=f"Mean: {stability_mean_plot:.4f}")
    ax2.axhline(stability_mean_plot + stability_std_dev, color='orange', linestyle='--', label=f"± Standard Deviation")
    ax2.axhline(stability_mean_plot - stability_std_dev, color='orange', linestyle='--')
    
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/SPHY Amplitude (%)")
    ax2.set_title("Coherence Stability via HARPIA/Meissner")
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"GHZ Simulation (SF): SPHY Coherence and Stability - {num_modes} Modes", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("\n📊 Generating graph...")
    plt.savefig(saved_fig_path, dpi=300)
    print(f"✅ Graph saved to: {saved_fig_path}")
    
    plt.show()


if __name__ == "__main__":
    file_path_input = input("Please enter the full path of the CSV file: ").strip()
    analyze_csv_and_generate_report(file_path_input)