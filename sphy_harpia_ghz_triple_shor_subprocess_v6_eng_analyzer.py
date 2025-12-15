#!/usr/bin/env python3

# File: sphy_harpia_master_auditor_v4_eng.py
# Purpose: Robust auditing tool for HARPIA logs.
# 🚀 FINAL FIX: Reference time updated to 5.04s.
# Maintains the total fidelity logic (V3) in the calculations.

import os
import csv
from statistics import mean, stdev
import time
import math
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import platform
import subprocess

# --- REFERENCE CONFIGURATION ---
LOG_DIR = "logs_triple_shor_subp"
# 🚀 FINAL FIX: Reference time updated to 5.04s (from the simulator log).
# Manually change this value for the target simulation time of each run, if necessary.
AUDIT_DURATION = 5.04 

def find_header_index(header, possible_names):
    """Attempts to find the index of a column from a list of possible names."""
    for name in possible_names:
        if name in header:
            return header.index(name)
    return None

def parse_success(value):
    """Robust helper function for parsing True/False, 1/0, or the symbol '✅'."""
    if isinstance(value, str):
        lower_value = value.lower().strip()
        # Checks for 'True', '1', or the success character (ignores space/null)
        return lower_value == 'true' or lower_value == '1' or lower_value == 't' or lower_value == '✅'
    # Explicitly converts 1/0 from int/float to True/False
    return value == 1

def analyze_csv_and_generate_report(file_path):
    """
    Loads the CSV, performs the benchmark recalculation, and generates the summary.
    """
    try:
        data = []
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = [h.strip() for h in next(reader)] 
            for row in reader:
                data.append(row)

    except Exception as e:
        print(f"\n❌ ERROR reading CSV: {e}")
        return

    if not data:
        print("\n❌ ERROR: The CSV file is empty or unreadable after the header.")
        return

    # 1. Extract column indices and prepare for shift correction
    coherence_idx = find_header_index(header, ["Coherence"])
    symbol_idx = find_header_index(header, ["Symbol", "Valid_Throughput"]) # Added Valid_Throughput
    n15_idx = find_header_index(header, ["Shor N15"])
    n21_idx = find_header_index(header, ["Shor N21"])
    n35_idx = find_header_index(header, ["Shor N35"])

    required_indices = {
        "Coherence": coherence_idx, "Shor N15": n15_idx, 
        "Shor N21": n21_idx, "Shor N35": n35_idx
    }

    missing_cols = [k for k, v in required_indices.items() if v is None]
    if missing_cols:
        print("\n❌ ERROR: The CSV file does not contain all required columns.")
        print(f"Missing or differently named columns: {', '.join(missing_cols)}")
        return

    # 2. Logic for index shift correction
    first_row_len = len(data[0]) 
    apply_shift = False
    
    if len(header) > first_row_len and len(header) == first_row_len + 1 and symbol_idx is not None:
        apply_shift = True
        print(f"ℹ️ Detected 1 column difference (Scenario: {len(header)} in header vs. {first_row_len} in data).")
        # Assuming the missing column (Symbol/Valid_Throughput) is the one causing the shift.
        print(f"ℹ️ Compensating for index shift of -1 for columns after '{header[symbol_idx]}'.")

    def get_corrected_index(original_idx):
        if apply_shift and symbol_idx is not None and original_idx > symbol_idx:
            return original_idx - 1
        return original_idx

    # Apply correction to all required indices
    coherence_idx = get_corrected_index(coherence_idx)
    n15_idx = get_corrected_index(n15_idx)
    n21_idx = get_corrected_index(n21_idx)
    n35_idx = get_corrected_index(n35_idx)
    
    max_required_index = max(coherence_idx, n35_idx) 
    
    # 3. Process and collect data
    coherence_values = []
    processed_data = []
    lines_skipped = 0

    for i, row in enumerate(data):
        if len(row) < max_required_index + 1:
            lines_skipped += 1
            continue
            
        try:
            coherence_val = float(row[coherence_idx])
            
            processed_row = {
                "coherence": coherence_val,
                "success_n15": parse_success(row[n15_idx]),
                "success_n21": parse_success(row[n21_idx]),
                "success_n35_raw": parse_success(row[n35_idx]), 
            }
            processed_data.append(processed_row)
            coherence_values.append(coherence_val)
            
        except ValueError:
            lines_skipped += 1
            continue
            
    if not processed_data:
        print("\n❌ ERROR: No processable data rows found.")
        return

    total_frames = len(processed_data)

    print(f"\nℹ️ Processed Rows: {total_frames}. Skipped Rows: {lines_skipped}.")

    # 4. BENCHMARK CALCULATION WITH TOTAL FIDELITY

    # 4.1. N=15 Success (S15)
    n15_success_count = sum(1 for row in processed_data if row["success_n15"])
    
    # 4.2. N=15 Rejections (R15) -> N21 Reuse Space
    n21_reuse_space = total_frames - n15_success_count
    
    # 4.3. N=21 Success WITHIN THE REUSE SPACE (S21)
    s_n21_in_reuse_space = sum(1 for row in processed_data if not row["success_n15"] and row["success_n21"])
    n21_reuse_success_rate = (s_n21_in_reuse_space / n21_reuse_space) * 100 if n21_reuse_space > 0 else 0.0
    
    
    # 4.4. N=35 CALCULATION (CORRECT LOGIC)
    
    # STEP 1: Calculate the total number of N21 SUCCESSES (anywhere)
    total_n21_success_count = sum(1 for row in processed_data if row["success_n21"])
    
    # STEP 2: N35 Denominator is the TOTAL N21 REJECTIONS (anywhere)
    n35_denominator = total_frames - total_n21_success_count 
    
    # STEP 3: N35 Success (S35) is the ACTUAL reading from the CSV, conditioned on N21 failure.
    s_n35_in_reuse_space = sum(1 for row in processed_data if not row["success_n21"] and row["success_n35_raw"])
    
    # N35 Reuse Success Rate
    reuse_n35_rate = (s_n35_in_reuse_space / n35_denominator) * 100 if n35_denominator > 0 else 0.00
    
    # 4.5. Total Utility and HARPIA Advantage
    total_utility_count = n15_success_count + s_n21_in_reuse_space + s_n35_in_reuse_space 
    total_utility_percent = (total_utility_count / total_frames) * 100
    harpia_advantage = s_n21_in_reuse_space + s_n35_in_reuse_space 
    
    # 5. Stability Index Calculation
    mean_stability = mean(coherence_values) if coherence_values else 0.0
    stability_variance = stdev(coherence_values) ** 2 if len(coherence_values) > 1 else 0.0
    
    
    # 6. Summary Printout
    
    n15_success_rate = (n15_success_count/total_frames)*100
    
    print("\n" + "="*60)
    print("HARPIA TRIPLE SHOR SUMMARY (MASTER AUDIT V4 - CORRECTED)") 
    print("="*60)
    print(f"⏱ Total simulation time: {AUDIT_DURATION:.2f}s (Reference Value)") 
    print(f"🔢 Simulated frames: {total_frames:,}\n") 
    
    print("| **HARPIA UTILITY SUMMARY** |")
    # Line 1: N15 Success 
    print(f"| 1. Success (N=15): {n15_success_count}/{total_frames} ({n15_success_rate:.2f}%)")
    # Line 2: N15 Rejections 
    print(f"| 2. Rejections (N=15): {n21_reuse_space}/{total_frames}")
    # Line 3: N21 Success 
    print(f"| 3. Reuse Success (N=21): {s_n21_in_reuse_space}/{n21_reuse_space} ({n21_reuse_success_rate:.2f}%)")
    # Line 4: Remaining N21 Rejections (DYNAMIC and CORRECT)
    print(f"| 4. Rejections (N=21, rest.): {n35_denominator}/{total_frames}")
    # Line 5: N35 Success (ACTUAL Log Value)
    print(f"| 5. Reuse Success (N=35): {s_n35_in_reuse_space}/{n35_denominator} ({reuse_n35_rate:.2f}%)")
    
    print(f"| **TOTAL UTILITY (N1, N2, or N3):** {total_utility_count}/{total_frames} | **{total_utility_percent:.2f}%**\n") 
    
    print(f"<<< **HARPIA ADVANTAGE:** {harpia_advantage} factors of N=21/N=35 found by Immediate Reuse.") 
    
    # Stability Indices 
    print(f"\n📊 Mean Stability Index: {mean_stability:.6f}")
    print(f"📊 Stability Variance Index: {stability_variance:.6f}")
    
    csv_file_name = os.path.basename(file_path)
    # Correct file name, changing 'rastreio' to 'tracking'
    plot_file_path = os.path.join(LOG_DIR, csv_file_name.replace('.csv', '_tracking.png'))
    
    print(f"📁 Audited CSV: {file_path}") 
    print(f"📊 Qubit Tracking Plot expected at: {plot_file_path}") 
    print("="*60)

    # 7. Attempt to open the generated plot file
    if os.path.exists(plot_file_path):
        print("\n✅ Plot file found. Attempting to open with system viewer...")
        try:
            if platform.system() == "Windows":
                # Windows command to open file with default viewer
                os.startfile(plot_file_path)
            elif platform.system() == "Darwin": # macOS
                # macOS command to open file with default viewer
                subprocess.call(('open', plot_file_path))
            else: # Linux/other (requires xdg-open to be installed)
                # Linux command to open file with default viewer
                subprocess.call(('xdg-open', plot_file_path))
            print("Successfully requested to open the plot.")
        except Exception as e:
            print(f"⚠️ Could not open plot automatically. Error: {e}")
    else:
        print("❌ Plot file not found at the expected location. Please ensure the simulator script has run and created the '_tracking.png' file.")


def main_auditor():
    root = tk.Tk()
    root.withdraw()

    print("\n----------------------------------------------------")
    print("       HARPIA Master Auditor Analyzer V4")
    print("----------------------------------------------------")
    print("Select the log CSV file for validation...")

    csv_path = filedialog.askopenfilename(
        title="Select the HARPIA Log CSV File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if not csv_path:
        print("❌ No selection made. Shutting down the analyzer.")
        return
        
    print(f"✅ File loaded: {csv_path}")

    start_time = time.time()
    analyze_csv_and_generate_report(csv_path)
    end_time = time.time()
    print(f"\nAudit completed in: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    main_auditor()