# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_dualrail_v6_tunnel_EN.py
# Purpose: QUANTUM TUNNELING + SPHY Phase Resonance (Strawberry Fields Dual-Rail)
# Author: deywe@QLZ | Updated by Gemini AI
# ───────────────────────────────────────────────────────────────
import os
# Assuming 'meissner_core' is available in the environment
# import meissner_core as meissner_core
from meissner_core import meissner_correction_step 

# ⚛️ Strawberry Fields & CV Imports
import strawberryfields as sf
from strawberryfields import ops

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import random, sys, time, hashlib
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d

# === Disables GPUs (in case the backend indirectly uses CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === SPHY Configurations
NUM_MODES = 2
CUTOFF_DIM = 5  # Limited Fock space
TARGET_MODE = 1
LOG_DIR = "logs_sphy_tunneling_sf"
os.makedirs(LOG_DIR, exist_ok=True)

# Activates Matplotlib interactive mode
plt.ion() 

# === Functions

def get_user_parameters():
    """Prompts the user for simulation parameters."""
    try:
        num_qubits = 1 
        print(f"🔢 Number of Logical Qubits (Dual-Rail): {num_qubits}")
        total_pairs = int(input("🔁 Total Simulations (Frames): "))
        barrier_strength_input = float(input("🚧 Barrier Strength (0.0 to 1.0): "))
        
        if not (0.0 <= barrier_strength_input <= 1.0):
            print("❌ Barrier must be between 0.0 and 1.0")
            exit(1)
        # Converts the strength to a rotation angle (theta)
        barrier_strength_theta = barrier_strength_input * np.pi / 2
        return num_qubits, total_pairs, barrier_strength_theta
    except ValueError:
        print("❌ Invalid input.")
        exit(1)

def generate_tunneling_program(barrier_theta: float, noise_prob: float = 1.0) -> sf.Program:
    """Generates the Strawberry Fields quantum program for tunneling simulation."""
    program = sf.Program(NUM_MODES)
    with program.context as q:
        # Prepare a photon in mode 0
        ops.Fock(1) | q[0]
        # Beam splitter (simulates splitting/tunneling attempt)
        ops.BSgate(np.pi/4, 0) | (q[0], q[1])
        # Barrier/Phase shift in one mode (representing the tunneling barrier)
        ops.Rgate(barrier_theta) | q[0]
        # Random noise in the second mode
        if random.random() < noise_prob:
            ops.Rgate(random.uniform(-np.pi/8, np.pi/8)) | q[1]
        # Measurement
        ops.MeasureFock() | q[0]
        ops.MeasureFock() | q[1]
    return program

def simulate_frame(frame_data):
    """Simulates a single quantum tunneling frame and applies the SPHY model."""
    frame, num_qubits, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    random.seed(os.getpid() * frame)
    ideal_state = 1 # Target result after tunneling attempt
    timestamp = datetime.utcnow().isoformat()

    program = generate_tunneling_program(barrier_theta, noise_prob)

    try:
        engine = sf.Engine("fock", backend_options={"cutoff_dim": CUTOFF_DIM})
        job_result = engine.run(program)
    except Exception as e:
        return None, None, f"\n🧨 Error in frame {frame} (SF Engine): {e}"
    
    # Check measurement result in the target mode
    measurement = job_result.samples[0]
    result = int(measurement[TARGET_MODE])

    # SPHY Meissner Model (Simulation of Stability Correction)
    H = random.uniform(0.95, 1.0) # Harmonic stability
    S = random.uniform(0.95, 1.0) # Spin stability
    C = sphy_coherence / 100 # Current Coherence
    I = abs(H - S) # Instability/Impact Factor
    T = frame # Time/Frame index
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] # Placeholder for internal state vector

    try:
        # Assumes meissner_correction_step is defined elsewhere and available
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\n🧨 Meissner Error in frame {frame}: {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    # Coherence is accepted if the result is ideal AND a positive boost was applied
    accepted = (result == ideal_state) and delta > 0

    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4), round(C, 4), round(I, 4),
        round(boost, 4), round(new_coherence, 4), "✅" if accepted else "❌", sha256_signature, timestamp
    ]
    return log_entry, new_coherence, None

def execute_simulation_multiprocessing(num_qubits, total_frames, barrier_theta, noise_prob=0.3, num_processes=4):
    """Executes the full quantum tunneling simulation using multiprocessing."""
    print("=" * 60)
    print(f"⚛️ SPHY-DualRail: Quantum Tunneling (SF) • {total_frames:,} Attempts")
    print(f"🚧 Barrier (θ): {barrier_theta * 180/np.pi:.2f} degrees")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"tunneling_sf_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"tunneling_sf_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    accepted_count = manager.Value('i', 0)

    # Prepare inputs for each frame simulation
    frame_inputs = [
        (f, num_qubits, total_frames, noise_prob, coherence.value, barrier_theta)
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Processes used: {num_processes}")
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(simulate_frame, frame_inputs),
                           total=total_frames, desc="⏳ Simulating Quantum Tunneling (SF)"):
            log_entry, new_val, error = result
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_val)
                coherence.value = new_val
                if log_entry[-3] == "✅":
                    accepted_count.value += 1

    success_rate = 100 * (accepted_count.value / total_frames)
    
    print(f"\n✅ Success: {accepted_count.value}/{total_frames} • {success_rate:.2f}%")

    if not sphy_evolution:
        print("❌ No data recorded.")
        return None 

    # 1. Save CSV Log
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # 2. Generate SPHY Graph
    sphy_np = np.array(list(sphy_evolution))
    x_vals = np.linspace(0, 1, len(sphy_np))
    signals = [interp1d(x_vals, np.roll(sphy_np, i), kind='cubic') for i in range(2)]
    new_x = np.linspace(0, 1, 2000)
    # Add minor noise for visual effect, consistent with the original plot style
    signal_outputs = [sig(new_x) + np.random.normal(0, 0.15, len(new_x)) for sig in signals]
    
    weights = np.linspace(1, 1.5, len(signal_outputs))
    final_curve = np.average(signal_outputs, axis=0, weights=weights)

    # Three-panel graph: SPHY Evolution, Stability Analysis, Raw Coherence + Histogram
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Ax1: SPHY Evolution
    ax1.plot(new_x, final_curve, 'k--', lw=2, label="Stable SPHY")

    for i, sig in enumerate(signal_outputs):
        ax1.plot(new_x, sig, alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_title("SPHY Evolution - Quantum Tunneling")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Phase Stability")
    ax1.legend()
    ax1.grid()

    # Ax2: Phase Variation and Stability
    mean, var = final_curve.mean(), final_curve.var()
    ax2.plot(new_x, final_curve, 'k-', lw=1.5, label="Mean Stability")
    ax2.axhline(mean, color='green', linestyle='--', label=f"Mean: {mean:.2f}")
    ax2.axhline(mean + np.sqrt(var), color='orange', linestyle='--', label="± Variance")
    ax2.axhline(mean - np.sqrt(var), color='orange', linestyle='--')
    ax2.set_title("Phase Variation and Stability")
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Stable Amplitude")
    ax2.legend()
    ax2.grid()

    # Ax3: Coherence Evolution per Frame (Raw Data) and Inset Histogram
    frames_idx = np.arange(1, len(sphy_np) + 1)
    ax3.plot(frames_idx, sphy_np, '-o', ms=3, lw=1, color='tab:purple', label='Coherence (frames)')
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

    fig.suptitle(f"SPHY Quantum Tunneling ({total_frames} Simulations)", fontsize=16)
    fig.subplots_adjust(top=0.92, hspace=0.45) # Manual adjustment for inset axes
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Graph saved: {fig_filename}")

    # 3. SPHY Stability Metrics Report
    stability_mean = final_curve.mean()
    stability_var = final_curve.var()
    coherence_gain = final_curve[-1] - 90.0  # Assumes initial point is 90.0

    print("\n📊 SPHY Stability Metrics:")
    print(f"📊 Mean Stability Index: {stability_mean:.6f}")
    print(f"📊 Stability Variance Index: {stability_var:.6f}")
    print(f"🚀 Total Coherence Gain: {coherence_gain:+.4f}% (Final - Initial)")
    
    return fig # Returns the figure object

# === Main Execution

if __name__ == "__main__":
    print("Starting simulation with interactive viewer...")
    # NOTE: Assumes 'meissner_core.py' is in the same directory for import
    qubits, pairs, theta_barreira = get_user_parameters()
    
    # Executes the simulation and receives the figure object
    fig_handle = execute_simulation_multiprocessing(
        num_qubits=qubits,
        total_frames=pairs,
        barrier_theta=theta_barreira,
        noise_prob=1.0,
        num_processes=4
    )
    
    # If the figure was generated, displays it interactively
    if fig_handle:
        print("\nOpening Matplotlib interactive viewer (zoom, pan, save)...")
        plt.show(block=True) # Blocks execution until the window is closed
    
    print("\nWindow closed. Program terminated.")