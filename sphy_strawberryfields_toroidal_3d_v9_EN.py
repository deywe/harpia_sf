# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_strawberryfields_toroidal_3d_v9_EN.py <--- UPDATED VERSION
# Purpose: QUANTUM TUNNELING IN A TOROIDAL LATTICE + SPHY FIELD ENGINEERING (DATA COLLECTION)
# Author: deywe@QLZ | Converted to SF by Gemini AI
# In the SPHY paradigm, the qubit is a universal sensor, so the degree of purity (μ) is a measure
#  of interactions, where 1 represents total disconnection from the universe.
# ───────────────────────────────────────────────────────────────

# 1️⃣ Import necessary modules
from meissner_core import meissner_correction_step 

# ⚛️ Strawberry Fields & CV Imports
import strawberryfields as sf
from strawberryfields import ops
import numpy as np 
from scipy.interpolate import griddata 
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Manager
from datetime import datetime
import os, random, sys, hashlib, csv
from scipy.interpolate import interp1d

# Fallback for mean_photon_number
try:
    from strawberryfields.measurements import mean_photon_number
except Exception:
    def mean_photon_number(cov, means, hbar=1.0):
        cov = np.array(cov)
        means = np.array(means)
        n_modes = means.size // 2
        n_list = []
        for i in range(n_modes):
            q_idx = 2 * i
            p_idx = q_idx + 1
            vq = cov[q_idx, q_idx]
            vp = cov[p_idx, p_idx]
            mq = means[q_idx]
            mp = means[p_idx]
            n_i = (vq + vp + mq * mq + mp * mp) / (2.0 * hbar) - 0.5
            n_list.append(float(n_i))
        return np.array(n_list)

# === SPHY Toroidal Lattice Configuration ===
GRID_SIZE = 2 
NUM_MODES = GRID_SIZE * GRID_SIZE # 4 Modes

# Target mode 
TARGET_MODE = 0 

TUNNELING_DIRECTION = 'either'
TUNNELING_THRESHOLD = 0.05     

# === Log Directory
LOG_DIR = "logs_sphy_toroidal_v9"
os.makedirs(LOG_DIR, exist_ok=True)

ENGINE_BACKEND = "gaussian"
INITIAL_COHERENCE = 90.0 # Default value added for consistency

# === Configuration and Helper Functions ===

def get_user_parameters():
    try:
        num_modes = NUM_MODES
        print(f"🔢 Number of Modes (Lattice {GRID_SIZE}x{GRID_SIZE}): {num_modes}")
        total_pairs = int(input("🔁 Total Tunneling Attempts (Frames) to simulate: "))
        
        barrier_strength_input = float(input("🚧 Barrier Strength (0.0 to 1.0): "))
        if not (0.0 <= barrier_strength_input <= 1.0):
             print("❌ Barrier Strength must be between 0.0 and 1.0.")
             exit(1)
             
        # Convert strength to a rotation angle for the Rgate
        barrier_strength_theta = barrier_strength_input * np.pi / 2 
        
        return num_modes, total_pairs, barrier_strength_theta
    except ValueError:
        print("❌ Invalid input. Please enter integers/floats.")
        exit(1)

def toroidal_tunneling_program_3d_log(barrier_theta, sphy_perturbation_angle):
    prog = sf.Program(NUM_MODES)
    with prog.context as q:
        # State Preparation (Squeezed coherent state on all modes)
        for mode in range(NUM_MODES):
            ops.Sgate(0.5) | q[mode]
            ops.Dgate(0.5, 0) | q[mode]

        # Toroidal Coupling (CZ Gates for connections)
        # Attention: The original code contained redundant/duplicated gates. 
        # Keeping the original code from 'ideal.txt' (v6) for compatibility.
        ops.CZgate(2) | (q[0], q[1])
        ops.CZgate(2) | (q[1], q[3])
        ops.CZgate(2) | (q[2], q[3])
        ops.CZgate(2) | (q[3], q[2]) 
        ops.CZgate(2) | (q[3], q[1]) 
        ops.CZgate(2) | (q[2], q[0]) 
        ops.CZgate(2) | (q[0], q[2])
        ops.CZgate(2) | (q[1], q[3])
        ops.CZgate(2) | (q[2], q[0]) 
        ops.CZgate(2) | (q[3], q[1]) 
        
        # Barrier (Rgate on target mode)
        ops.Rgate(barrier_theta) | q[TARGET_MODE]
        # SPHY Field Perturbation (Noise/Modulation on target mode)
        ops.Rgate(sphy_perturbation_angle) | q[TARGET_MODE]
        
        # Applying a smaller perturbation to adjacent modes
        for mode in [1, 2, 3]:
             ops.Rgate(sphy_perturbation_angle / 2) | q[mode]
             
    return prog

def simulate_frame(frame_data):
    frame, num_modes, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    
    random.seed(os.getpid() * frame) 
    
    sphy_perturbation_angle = 0.0
    # Apply random noise/perturbation based on noise_prob
    if random.random() < noise_prob:
        sphy_perturbation_angle = random.uniform(-np.pi/8, np.pi/8)
    
    program = toroidal_tunneling_program_3d_log(barrier_theta, sphy_perturbation_angle)
    
    try:
        eng = sf.Engine(ENGINE_BACKEND) 
        result = eng.run(program)
        state = result.state
    except Exception as e:
        return None, None, None, f"\nCritical Error running SF Engine on frame {frame}: {e}"

    # --- EXPECTATION VALUE & STATE DATA COLLECTION ---
    try:
        mean_photon_numbers = mean_photon_number(state.cov(), state.means())
        
        # Collect Covariance Matrix and Displacement Vector for the frame (Flat)
        # ONLY FOR THE TARGET MODE (TARGET_MODE = 0)
        q_idx = 2 * TARGET_MODE
        cov_target = state.cov()[q_idx:q_idx+2, q_idx:q_idx+2].flatten()
        means_target = state.means()[q_idx:q_idx+2].flatten()
        
    except Exception as e:
        return None, None, None, f"\nError calculating observables on frame {frame}: {e}"
    
    # Proxy for Pauli Z (relative displacement/population change)
    z_exp_proxy_base = 0.25 
    z_exp_proxies = [(n - z_exp_proxy_base) for n in mean_photon_numbers]

    target_z_expval = z_exp_proxies[TARGET_MODE]
    proxy_mag = abs(target_z_expval)

    if TUNNELING_DIRECTION == 'decrease':
        result_raw = 1 if target_z_expval < -TUNNELING_THRESHOLD else 0
    elif TUNNELING_DIRECTION == 'increase':
        result_raw = 1 if target_z_expval > TUNNELING_THRESHOLD else 0
    else:
        # Tunneling is successful if magnitude of change exceeds threshold
        result_raw = 1 if proxy_mag > TUNNELING_THRESHOLD else 0

    ideal_state = 1

    # === SPHY/Meissner Logic ===
    # These represent the stability factors of the environment/system
    H = random.uniform(0.95, 1.0) 
    S = random.uniform(0.95, 1.0) 
    C = sphy_coherence / 100    # Current Coherence (normalized)
    I = abs(H - S)             # Instability Index (analogous to Phi(t))
    T = frame                   
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] # Placeholder for phase/state vector

    try:
        # Apply the Quantum Gravity Modulation (QGM) correction step
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, None, f"\nCritical Error on frame {frame} (AI Meissner): {e}"

    # Calculate new coherence based on the boost
    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0 
    accepted = (result_raw == ideal_state) and activated
    
    # Log entry
    current_timestamp = datetime.utcnow().isoformat()
    data_to_hash = f"{frame}:{result_raw}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    phase_logs = [round(z, 4) for z in z_exp_proxies] 
    proxy_sign = '-' if target_z_expval < 0 else '+'
    
    log_entry = [
        frame, result_raw,
        *phase_logs,
        round(proxy_mag, 4), proxy_sign,
        *cov_target, # Vqq, Vqp, Vpq, Vpp
        *means_target, # mu_q, mu_p
        round(H, 4), round(S, 4), round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, new_coherence, (cov_target, means_target), None

# --- NEW SCIENTIFIC PLOTTING FUNCTIONS ---

def plot_wigner_function(cov_target, means_target, fig_filename_wigner):
    # ... (Wigner function placeholder)
    if cov_target is None or means_target is None:
        print("❌ Error: Quantum state data (Wigner) not available for plotting.")
        return

    # 2x2 Covariance Matrix
    cov = np.array([[cov_target[0], cov_target[1]], [cov_target[2], cov_target[3]]])
    means = np.array([means_target[0], means_target[1]])
    
    q_lim = max(3.0, np.max(np.abs(means)))
    q_grid = np.linspace(-q_lim, q_lim, 100)
    Q, P = np.meshgrid(q_grid, q_grid)
    
    coords = np.vstack([Q.flatten(), P.flatten()]).T
    
    try:
        wigner_pdf = multivariate_normal.pdf(coords, mean=means, cov=cov)
    except np.linalg.LinAlgError:
        print("⚠️ Linear Algebra Error in Wigner. Plotting vacuum state (Cov=I).")
        cov_id = np.identity(2) * 0.5 
        wigner_pdf = multivariate_normal.pdf(coords, mean=[0,0], cov=cov_id)
        
    W = wigner_pdf.reshape(Q.shape)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(Q, P, W, 100, cmap='RdBu_r')
    ax.scatter(means[0], means[1], marker='x', color='black', s=100, label='Center ($\mu_q, \mu_p$)')
    
    ax.set_title(f'Wigner Function of the Target Mode (Final Frame)', fontsize=14)
    ax.set_xlabel('Position Quadrature ($q$)')
    ax.set_ylabel('Momentum Quadrature ($p$)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(contour, label='W(q, p) Amplitude')
    
    plt.savefig(fig_filename_wigner, dpi=300)
    print(f"🖼️ Wigner Function saved: {fig_filename_wigner}")


def plot_tunneling_histogram(df, threshold, fig_filename_hist):
    # ... (Tunneling histogram function placeholder)
    if df.empty:
        print("❌ Error: Empty DataFrame for Histogram plotting.")
        return

    proxy_data = df['Proxy_Mag']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(proxy_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue', 
            label='Tunneling Proxy Magnitude ( |$\\Delta \\bar{n}$| )')
    
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Tunneling Threshold ({threshold})')
    
    success_count = (proxy_data >= threshold).sum()
    total_count = len(proxy_data)
    success_rate = 100 * (success_count / total_count)
    
    ax.text(0.95, 0.90, f'Total Success: {success_rate:.2f}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
            horizontalalignment='right', fontsize=12, color='darkgreen', weight='bold')

    ax.set_title(f'Performance Distribution (Tunneling Proxy) over {total_count} Frames', fontsize=14)
    ax.set_xlabel('Tunneling Proxy Magnitude ( |$\\Delta \\bar{n}$| )')
    ax.set_ylabel('Frequency of Occurrence (Frames)')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    plt.savefig(fig_filename_hist, dpi=300)
    print(f"🖼️ Tunneling Histogram saved: {fig_filename_hist}")


# === Main Simulation Function with New Metrics ===

def execute_simulation_multiprocessing(num_modes, total_frames, barrier_theta, noise_prob=1.0, num_processes=4):
    print("=" * 60)
    print(f" ⚛️ SPHY WAVES (SF): Toroidal Tunneling ({GRID_SIZE}x{GRID_SIZE}) • {total_frames:,} Frames")
    print(f" 🚧 Barrier Strength: {barrier_theta*180/np.pi:.2f} degrees Rgate (CV Analog)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"toroidal_{num_modes}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"toroidal_{num_modes}q_graph_2D_{timecode}.png")
    fig_filename_wigner = os.path.join(LOG_DIR, f"toroidal_{num_modes}q_WIGNER_FN_{timecode}.png")
    fig_filename_hist = os.path.join(LOG_DIR, f"toroidal_{num_modes}q_HISTOGRAM_{timecode}.png")


    manager = Manager()
    sphy_coherence = manager.Value('f', INITIAL_COHERENCE)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    last_state_data = manager.dict({'cov': None, 'means': None}) 

    frame_inputs = [
        (f, num_modes, total_frames, noise_prob, sphy_coherence.value, barrier_theta) 
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Using {num_processes} processes for simulation...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, state_data, error in tqdm(pool.imap_unordered(simulate_frame, frame_inputs),
                                            total=total_frames, desc="⏳ Simulating Toroidal SPHY (SF)"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence 
                if log_entry[-3] == "✅":
                    valid_states.value += 1
                
                if state_data:
                    last_state_data['cov'] = state_data[0]
                    last_state_data['means'] = state_data[1]


    # --- Metric Calculation (Including Purity, Squeezing, and Maximum Wigner) ---
    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    
    sphy_evolution_list = list(sphy_evolution)
    
    # 1. Preparation and Calculation of Classical SPHY Metrics
    if not sphy_evolution_list:
        print("❌ No data to calculate SPHY metrics or plot 2D graph.")
        return

    sphy_evolution_np = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution_np))
    
    n_redundancies = 2 
    signals = [interp1d(time_points, np.roll(sphy_evolution_np, i), kind='cubic') for i in range(n_redundancies)]
    new_time = np.linspace(0, 1, 2000)
    
    data = [sinal(new_time) + np.random.normal(0, 0.15, len(new_time)) for sinal in signals]
    weights = np.linspace(1, 1.5, n_redundancies)
    tunneling_stability = np.average(data, axis=0, weights=weights) 
    
    mean_sphy_stability = np.mean(data[1]) 
    stability_variance = np.var(data[1])
    
    # 2. Calculation of Quantum Metrics (Purity, Squeezing, Max Wigner)
    cov_target_flat = last_state_data['cov']
    purity = float('nan')
    squeezing_min = float('nan')
    wigner_max = float('nan')
    
    if cov_target_flat is not None:
        try:
            # Reconstruct the 2x2 Covariance Matrix for the Target Mode
            V = np.array([[cov_target_flat[0], cov_target_flat[1]], 
                          [cov_target_flat[2], cov_target_flat[3]]])
            
            # Purity (mu = 1 / sqrt(det(2V)))
            det_2V = np.linalg.det(2 * V)
            purity = 1.0 / np.sqrt(det_2V)
            
            # Squeezing (Minimum eigenvalue of V)
            # lambda_min = 0.5 * (Tr(V) - sqrt(Tr(V)^2 - 4 * det(V)))
            trace_V = np.trace(V)
            det_V = np.linalg.det(V)
            squeezing_min = 0.5 * (trace_V - np.sqrt(trace_V**2 - 4 * det_V))
            
            # Maximum Wigner (Wmax) at the center of the phase space (mu_q, mu_p)
            # Wmax = 1 / (pi * sqrt(det(2V))) = purity / pi
            wigner_max = purity / np.pi
            
        except np.linalg.LinAlgError:
            print("⚠️ LinAlgError when calculating Purity/Squeezing/Wigner. Invalid matrix.")
        except ValueError:
            print("⚠️ Squeezing calculation error. Negative value under the root.")

    # 3. Print Metrics to Console (Complete Report)
    print("\n" + "=" * 60)
    print("           📊 SPHY BENCHMARK REPORT")
    print("-" * 60)
    print(f"| ✅ Tunneling Success Rate (Toroidal SPHY): {valid_states.value}/{total_frames} | **{acceptance_rate:.2f}%**")
    print("-" * 60)
    print(f"| ⭐ Mean SPHY Stability: {mean_sphy_stability:.4f}")
    print(f"| 🌊 Stability Variance: {stability_variance:.6f}")
    print("-" * 60)
    # NEW QUANTUM METRICS
    # MODIFIED LINE:
    print(f"| ⚛️ Final State Purity (μ): {purity:.4f} (Ideal SPHY < 1.0 | Ideal QEC = 1.0)")
    # END OF MODIFIED LINE
    print(f"| 🔬 Min Squeezing (λ_min): {squeezing_min:.4f} (Squeezed < 0.5)")
    print(f"| 📈 Max Wigner (W_max): {wigner_max:.4f} (W_max <= 1/π)")
    print("=" * 60)
    
    # ... (CSV Writing)
    phase_cols = [f"Qubit_{i+1}_Phase" for i in range(NUM_MODES)]
    cov_headers = ["Vqq_0", "Vqp_0", "Vpq_0", "Vpp_0"]
    means_headers = ["mu_q_0", "mu_p_0"]

    header = [
        "Frame", "Result", 
        *phase_cols,
        "Proxy_Mag", "Proxy_Sign",
        *cov_headers, 
        *means_headers,
        "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", 
        "SHA256_Signature", "Timestamp"
    ]
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # --- Plotting Graphs ---
    plot_wigner_function(last_state_data['cov'], last_state_data['means'], fig_filename_wigner)
    
    df_results = pd.DataFrame(list(log_data), columns=header)
    plot_tunneling_histogram(df_results, TUNNELING_THRESHOLD, fig_filename_hist)

    # === 2D STABILITY PLOTTING CODE ===
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
    
    ax2.axhline(mean_sphy_stability, color='green', linestyle='--', label=f"Mean: {mean_sphy_stability:.2f}")
    ax2.axhline(mean_sphy_stability + np.sqrt(stability_variance), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(mean_sphy_stability - np.sqrt(stability_variance), color='orange', linestyle='--')

    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Quantum Tunneling Simulation (SF CV): {total_frames} Attempts (Toroidal SPHY)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)
    print(f"🖼️ 2D Stability Graph saved: {fig_filename}")
    plt.show(block=True) 


if __name__ == "__main__":
    modes, pairs, barrier_theta = get_user_parameters()
    
    execute_simulation_multiprocessing(num_modes=modes, total_frames=pairs, barrier_theta=barrier_theta, noise_prob=1.0, num_processes=4)