# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_qaoa_v02.3_.py (VERSÃO ROBUSTA COM TRATAMENTO NUMÉRICO)
# Purpose: QAOA + HARPIA + SPHY + Strawberry Fields (SF) + Metrics + PLOTS
# Author: deywe@QLZ | Converted to SF by Gemini AI
# ❗ ROBUSTNESS FIX: Implemented error handling in compute_zz_expectation 
#    to prevent RuntimeWarnings (invalid det/sqrt/divide by zero) during optimization.
# ───────────────────────────────────────────────────────────────

import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import hashlib
import csv
import os
import sys
from datetime import datetime
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tqdm import tqdm
from multiprocessing import Pool
import time
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import platform 
import webbrowser 
# -------------------------------------------------------------------

# === Configuração de Logging/Warnings (Mantida a minimização)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# Note: O tratamento de erros na função compute_zz_expectation 
# deve ser suficiente, mas o silenciamento pode ser reativado se necessário.

# ====================================================================
# === ENTROPY CONTROL SELECTION LOGIC
# ====================================================================

try:
    user_mode = input("Tunneling Mode:\n1. Low Entropy Control (Default SPHY, harp_ia_simbiotic)\n2. High Entropy Control (Sphy_e)\nChoose (default 1): ").strip()
    mode = int(user_mode) if user_mode else 1
except ValueError:
    mode = 1

if mode == 2:
    module_suffix = "_e"
    print("\n⚠️ Loading HIGH ENTROPY CONTROL modules (Maximum Determinism).")
else:
    module_suffix = ""
    print("\n✅ Loading LOW ENTROPY CONTROL modules (Default SPHY).")

try:
    exec(f"from harp_ia_simbiotic{module_suffix} import calcular_F_opt")
    exec(f"from harp_ia_noise_3d_dynamics{module_suffix} import sphy_harpia_3d_noise")
except ImportError as e:
    print(f"External modules not found. Check files with suffix '{module_suffix}'. Error: {e}")
    sys.exit(1)

# === Physical Constants
OMEGA = 2.0
DAMPING = 0.06
GAMMA = 0.5
LAMBDA_G = 1.0
G = 6.67430e-11
NOISE_LEVEL = 1.00 
CANCEL_THRESHOLD = 0.05
STDJ_THRESHOLD = 0.88

OUTPUT_DIR = "logs_harpia_sphy_sf_controlled"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
LOG_CSV = os.path.join(OUTPUT_DIR, f"harpia_tunnel_sf_batch_{TIMESTAMP}_MODE{mode}.csv")
UIDS_CSV = os.path.join(OUTPUT_DIR, f"uid_accepted_sf_{TIMESTAMP}_MODE{mode}.csv")

HEADER = ["round", "time", "energy", "SHA256", "status", "psi0_noise", "f_opt", "timestamp"]
for path in [LOG_CSV, UIDS_CSV]:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)

# ====================================================================
# === STRAWBERRY FIELDS CORE
# ====================================================================

NUM_MODES = 4 

def prepare_dual_rail(q):
    ops.Dgate(0.5) | q[0] 
    ops.Dgate(0.5) | q[2] 

def dual_rail_hadamard(q_mode_a, q_mode_b):
    ops.BSgate(np.pi/4, 0) | (q_mode_a, q_mode_b)

def dual_rail_rz(q_mode_a, q_mode_b, angle):
    ops.Rgate(angle) | q_mode_a
    ops.Rgate(-angle) | q_mode_b

def dual_rail_rx(q_mode_a, q_mode_b, angle):
    dual_rail_hadamard(q_mode_a, q_mode_b)
    dual_rail_rz(q_mode_a, q_mode_b, angle)
    dual_rail_hadamard(q_mode_a, q_mode_b)


def build_qaoa_program(beta, gamma):
    prog = sf.Program(NUM_MODES)
    with prog.context as q:
        prepare_dual_rail(q)
        dual_rail_hadamard(q[0], q[1]) 
        dual_rail_hadamard(q[2], q[3]) 
        ops.CZgate(2 * gamma) | (q[0], q[2]) 
        dual_rail_rz(q[2], q[3], 2 * gamma)
        dual_rail_rx(q[0], q[1], 2 * beta) 
        dual_rail_rx(q[2], q[3], 2 * beta) 
    return prog

# === FUNÇÃO DE CÁLCULO DE PURITY ROBUSTA
def compute_zz_expectation(state):
    """
    Calcula a Purity (como proxy de coerência) de forma robusta.
    Trata erros numéricos que causam RuntimeWarnings (det < 0 ou det = 0).
    """
    V = state.cov()
    
    # ❗ IMPLEMENTAÇÃO ROBUSTA DE TRATAMENTO DE ERROS
    try:
        det_2V = np.linalg.det(2 * V)
    except np.linalg.LinAlgError:
        # Se a matriz for singular e o det falhar (raro, mas possível)
        return 1e6 
        
    # Se o determinante for não-positivo (causa RuntimeWarning: invalid value encountered in sqrt)
    if det_2V <= 0:
        # Retorna uma energia muito ruim (minimização evita este ponto)
        return 1e5 
    
    # Se det_2V for muito próximo de zero (causa RuntimeWarning: divide by zero)
    if det_2V < 1e-18:
        return 1e5 

    purity = 1.0 / np.sqrt(det_2V)
    
    # Retorna Purity Negativa (para que minimize a energia)
    return -purity

def compute_energy_sf(params):
    beta, gamma = params
    program = build_qaoa_program(beta, gamma)
    ENG_LOCAL = sf.Engine("gaussian") 
    result = ENG_LOCAL.run(program)
    return compute_zz_expectation(result.state)

# === Controlled Tunneling with QAOA + SHA256
def attempt_controlled_tunneling_sf(valid_time, round_id, psi0_trace_str, f_opt):
    init_params = np.random.uniform(-np.pi, np.pi, size=2)
    
    # Otimização. Graças ao tratamento de erros na função de custo, será mais robusta.
    res = minimize(compute_energy_sf, init_params, method='Powell')
    energy = res.fun 
    
    # Verifica se a otimização resultou em um valor inválido (do nosso tratamento de erro)
    if energy >= 1e5:
        row = [round_id, valid_time, "-", "-", f"QAOA Failed (Energy {energy:.0f})", psi0_trace_str, f_opt, datetime.utcnow().isoformat()]
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return False
        
    data_to_hash = f"{valid_time:.6f}:{energy:.4f}:{datetime.utcnow().isoformat()}:{round_id}:{f_opt:.4f}:{psi0_trace_str}"
    signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()
    
    row = [round_id, valid_time, energy, signature, "Accepted", psi0_trace_str, f_opt, datetime.utcnow().isoformat()]
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f1, open(UIDS_CSV, "a", newline="", encoding="utf-8") as f2:
        writer = csv.writer(f1)
        writer.writerow(row)
        csv.writer(f2).writerow(row)
    return True

# === HARPIA + SPHY Round 
def run_harpia_round(args):
    round_id, total_rounds = args
    
    psi0 = np.random.uniform(-1.5, 1.5, size=6)
    psi0_trace_str = ";".join(f"{v:.4f}" for v in psi0)

    sol = solve_ivp(
        sphy_harpia_3d_noise,
        t_span=(0, 40),
        y0=psi0,
        t_eval=np.linspace(0, 40, 1000),
        args=(OMEGA, DAMPING, GAMMA, G, LAMBDA_G, NOISE_LEVEL)
    )

    Px, Py, Pz = sol.y[0], sol.y[1], sol.y[2]
    power = Px**2 + Py**2 + Pz**2
    valid_indices = np.where(power < CANCEL_THRESHOLD)[0]

    if len(valid_indices) == 0:
        row = [round_id, "-", "-", "-", "No SPHY zone", psi0_trace_str, "-", datetime.utcnow().isoformat()]
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return False

    valid_t = sol.t[valid_indices[0]]
    H, S, C, I = np.random.uniform(0.75, 1.0, size=4)
    P = np.mean(np.abs(sol.y[3:6]))

    f_opt = calcular_F_opt(H, S, C, I, valid_t, P)

    if f_opt >= STDJ_THRESHOLD:
        return attempt_controlled_tunneling_sf(valid_t, round_id, psi0_trace_str, round(f_opt, 4))
    else:
        row = [round_id, valid_t, "-", "-", f"Rejected STDJ={f_opt:.4f}", psi0_trace_str, f_opt, datetime.utcnow().isoformat()]
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return False

# ====================================================================
# === FUNÇÕES DE PLOTAGEM (FINAL)
# ====================================================================

def generate_plots_from_log(log_file, timestamp_id):
    """
    Gera o gráfico 2D de Energia vs. Rounds e o gráfico 3D da Trajetória SPHY.
    """
    print("\n\n📊 Generating Visual Reports...")
    
    try:
        df = pd.read_csv(log_file)
        
        # --- 1. Gráfico 2D: Energia (Purity) vs. Rounds ---
        
        df_accepted = df[df['status'] == 'Accepted'].copy()
        
        if df_accepted.empty:
            print("❌ No accepted rounds to plot Energy evolution.")
            return

        # Correção para garantir que 'energy' seja float
        df_accepted['energy'] = pd.to_numeric(df_accepted['energy'], errors='coerce')
        df_accepted.dropna(subset=['energy'], inplace=True) 

        if df_accepted.empty:
            print("❌ No accepted rounds remain after cleaning for 2D plot.")
            return
            
        df_accepted['purity'] = -df_accepted['energy']
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_accepted['round'], df_accepted['purity'], label='Purity (Maximized Coherence)', color='blue', alpha=0.7)
        plt.xlabel('Round (Tentativa de Tunelamento)')
        plt.ylabel('Purity (Maximized Coherence)')
        plt.title('2D Plot: QAOA Solution Coherence (Purity) Over Rounds')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plot_2d_file = os.path.join(OUTPUT_DIR, f"plot_2d_energy_{timestamp_id}.png")
        plt.savefig(plot_2d_file, dpi=300)
        plt.close()
        print(f"✅ 2D Plot saved: {plot_2d_file}")
        
        # Abrir o gráfico 2D
        try:
            if platform.system() == "Windows":
                os.startfile(plot_2d_file)
            else:
                webbrowser.open_new_tab('file://' + os.path.abspath(plot_2d_file))
        except Exception as e:
            print(f"⚠️ Could not automatically open 2D plot: {e}")

        # --- 2. Gráfico 3D: Trajetória SPHY (Última Rodada Aceita) ---
        
        last_accepted_row = df_accepted.iloc[-1]
        psi0_str = last_accepted_row['psi0_noise']
        psi0_last = np.array([float(x) for x in psi0_str.split(';')])
        
        sol_3d = solve_ivp(
            sphy_harpia_3d_noise,
            t_span=(0, 40),
            y0=psi0_last,
            t_eval=np.linspace(0, 40, 1000),
            args=(OMEGA, DAMPING, GAMMA, G, LAMBDA_G, NOISE_LEVEL)
        )
        
        Px_traj, Py_traj, Pz_traj = sol_3d.y[0], sol_3d.y[1], sol_3d.y[2]
        time_traj = sol_3d.t

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(Px_traj, Py_traj, Pz_traj, label='SPHY 3D Trajectory', color='purple')
        
        power_sum = Px_traj**2 + Py_traj**2 + Pz_traj**2
        valid_indices = np.where(power_sum < CANCEL_THRESHOLD)[0]
        
        if valid_indices.size > 0:
            tunnel_point = valid_indices[0]
            ax.scatter(Px_traj[tunnel_point], Py_traj[tunnel_point], Pz_traj[tunnel_point], 
                       color='red', s=50, label=f'Tunneling Point (t={time_traj[tunnel_point]:.2f}s)')
        
        ax.set_xlabel('Px (Ruído Simétrico)')
        ax.set_ylabel('Py (Ruído Assimétrico)')
        ax.set_zlabel('Pz (Ruído de Fase)')
        ax.set_title(f'3D Plot: SPHY Noise Dynamics Trajectory (Round {last_accepted_row["round"]})')
        ax.legend()
        
        plot_3d_file = os.path.join(OUTPUT_DIR, f"plot_3d_sphy_{timestamp_id}.png")
        plt.savefig(plot_3d_file, dpi=300)
        plt.close()
        print(f"✅ 3D Plot saved: {plot_3d_file}")
        
        # Abrir o gráfico 3D
        try:
            if platform.system() == "Windows":
                os.startfile(plot_3d_file)
            else:
                webbrowser.open_new_tab('file://' + os.path.abspath(plot_3d_file))
        except Exception as e:
            print(f"⚠️ Could not automatically open 3D plot: {e}")
        
    except Exception as e:
        print(f"❌ Error during plot generation: {e}")


# ====================================================================
# === MAIN EXECUTION
# ====================================================================
if __name__ == "__main__":
    # --- Ask: Cores ---
    try:
        max_cores = os.cpu_count()
        user_cores = input(f"Available cores: {max_cores}. How many to use? (1-{max_cores}): ").strip()
        num_cores = int(user_cores)
        if not (1 <= num_cores <= max_cores):
            raise ValueError
    except:
        num_cores = min(8, max_cores)
        print(f"Invalid input. Using {num_cores} cores.")

    # --- Ask: Rounds ---
    DEFAULT_ROUNDS = 1000
    try:
        user_input = input(f"How many rounds? (default={DEFAULT_ROUNDS}): ").strip()
        rounds = int(user_input) if user_input else DEFAULT_ROUNDS
        if rounds <= 0:
            rounds = DEFAULT_ROUNDS
    except:
        rounds = DEFAULT_ROUNDS

    # --- Initial Timestamp ---
    start_time = time.time()
    start_iso = datetime.utcnow().isoformat()
    print(f"\nStarting simulation: {start_iso}")
    print(f"Rounds: {rounds} | Cores: {num_cores}")
    print(f"Logs -> {OUTPUT_DIR}\n")

    # --- Multithreading + Progress Bar
    args_list = [(r, rounds) for r in range(1, rounds + 1)]
    accepted_count = 0

    with Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(run_harpia_round, args_list),
                            total=rounds, desc="HARPIA + QAOA (SF)", colour="cyan", leave=False):
            if result:
                accepted_count += 1

    # --- Final Metrics ---
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    speed = rounds / total_time if total_time > 0 else 0
    acceptance_rate = 100 * accepted_count / rounds if rounds > 0 else 0

    end_iso = datetime.utcnow().isoformat()

    print("\n" + "="*70)
    print("                FINAL SIMULATION METRICS".center(70))
    print("="*70)
    print(f"{'Total Rounds:':<30} {rounds}")
    print(f"{'Accepted (QAOA):':<30} {accepted_count} ({acceptance_rate:.2f}%)")
    print(f"{'Total Time:':<30} {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'Average Speed:':<30} {speed:.2f} rounds/s")
    print(f"{'Cores Used:':<30} {num_cores}")
    print(f"{'Framework:':<30} Strawberry Fields (Gaussian)")
    print(f"{'Start:':<30} {start_iso}")
    print(f"{'End:':<30} {end_iso}")
    print(f"{'Logs Saved in:':<30} {OUTPUT_DIR}")
    print("="*70)
    print(f"Main CSV -> {os.path.basename(LOG_CSV)}")
    print(f"Accepted UIDs -> {os.path.basename(UIDS_CSV)}")
    print("="*70)
    
    generate_plots_from_log(LOG_CSV, TIMESTAMP) 
    
    print("\nSimulation complete.")