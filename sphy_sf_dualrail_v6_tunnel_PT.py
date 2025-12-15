# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_dualrail_v6_tunnel_PT.py
# Purpose: QUANTUM TUNNELING + SPHY Phase Resonance (Strawberry Fields Dual-Rail)
# Author: deywe@QLZ | Atualização por Tess AI
# ───────────────────────────────────────────────────────────────
import os
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

# === Desativa GPUs (caso backend utilize CUDA indiretamente)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === SPHY Configurações
NUM_MODES = 2
CUTOFF_DIM = 5  # Espaço Fock limitado
TARGET_MODE = 1
LOG_DIR = "logs_sphy_tunneling_sf"
os.makedirs(LOG_DIR, exist_ok=True)

# Ativa o modo interativo do Matplotlib
plt.ion() 

# === Funções

def get_user_parameters():
    try:
        num_qubits = 1 
        print(f"🔢 Número de Qubits Lógicos (Dual-Rail): {num_qubits}")
        total_pairs = int(input("🔁 Total de Simulações (Frames): "))
        barrier_strength_input = float(input("🚧 Força da Barreira (0.0 a 1.0): "))
        
        if not (0.0 <= barrier_strength_input <= 1.0):
            print("❌ Barreira deve estar entre 0.0 e 1.0")
            exit(1)
        barrier_strength_theta = barrier_strength_input * np.pi / 2
        return num_qubits, total_pairs, barrier_strength_theta
    except ValueError:
        print("❌ Entrada inválida.")
        exit(1)

def generate_tunneling_program(barrier_theta: float, noise_prob: float = 1.0) -> sf.Program:
    program = sf.Program(NUM_MODES)
    with program.context as q:
        ops.Fock(1) | q[0]
        ops.BSgate(np.pi/4, 0) | (q[0], q[1])
        ops.Rgate(barrier_theta) | q[0]
        if random.random() < noise_prob:
            ops.Rgate(random.uniform(-np.pi/8, np.pi/8)) | q[1]
        ops.MeasureFock() | q[0]
        ops.MeasureFock() | q[1]
    return program

def simulate_frame(frame_data):
    frame, num_qubits, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    random.seed(os.getpid() * frame)
    ideal_state = 1
    timestamp = datetime.utcnow().isoformat()

    program = generate_tunneling_program(barrier_theta, noise_prob)

    try:
        engine = sf.Engine("fock", backend_options={"cutoff_dim": CUTOFF_DIM})
        job_result = engine.run(program)
    except Exception as e:
        return None, None, f"\n🧨 Erro no frame {frame} (SF Engine): {e}"
    
    measurement = job_result.samples[0]
    result = int(measurement[TARGET_MODE])

    # SPHY Meissner Model
    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T = frame
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\n🧨 Erro Meissner no frame {frame}: {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    accepted = (result == ideal_state) and delta > 0

    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4), round(C, 4), round(I, 4),
        round(boost, 4), round(new_coherence, 4), "✅" if accepted else "❌", sha256_signature, timestamp
    ]
    return log_entry, new_coherence, None

def execute_simulation_multiprocessing(num_qubits, total_frames, barrier_theta, noise_prob=0.3, num_processes=4):
    print("=" * 60)
    print(f"⚛️ SPHY-DualRail: Quantum Tunneling (SF) • {total_frames:,} Tentativas")
    print(f"🚧 Barreira (θ): {barrier_theta * 180/np.pi:.2f} graus")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"tunneling_sf_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"tunneling_sf_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    accepted_count = manager.Value('i', 0)

    frame_inputs = [
        (f, num_qubits, total_frames, noise_prob, coherence.value, barrier_theta)
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Processos usados: {num_processes}")
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(simulate_frame, frame_inputs),
                           total=total_frames, desc="⏳ Simulando Túnel Quântico (SF)"):
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
    
    print(f"\n✅ Sucesso: {accepted_count.value}/{total_frames} • {success_rate:.2f}%")

    if not sphy_evolution:
        print("❌ Nenhum dado gravado.")
        return None # Retorna None para indicar que não há figura para mostrar

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV salvo: {csv_filename}")

    # 🌊 Gráfico SPHY
    sphy_np = np.array(list(sphy_evolution))
    x_vals = np.linspace(0, 1, len(sphy_np))
    signals = [interp1d(x_vals, np.roll(sphy_np, i), kind='cubic') for i in range(2)]
    new_x = np.linspace(0, 1, 2000)
    signal_outputs = [sig(new_x) + np.random.normal(0, 0.15, len(new_x)) for sig in signals]
    
    weights = np.linspace(1, 1.5, len(signal_outputs))
    final_curve = np.average(signal_outputs, axis=0, weights=weights)

    # Adiciona um terceiro painel: evolução da coerência por frame + histograma
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    ax1.plot(new_x, final_curve, 'k--', lw=2, label="SPHY Estável")

    for i, sig in enumerate(signal_outputs):
        ax1.plot(new_x, sig, alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_title("Evolução SPHY - Túnel Quântico")
    ax1.set_xlabel("Tempo Normalizado")
    ax1.set_ylabel("Estabilidade Fásica")
    ax1.legend()
    ax1.grid()

    media, var = final_curve.mean(), final_curve.var()
    ax2.plot(new_x, final_curve, 'k-', lw=1.5, label="Estabilidade Média")
    ax2.axhline(media, color='green', linestyle='--', label=f"Média: {media:.2f}")
    ax2.axhline(media + np.sqrt(var), color='orange', linestyle='--', label="± Variância")
    ax2.axhline(media - np.sqrt(var), color='orange', linestyle='--')
    ax2.set_title("Variação Fásica e Estabilidade")
    ax2.set_xlabel("Tempo Normalizado")
    ax2.set_ylabel("Amplitude Estável")
    ax2.legend()
    ax2.grid()

    # ax3: evolução da coerência por frame (dados brutos) e histograma sobreposto
    frames_idx = np.arange(1, len(sphy_np) + 1)
    ax3.plot(frames_idx, sphy_np, '-o', ms=3, lw=1, color='tab:purple', label='Coerência (frames)')
    ax3.set_title('Evolução da Coerência por Frame')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('SPHY (%)')
    ax3.grid(alpha=0.4)

    # Insere um pequeno histograma como inset no ax3
    left, bottom, width, height = 0.70, 0.08, 0.20, 0.18
    ax_hist = fig.add_axes([left, bottom, width, height])
    ax_hist.hist(sphy_np, bins=20, color='gray', alpha=0.8)
    ax_hist.set_title('Histograma SPHY', fontsize=9)
    ax_hist.tick_params(labelsize=8)

    fig.suptitle(f"Túnel Quântico SPHY ({total_frames} Simulações)", fontsize=16)
    # Ajuste manual em vez de tight_layout (inset axes não são compatíveis)
    fig.subplots_adjust(top=0.92, hspace=0.45)
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {fig_filename}")

    # 📊 MÉTRICAS DE ESTABILIDADE SPHY
    stability_mean = final_curve.mean()
    stability_var = final_curve.var()
    coherence_gain = final_curve[-1] - 90.0  # Considerando ponto inicial como 90.0

    print("\n📊 Métricas de Estabilidade SPHY:")
    print(f"📊 Mean Stability Index: {stability_mean:.6f}")
    print(f"📊 Stability Variance Index: {stability_var:.6f}")
    print(f"🚀 Total Coherence Gain: {coherence_gain:+.4f}% (Final - Inicial)")
    
    return fig # Retorna o objeto da figura

# === Execução Principal

if __name__ == "__main__":
    print("Iniciando simulação com visualizador interativo...")
    qubits, pairs, theta_barreira = get_user_parameters()
    
    # Executa a simulação e recebe o objeto da figura
    fig_handle = execute_simulation_multiprocessing(
        num_qubits=qubits,
        total_frames=pairs,
        barrier_theta=theta_barreira,
        noise_prob=1.0,
        num_processes=4
    )
    
    # Se a figura foi gerada, exibe-a interativamente
    if fig_handle:
        print("\nAbrindo visualizador interativo do Matplotlib (zoom, pan, salvar)...")
        plt.show(block=True) # Bloqueia a execução até a janela ser fechada
    
    print("\nJanela fechada. Programa encerrado.")