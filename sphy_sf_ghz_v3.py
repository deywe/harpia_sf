# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_ghz_v3.py
# Purpose: GHZ + HARPIA (Strawberry Fields) + Adaptive Coherence Simulation + Meissner IA + MUX
# Author: deywe@QLZ | Versão Final e Robusta com MUX e Cores Dinâmicos por Gemini
# ───────────────────────────────────────────────────────────────

# 1️⃣ Importação dos módulos necessários: IA Meissner
from meissner_core import meissner_correction_step # Assumindo que este módulo existe e funciona

import strawberryfields as sf
# ❗ Rgate adicionado para simular o chaveamento no estágio MUX
from strawberryfields.ops import S2gate, BSgate, LossChannel, MeasureHomodyne, Rgate 
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os, random, sys, time, hashlib, re
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d
import logging

# ----------------------------------------------------------------------
# BLOC DE SILENCIAMENTO (Para evitar warnings de CUDA/TensorFlow)
# ----------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# ----------------------------------------------------------------------

# Define o backend e diretórios
ENG_BACKEND = 'gaussian' 
LOG_DIR = "logs_harpia_sf_meissner"
os.makedirs(LOG_DIR, exist_ok=True)


def get_user_parameters():
    """Coleta parâmetros de simulação do usuário."""
    try:
        num_modes = int(input("🔢 Number of Modes (Photonic Qubits) in GHZ circuit: "))
        total_frames = int(input("🔁 Total GHZ states to simulate: "))
        
        # Nível de Ruído (Transmissão T)
        transmittance_T = float(input("🌡️ Transmissão do Canal (T, 0.0 a 1.0. Ex: 0.1 para 90% de Perda): "))
        if not (0.0 <= transmittance_T <= 1.0):
            print("❌ Transmissão inválida. Deve estar entre 0.0 e 1.0.")
            sys.exit(1)
            
        # ❗ NOVA ENTRADA: Número de Processos
        num_processes = int(input("💻 Number of CPU cores for parallel processing (1 to 8): "))
        if not (1 <= num_processes <= 8):
            print("❌ Número de núcleos inválido. Escolha entre 1 e 8.")
            sys.exit(1)

            
        return num_modes, total_frames, transmittance_T, num_processes # Retorna 4 valores
    except ValueError:
        print("❌ Entrada inválida. Por favor, insira números inteiros ou floats válidos.")
        sys.exit(1)

def generate_ghz_state_sf(num_modes, transmittance_T):
    """Gera o circuito GHZ fotônico (CV) usando S2gate e BSgate com ruído de perda e MUX."""
    prog = sf.Program(num_modes)
    
    with prog.context as q:
        # 1. Preparação GHZ (Analogia ao H e CNOT)
        for i in range(1, num_modes):
            S2gate(0.8) | (q[0], q[i])

        # Beamsplitter para misturar o emaranhamento
        for i in range(num_modes - 1):
            BSgate(np.pi/4, 0) | (q[i], q[i+1])

        # 2. ❗ NOVO: ESTÁGIO DE MULTIPLEXAÇÃO/ROTEAMENTO (MUX)
        # Adiciona interconexão para simular multiplexação e reconfigurabilidade.
        if num_modes >= 2:
            # Cria um pequeno interferômetro de mistura
            for i in range(num_modes // 2):
                # Rgate simula um chaveamento/controle de fase
                Rgate(np.pi/8 * i) | q[2*i]
                BSgate(np.pi/8, np.pi/16) | (q[2*i], q[2*i+1])
            
            # Mistura final das saídas
            if num_modes > 2:
                BSgate(np.pi/4, 0) | (q[0], q[num_modes-1])
        # -------------------------------------------------------------

        # 3. Inserção de Ruído (Perda Térmica)
        # Transmissão de T é passada para o LossChannel(T)
        if num_modes > 1:
            mode_to_noise = random.randint(0, num_modes - 1)
            LossChannel(transmittance_T) | q[mode_to_noise]
            
        # 4. Medição
        MeasureHomodyne(0, q[0])
        
    return prog

def simulate_frame_sf(frame_data):
    """Simula um único frame, executa o circuito SF e aplica a correção Meissner/HARPIA."""
    frame, num_modes, total_frames, transmittance_T, sphy_coherence = frame_data
    random.seed(os.getpid() * frame)
    eng = sf.Engine(backend=ENG_BACKEND, backend_options={"cutoff_dim": 10}) 
    
    prog = generate_ghz_state_sf(num_modes, transmittance_T) # Transmitindo T

    current_timestamp = datetime.utcnow().isoformat()
    
    result_sf = eng.run(prog)
    
    # CORREÇÃO: Acessar o estado via result_sf.state
    mu = result_sf.state.means() 
    V = result_sf.state.cov()    
    
    pureza = 1.0 / np.sqrt(np.linalg.det(2 * V))
    
    result_string = "Emaranhado" if pureza > 0.6 else "Ruído/Perda"
    
    # Variáveis SPHY (Valores baseados no seu script Cirq original)
    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T_frame = frame
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] 

    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T_frame, psi_state)
    except Exception as e:
        return None, None, f"\nCritical error in frame {frame} (Meissner IA): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0
    
    # CRITÉRIO SPHY: Aceito se a correção Meissner foi ativa E a coerência resultante for mantida acima do limiar
    SPHY_SUCCESS_THRESHOLD = 90.0
    accepted = activated and (new_coherence > SPHY_SUCCESS_THRESHOLD)

    data_to_hash = f"{frame}:{pureza:.4f}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result_string, round(pureza * 100, 4), round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, new_coherence, None

def execute_simulation_multiprocessing_sf(num_modes, total_frames, transmittance_T, num_processes):
    """Executa a simulação em multiprocessamento e gera o relatório e o gráfico."""
    start_time = time.time()
    loss_percent = (1.0 - transmittance_T) * 100 # Para o título do relatório
    
    print("=" * 60)
    print(f" 🍓 HARPIA QGHZ STABILIZER + Meissner (SF) • {num_modes} Modes • {total_frames:,} Frames")
    print(f" 🌡️ RUÍDO TÉRMICO (PERDA): {loss_percent:.1f}%")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"qghz_sf_{num_modes}q_T{transmittance_T:.2f}_log_{timecode}.csv")
    
    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0) # Coerência inicial
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    
    # O parâmetro transmittance_T é incluído na tupla de entrada para o Pool
    frame_inputs = [(f, num_modes, total_frames, transmittance_T, sphy_coherence.value) for f in range(1, total_frames + 1)]

    # ❗ Usando o número de processos escolhido pelo usuário
    print(f"🔄 Using {num_processes} processes for simulation...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame_sf, frame_inputs),
                                                    total=total_frames, desc="⏳ Simulating GHZ (SF)"):
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
    
    total_time = time.time() - start_time
    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    
    # 4. Cálculo de Métricas e Relatório
    if sphy_evolution:
        sphy_np_array = np.array(list(sphy_evolution))
        mean_stability = np.mean(sphy_np_array)
        stability_variance = np.var(sphy_np_array)
        
        initial_coherence = 90.0 
        final_coherence = sphy_np_array[-1] if sphy_np_array.size > 0 else 0.0
        coherence_gain = final_coherence - initial_coherence
        
        boost_count = sum(1 for entry in log_data if entry[7] > 0)
        meissner_boost_success_rate = 100 * (boost_count / total_frames) if total_frames > 0 else 0
        
        print("\n" + "=" * 60)
        print("         HARPIA QPOC BENCHMARK REPORT (STRAWBERRY FIELDS)")
        print("=" * 60)
        print(f"⏱ Total time: {total_time:.2f}s")
        print(f"✅ GHZ States accepted: {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")
        print(f"📈 Meissner Boost Rate: {meissner_boost_success_rate:.2f}% (Correção Meissner aplicada)")
        
        print("\n--- Stability and Coherence Metrics ---")
        print(f"📊 Mean Stability Index: {mean_stability:.6f}")
        print(f"📊 Stability Variance Index: {stability_variance:.6f}")
        print(f"🚀 Total Coherence Gain: {coherence_gain:.4f}% (Final - Inicial)")


    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame", "CV_Result", "Purity (%)", "H", "S", "C", "I", 
            "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"
        ])
        writer.writerows(list(log_data))
    print(f"\n🧾 CSV saved: {csv_filename}")

    # 5. Geração de Gráfico (Lógica de plotagem mantida)
    sphy_evolution_list = list(sphy_evolution)
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    tempo = np.linspace(0, 1, len(sphy_evolution))
    
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    novo_tempo = np.linspace(0, 1, 2000)
    
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos) 

    estabilidade_media = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="SPHY Stabilized Coherence")
    for i in range(len(dados)):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("SPHY Coherence/Amplitude")
    ax1.set_title(f"GHZ CV Entanglement - {num_modes} Modes (Perda: {loss_percent:.1f}%)")
    ax1.legend()
    ax1.grid()

    ax2.plot(novo_tempo, emaranhamento, 'k-', label="SPHY Stabilized Coherence")
    ax2.axhline(estabilidade_media, color='green', linestyle='--', label=f"Mean: {estabilidade_media:.2f}")
    ax2.axhline(estabilidade_media + np.sqrt(estabilidade_variancia), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(estabilidade_media - np.sqrt(estabilidade_variancia), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("SPHY Coherence/Amplitude")
    ax2.set_title("Coherence Stability via HARPIA/Meissner")
    ax2.legend()
    ax2.grid()

    fig_filename = os.path.join(LOG_DIR, f"qghz_sf_{num_modes}q_T{transmittance_T:.2f}_graph_{timecode}.png")
    fig.suptitle(f"GHZ Simulation (SF): SPHY Coherence and Stability - {num_modes} Modes", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)
    print(f"\n📊 Graph saved as: {fig_filename}")

    plt.show()

if __name__ == "__main__":
    modes, pairs, transmittance, num_cores = get_user_parameters()
    execute_simulation_multiprocessing_sf(num_modes=modes, total_frames=pairs, transmittance_T=transmittance, num_processes=num_cores)