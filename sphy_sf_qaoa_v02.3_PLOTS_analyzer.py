# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_qaoa_v02.3_PLOTS_analyzer.py
# Purpose: Analyzer and Benchmarker for QAOA/SPHY CSV Logs
# Author: deywe@QLZ | Benchmark Tool by Gemini AI
# ❗ UPDATE: Gráficos abertos no visualizador interativo do Matplotlib.
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from scipy.integrate import solve_ivp
from datetime import datetime

# ====================================================================
# === CONSTANTES E FUNÇÕES SPHY (MOCKUP PARA REPRODUÇÃO 3D)
# ====================================================================

# Constantes do script QAOA original
OMEGA = 2.0
DAMPING = 0.06
GAMMA = 0.5
LAMBDA_G = 1.0
G = 6.67430e-11
NOISE_LEVEL = 1.00 
CANCEL_THRESHOLD = 0.05 

# Mockup da função de dinâmica do SPHY
def sphy_harpia_3d_noise(t, psi, OMEGA, DAMPING, GAMMA, G, LAMBDA_G, NOISE_LEVEL):
    """
    Função Mockup para simular a dinâmica SPHY do script principal (6 Dimensões).
    """
    
    dpsi_dt = np.zeros_like(psi)
    
    # Velocidades (d(Posição)/dt = Velocidade)
    dpsi_dt[0:3] = psi[3:6]
    
    # Acelerações (d(Velocidade)/dt = Aceleração)
    restoration_term = -(OMEGA**2) * psi[0:3]
    damping_term = -DAMPING * psi[3:6]
    noise_influence = NOISE_LEVEL * np.sin(t * GAMMA) * LAMBDA_G * psi[0:3]

    dpsi_dt[3:6] = restoration_term + damping_term + noise_influence
    
    return dpsi_dt


# ====================================================================
# === FUNÇÕES DE ANÁLISE E PLOTAGEM
# ====================================================================

def calculate_metrics(df):
    """Calcula e exibe as métricas de desempenho da simulação."""
    
    total_frames = len(df)
    df_accepted = df[df['status'] == 'Accepted'].copy()
    accepted_count = df_accepted.shape[0]

    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    df_accepted['energy'] = pd.to_numeric(df_accepted['energy'], errors='coerce')
    df_accepted.dropna(subset=['energy'], inplace=True)
    
    rounds = df['round'].max() if not df['round'].empty else total_frames
    
    acceptance_rate = 100 * accepted_count / rounds if rounds > 0 else 0
    
    df_accepted['purity'] = -df_accepted['energy']
    
    mean_purity = df_accepted['purity'].mean() if accepted_count > 0 else 0.0
    max_purity = df_accepted['purity'].max() if accepted_count > 0 else 0.0
    
    print("\n" + "="*70)
    print("                REPORTS DE BENCHMARK QAOA/SPHY".center(70))
    print("="*70)
    print(f"{'Total Rounds (Tentativas):':<30} {rounds}")
    print(f"{'Accepted Rounds (QAOA):':<30} {accepted_count} ({acceptance_rate:.2f}%)")
    print("\n--- Métricas de Estabilidade (Coerência) ---")
    print(f"📈 Média de Purity (Coerência):{mean_purity:>30.6f}")
    print(f"🚀 Máxima Purity Alcançada:{max_purity:>30.6f}")
    print("="*70)
    
    return df_accepted, rounds, mean_purity


def generate_plots(df_accepted, rounds, timestamp_id):
    """Gera os gráficos 2D e 3D e os exibe no visualizador."""

    if df_accepted.empty:
        print("❌ Não há rodadas aceitas para plotar. Verifique os dados.")
        return

    # --- 1. Gráfico 2D: Purity (Coerência) vs. Rounds ---
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_accepted['round'], df_accepted['purity'], label='Purity (Maximized Coherence)', color='blue', marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('Round (Tentativa de Tunelamento)')
    plt.ylabel('Purity (Maximized Coerência)')
    plt.title(f'2D Plot: QAOA Solution Coherence (Purity) Over Rounds (Total: {rounds} Rounds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plot_2d_file = f"plot_2d_purity_REPRO_{timestamp_id}.png"
    plt.savefig(plot_2d_file, dpi=300)
    # ❗ REMOVIDO: plt.close() - Deixa a figura aberta
    print(f"✅ 2D Plot salvo: {plot_2d_file}")
    

    # --- 2. Gráfico 3D: Trajetória SPHY (Última Rodada Aceita) ---
    
    last_accepted_row = df_accepted.iloc[-1]
    psi0_str = last_accepted_row['psi0_noise']
    try:
        psi0_last = np.array([float(x) for x in psi0_str.split(';')])
    except:
        print("❌ Erro ao parsear o vetor inicial psi0_noise. Não é possível reproduzir o 3D.")
        return
    
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
    
    ax.plot(Px_traj, Py_traj, Pz_traj, label='SPHY 3D Trajectory (Reproduced)', color='purple')
    
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
    
    plot_3d_file = f"plot_3d_sphy_REPRO_{timestamp_id}.png"
    plt.savefig(plot_3d_file, dpi=300)
    # ❗ REMOVIDO: plt.close() - Deixa a figura aberta
    print(f"✅ 3D Plot salvo: {plot_3d_file}")

    # ❗ NOVO PASSO: ABRIR AMBOS OS GRÁFICOS INTERATIVAMENTE
    print("\n👁️ Abrindo gráficos interativos. Feche ambos para finalizar o script.")
    plt.show(block=True) 
    

def main():
    """Função principal para solicitar o caminho do CSV e iniciar a análise."""
    
    print("\n=============================================")
    print("  SPHY Benchmarker: Análise de Log CSV (QAOA)")
    print("=============================================")
    
    selected_csv = input("👉 Por favor, insira o caminho COMPLETO do arquivo CSV principal de log: ")
    
    if not os.path.exists(selected_csv):
        print(f"\n❌ Erro: O caminho '{selected_csv}' não é um arquivo válido ou não foi encontrado.")
        return

    try:
        df = pd.read_csv(selected_csv)
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo CSV: {e}")
        return

    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df_accepted, rounds, mean_purity = calculate_metrics(df)
    
    generate_plots(df_accepted, rounds, timestamp_id)
    
    print("\nAnálise de Benchmark concluída.")

if __name__ == "__main__":
    main()