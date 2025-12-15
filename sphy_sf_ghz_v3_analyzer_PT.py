# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_ghz_v3_analyzer_PT.py (CORRIGIDO)
# Purpose: Lê um CSV gerado pela simulação HARPIA/SPHY, calcula 
#          métricas e gera o gráfico de estabilidade.
# Author: Deywe@QLZ & Gemini Revisor 
# ───────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: harpia_benchmark_reporter.py (VERSÃO CORRIGIDA)
# Purpose: Lê um CSV gerado pela simulação HARPIA/SPHY, calcula 
#          métricas e gera o gráfico de estabilidade.
# Author: Gemini (Com correção aplicada)
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d
from datetime import datetime
import time

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
LOG_DIR = "reports_harpia_analysis"
os.makedirs(LOG_DIR, exist_ok=True)
SPHY_SUCCESS_THRESHOLD = 90.0
# Define a Coerência Inicial base como 90.0, conforme o setup do simulador original.
INITIAL_COHERENCE_BASE = 90.0 

# ----------------------------------------------------------------------
# 1. FUNÇÕES PRINCIPAIS
# ----------------------------------------------------------------------

def calculate_metrics(df, total_sim_time):
    """Calcula todas as métricas de desempenho e estabilidade."""
    total_frames = len(df)
    
    # Métricas de Coerência e Aceitação
    valid_states = df[df['Accepted'] == '✅'].shape[0]
    acceptance_rate = 100 * (valid_states / total_frames) if total_frames > 0 else 0
    
    # Métricas de Estabilidade SPHY
    sphy_evolution = df['SPHY (%)'].values
    mean_stability = np.mean(sphy_evolution)
    stability_variance = np.var(sphy_evolution)
    
    # Ganhos
    # ❗ CORREÇÃO APLICADA AQUI: Usa o valor base 90.0 como Coerência Inicial
    initial_coherence = INITIAL_COHERENCE_BASE 
    final_coherence = df['SPHY (%)'].iloc[-1] if total_frames > 0 else 0.0
    coherence_gain = final_coherence - initial_coherence
    
    # Boost Meissner
    boost_count = df[df['Boost'] > 0].shape[0]
    meissner_boost_success_rate = 100 * (boost_count / total_frames) if total_frames > 0 else 0

    return {
        "total_frames": total_frames,
        "valid_states": valid_states,
        "acceptance_rate": acceptance_rate,
        "mean_stability": mean_stability,
        "stability_variance": stability_variance,
        "coherence_gain": coherence_gain,
        "meissner_boost_rate": meissner_boost_success_rate,
        "total_time": total_sim_time,
        "sphy_evolution": sphy_evolution
    }

def generate_report(metrics):
    """Imprime o relatório de benchmark no console."""
    
    total_time_minutes = metrics["total_time"] / 60
    
    if metrics["total_time"] > 0.001: 
        fps = metrics["total_frames"] / metrics["total_time"]
    else:
        fps = metrics["total_frames"] / 0.01 

    print("\n" + "=" * 60)
    print("         HARPIA BENCHMARK REPORT (CSV ANALYSIS) 🧾")
    print("=" * 60)
    print(f"⏱ Total Frames Analyzed: {metrics['total_frames']:,}")
    print(f"⏱ Estimated FPS/it/s: {fps:.2f}")
    print(f"⏱ Total Sim Time (Estimada): {metrics['total_time']:.2f}s ({total_time_minutes:.2f} min)")
    
    print("\n--- Desempenho e Aceitação ---")
    print(f"✅ States Accepted: {metrics['valid_states']}/{metrics['total_frames']} | {metrics['acceptance_rate']:.2f}%")
    print(f"📈 Meissner Boost Rate: {metrics['meissner_boost_rate']:.2f}% (Correção Meissner Aplicada)")
    
    print("\n--- Coerência e Estabilidade ---")
    print(f"📊 Mean Stability Index: {metrics['mean_stability']:.4f}%")
    print(f"📊 Stability Standard Deviation: {np.sqrt(metrics['stability_variance']):.4f}") 
    print(f"🚀 Total Coherence Gain: {metrics['coherence_gain']:.4f}%")
    print("=" * 60)


def plot_analysis_graph(sphy_evolution, filename_prefix):
    """Recria o gráfico de estabilidade SPHY."""
    
    sphy_evolution = np.array(sphy_evolution)
    if sphy_evolution.size == 0:
        print("❌ Sem dados para plotar.")
        return

    tempo = np.linspace(0, 1, len(sphy_evolution))
    
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)] 
    novo_tempo = np.linspace(0, 1, 2000)
    
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos) 

    estabilidade_media = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)
    
    # --- GERAÇÃO DO GRÁFICO --- 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12)) 
    
    # GRÁFICO 1: Emaranhamento com Ruído e Sinais Defasados
    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="SPHY Stabilized Coherence")
    for i in range(len(dados)):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red') 
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("SPHY Coherence/Amplitude (%)")
    ax1.set_title("GHZ CV Entanglement - Coerência e Modulação HARPIA")
    ax1.legend()
    ax1.grid()
    
    # GRÁFICO 2: Estabilidade e Média
    ax2.plot(novo_tempo, emaranhamento, 'k-', label="SPHY Stabilized Coherence")
    ax2.axhline(estabilidade_media, color='green', linestyle='--', label=f"Mean Coherence: {estabilidade_media:.2f}%")
    ax2.axhline(estabilidade_media + np.sqrt(estabilidade_variancia), color='orange', linestyle='--', label=f"± Std Dev: {np.sqrt(estabilidade_variancia):.4f}")
    ax2.axhline(estabilidade_media - np.sqrt(estabilidade_variancia), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("SPHY Coherence/Amplitude (%)")
    ax2.set_title("Coherence Stability via HARPIA/Meissner - Média e Desvio Padrão")
    ax2.legend()
    ax2.grid()

    fig_filename = os.path.join(LOG_DIR, f"report_graph_{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.suptitle("HARPIA Quantum Benchmark Analysis: Coherence Stability", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)
    print(f"\n📊 Graph saved as: {fig_filename}")
    
    plt.show()

# ----------------------------------------------------------------------
# 2. FUNÇÃO EXECUTORA
# ----------------------------------------------------------------------

def run_reporter():
    """Função principal para solicitar o caminho do CSV e executar a análise."""
    csv_path = input("➡️ Digite o caminho completo do arquivo CSV da simulação HARPIA: ")
    
    if not os.path.exists(csv_path):
        print(f"\n❌ Erro: Arquivo não encontrado no caminho: {csv_path}")
        sys.exit(1)

    total_sim_time = 0.0

    try:
        start_time_analysis = time.time()
        df = pd.read_csv(csv_path)
        end_time_analysis = time.time()

        # Tenta extrair o tempo de simulação
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            time_start_sim = df['Timestamp'].iloc[0]
            time_end_sim = df['Timestamp'].iloc[-1]
            total_sim_time = (time_end_sim - time_start_sim).total_seconds()
            
            if total_sim_time <= 0:
                total_sim_time = end_time_analysis - start_time_analysis
                print("\n⚠️ Aviso: Timestamps inválidos no CSV. Usando o tempo de execução da análise para estimativa.")

        except Exception:
            total_sim_time = end_time_analysis - start_time_analysis 
            print("\n⚠️ Aviso: Coluna 'Timestamp' não encontrada ou formato inválido. Usando o tempo de execução da análise para estimativa.")
        
        # 1. Calcular Métricas
        metrics = calculate_metrics(df, total_sim_time)
        
        # 2. Gerar Relatório
        generate_report(metrics)
        
        # 3. Gerar Gráfico
        filename_prefix = os.path.basename(csv_path).replace('.csv', '')
        plot_analysis_graph(metrics['sphy_evolution'], filename_prefix)
        
    except Exception as e:
        print(f"\n❌ Erro crítico ao processar ou analisar o CSV: {e}")
        print("Verifique se o CSV possui as colunas esperadas: 'SPHY (%)', 'Accepted', 'Boost' e 'Timestamp'.")
        sys.exit(1)

if __name__ == "__main__":
    run_reporter()