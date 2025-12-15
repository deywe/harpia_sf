# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_sf_dualrail_v6_tunnel_PT_analyzer.py
# Purpose: Analisa o CSV do Túnel Quântico e reproduz o gráfico SPHY.
# Author: Gemini AI
# ───────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d

# Ativa o modo interativo do Matplotlib
plt.ion() 

def input_csv_path():
    """Solicita o caminho do arquivo CSV e verifica sua existência."""
    while True:
        # Pede o caminho, removendo aspas e espaços
        path = input("\nDigite o caminho completo do CSV da Simulação Tunneling: ").strip().strip('"').strip("'")
        if os.path.exists(path) and path.lower().endswith('.csv'):
            print(f"Arquivo encontrado: {path}")
            return path
        print("Arquivo não encontrado ou inválido. Tente novamente.")

def analyze_and_plot(filepath):
    """Carrega dados, calcula métricas e gera o gráfico em 3 painéis."""
    print("Carregando dados...")
    
    # Lê apenas as colunas essenciais
    try:
        df = pd.read_csv(
            filepath,
            usecols=['Frame', 'SPHY (%)', 'Accepted'],
            dtype={'Frame': 'int32', 'SPHY (%)': 'float32', 'Accepted': 'category'},
            engine='c'
        )
    except KeyError:
        print("❌ Erro: O CSV não contém as colunas 'Frame', 'SPHY (%)' e 'Accepted'.")
        return None

    sphy_np = df['SPHY (%)'].values
    total_frames = len(sphy_np)
    
    if total_frames == 0:
        print("❌ O arquivo CSV está vazio.")
        return None

    print("Calculando métricas e reconstruindo a curva SPHY...")

    # === Reconstrução da Curva SPHY (Deve replicar a lógica do gerador) ===
    x_vals = np.linspace(0, 1, len(sphy_np))
    signals = [interp1d(x_vals, np.roll(sphy_np, i), kind='cubic') for i in range(2)]
    new_x = np.linspace(0, 1, 2000)
    
    # OBS: O ruído aleatório NÃO será reproduzido aqui, apenas a curva média principal.
    # Para ser fiel ao visual do plot de geração, usaremos as saídas interpoladas.
    signal_outputs = [sig(new_x) for sig in signals] 
    
    weights = np.linspace(1, 1.5, len(signal_outputs))
    final_curve = np.average(signal_outputs, axis=0, weights=weights)

    # === Cálculo das Métricas de Estabilidade ===
    stability_mean = final_curve.mean()
    stability_var = final_curve.var()
    coherence_gain = final_curve[-1] - 90.0 # Ponto inicial assumido

    stats = {
        "Total Frames": total_frames,
        "Mean Stability Index": stability_mean,
        "Stability Variance Index": stability_var,
        "Total Coherence Gain (%)": coherence_gain
    }
    
    # === Geração do Gráfico (3 Painéis) ===
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    # Ax1: Evolução SPHY - Túnel Quântico
    ax1.plot(new_x, final_curve, 'k--', lw=2, label="SPHY Estável (Reconstruído)")
    for i, sig in enumerate(signal_outputs):
        ax1.plot(new_x, sig, alpha=0.3, color='tab:blue' if i == 0 else 'tab:red', 
                 label=f"Sinal {i+1} (Interp.)" if total_frames < 2000 else None)
    ax1.set_title("Evolução SPHY - Túnel Quântico (Análise)")
    ax1.set_xlabel("Tempo Normalizado")
    ax1.set_ylabel("Estabilidade Fásica")
    ax1.legend()
    ax1.grid(alpha=0.5)

    # Ax2: Variação Fásica e Estabilidade
    ax2.plot(new_x, final_curve, 'k-', lw=1.5, label="Estabilidade Média")
    ax2.axhline(stability_mean, color='green', linestyle='--', label=f"Média: {stability_mean:.6f}")
    ax2.axhline(stability_mean + np.sqrt(stability_var), color='orange', linestyle='--', label="± Variância (Desvio Padrão)")
    ax2.axhline(stability_mean - np.sqrt(stability_var), color='orange', linestyle='--')
    ax2.set_title("Variação Fásica e Estabilidade (Controle SPHY)")
    ax2.set_xlabel("Tempo Normalizado")
    ax2.set_ylabel("Amplitude Estável")
    ax2.legend()
    ax2.grid(alpha=0.5)

    # Ax3: Evolução da Coerência por Frame (dados brutos) e Histograma
    frames_idx = df['Frame'].values
    ax3.plot(frames_idx, sphy_np, '-', ms=3, lw=1, color='tab:purple', label='Coerência Bruta (frames)')
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


    fig.suptitle(f"Análise de Benchmark Túnel Quântico SPHY ({total_frames} Frames)", fontsize=16)
    fig.subplots_adjust(top=0.92, hspace=0.45)
    
    # Salva o gráfico
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = filepath.replace(".csv", f"_ANALYZED_PLOT_{now}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico salvo: {plot_path}")

    return stats, plot_path, fig

def print_report(stats, plot_path):
    """Imprime o relatório de métricas na tela."""
    print("\n" + "═" * 70)
    print(" " * 15 + "RELATÓRIO DE BENCHMARK SPHY - TÚNEL QUÂNTICO")
    print("═" * 70)
    print(f"Total de Frames Analisados: {stats['Total Frames']:,}")
    print("-" * 70)
    print("Métricas de Estabilidade SPHY (Curva Reconstruída):")
    print(f"   Mean Stability Index (Média): {stats['Mean Stability Index']:.6f}")
    print(f"   Stability Variance Index (Variância): {stats['Stability Variance Index']:.6f}")
    print(f"   Total Coherence Gain (Ganho Líquido): {stats['Total Coherence Gain (%)']:+.4f}%")
    print("-" * 70)
    print(f"Gráfico de Análise (3 Painéis): {os.path.basename(plot_path)}")
    print("═" * 70)

# === Execução Principal

if __name__ == "__main__":
    print("Iniciando Análise de Benchmark com Visualizador Interativo")
    csv_path = input_csv_path()
    
    # 1. Analisa e gera o gráfico
    result = analyze_and_plot(csv_path)
    
    if result:
        stats, plot_path, fig_handle = result
        
        # 2. Imprime o relatório
        print_report(stats, plot_path)
        
        # 3. Exibe o visualizador interativo
        print("\nAbrindo visualizador interativo do Matplotlib (zoom, pan, salvar)...")
        plt.show(block=True) 
        print("\nJanela fechada. Programa encerrado.")
    else:
        print("Análise abortada devido a erro de dados.")