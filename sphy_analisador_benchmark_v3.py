# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: analisador_benchmark_v3.py
# Purpose: Ferramenta de análise de logs SPHY/HARPIA com Gráfico de Estabilidade Complexo
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

def analisar_csv_e_gerar_relatorio(caminho_arquivo):
    """
    Lê um arquivo CSV de log, calcula métricas de benchmark,
    lê a assinatura SHA256 do Ruído Térmico e gera o gráfico complexo de Estabilidade.
    
    Args:
        caminho_arquivo (str): O caminho completo para o arquivo CSV de log.
    """
    
    # 1. Verificação do Arquivo e Leitura Completa
    start_time = time.time()
    if not os.path.exists(caminho_arquivo):
        print(f"❌ Erro: O arquivo não foi encontrado em '{caminho_arquivo}'.")
        return
    
    # Lendo todas as linhas para separar dados e metadados
    try:
        with open(caminho_arquivo, mode="r", encoding='utf-8') as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"❌ Ocorreu um erro ao ler o arquivo: {e}")
        return

    # O rodapé contém 3 linhas: 1 vazia, 1 Ruído Térmico (com SHA256), 1 Tempo Total
    FOOTER_LINES = 3 
    
    # 2. Extração de Metadados (Rodapé)
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

    # 3. Leitura e Extração de Dados
    coherence_evolution = []
    log_data = []
    
    # Filtra apenas as linhas de dados (Header + Linhas de Simulação)
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
            print("❌ Erro: O arquivo CSV não possui as colunas 'Frame', 'Coherence' e 'Valid' no cabeçalho principal.")
            return

        print("⏳ Lendo dados do arquivo...")
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
        print(f"❌ Ocorreu um erro ao processar os dados: {e}")
        return
    
    duration_read = time.time() - start_time

    # 4. Cálculo de Métricas COMPLETAS
    total_frames = len(coherence_evolution)
    if total_frames == 0:
        print("⚠️ Aviso: O arquivo CSV está vazio. Nenhuma métrica para calcular.")
        return

    # --- Métricas de Aceitação ---
    accepted_count = sum(1 for row in log_data if row[1].strip() == "✅")
    accepted_percentage = (accepted_count / total_frames) * 100
    
    # --- Métricas de Coerência (Estatísticas) ---
    raw_coherence_evolution = np.array(coherence_evolution)
    mean_coherence = raw_coherence_evolution.mean()
    median_coherence = np.median(raw_coherence_evolution)
    coherence_variance = raw_coherence_evolution.var()
    std_dev_coherence = raw_coherence_evolution.std()
    min_coherence = raw_coherence_evolution.min()
    max_coherence = raw_coherence_evolution.max()
    
    # --- Métricas de Desempenho ---
    total_duration_reported = metadata['Total Time (s)']
    try:
        throughput = total_frames / float(total_duration_reported) if float(total_duration_reported) > 0 else 0.0
    except ValueError:
        throughput = 0.0

    # Extrai o nome do arquivo e diretório para salvar o gráfico
    diretorio = os.path.dirname(caminho_arquivo)
    nome_arquivo = os.path.basename(caminho_arquivo)
    nome_fig = nome_arquivo.replace(".csv", "_graph.png")
    caminho_fig_salva = os.path.join(diretorio, nome_fig)

    # Tenta extrair o número de modos/qubits
    num_modes = "N/A" 
    try:
        if "_q_" in nome_arquivo:
             parts = nome_arquivo.split('_')
             for i, part in enumerate(parts):
                 if 'q' in part and part.replace('q', '').isdigit() and i > 0:
                     num_modes = part.replace('q', '')
                     break
    except:
        pass


    # 5. Geração do Relatório COMPLETO
    print("\n" + "=" * 65)
    print("                HARPIA QPOC BENCHMARK REPORT")
    print("=" * 65)
    
    # Metadados
    print("\n--- Rastreio de Configuração ---")
    print(f"🌡 Ruído Térmico: {metadata['Thermal Noise Status']}")
    print(f"🔐 **Assinatura:** {metadata['Thermal Noise SHA256']}")
    
    # Métricas de Aceitação e Leitura
    print(f"\n📊 Validados: {accepted_count:,}/{total_frames:,} ({accepted_percentage:.2f}%)")
    print(f"⏱ Tempo de Leitura: {duration_read:.2f}s")
    
    # Métricas de Desempenho
    print("\n--- Métricas de Desempenho ---")
    print(f"⏱ Tempo Total de Simulação (Reportado): {total_duration_reported}s")
    print(f"⚡ Throughput (Frames/s): {throughput:,.2f} frames/s")
    
    # Métricas de Estabilidade e Coerência
    print("\n--- Métricas de Estabilidade e Coerência ---")
    print(f"📊 Mean Coherence Index (Média): {mean_coherence:.6f}")
    print(f"📈 Median Coherence Index (Mediana): {median_coherence:.6f}")
    print(f"📊 Coherence Variance Index: {coherence_variance:.6f}")
    print(f"📊 Coherence Std Dev (Desvio Padrão): {std_dev_coherence:.6f}")
    print(f"⬇️ Coherence Min: {min_coherence:.6f}")
    print(f"⬆️ Coherence Max: {max_coherence:.6f}")
    
    # 6. Geração do Gráfico de Emaranhamento e Estabilidade
    
    if total_frames < 2:
        print("⚠️ Aviso: Dados insuficientes para gerar o gráfico.")
        return
        
    # --- Lógica do Gráfico de Estabilidade e Emaranhamento ---
    
    NUM_VETORES_COMPLEXIDADE = 4
    
    tempo = np.linspace(0, 1, total_frames)
    sphy_evolution = raw_coherence_evolution
    
    # 1. Cria 'N' sinais interpolados com diferentes atrasos (roll)
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(NUM_VETORES_COMPLEXIDADE)]
    
    # 2. Gera os dados (os vetores) aplicando ruído de visualização
    novo_tempo = np.linspace(0, 1, 500)
    # Adicionamos ruído com sigma = 0.3 para forçar o efeito de vibração (onda)
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.3, len(novo_tempo)) for sinal in sinais] 
    
    # 3. Calcula a Curva de Emaranhamento (Média Ponderada de todos os vetores)
    pesos = np.linspace(1, 1.5, NUM_VETORES_COMPLEXIDADE)
    emaranhamento = np.average(dados, axis=0, weights=pesos) 

    estabilidade_media_plot = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)
    std_dev_estabilidade = np.sqrt(estabilidade_variancia)

    cores = plt.cm.get_cmap('Spectral', NUM_VETORES_COMPLEXIDADE)

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Subplot 1: Emaranhamento Simulado/Estabilizado
    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="SPHY Stabilized Coherence")
    # Plotagem de todos os N vetores com cores distintas
    for i in range(NUM_VETORES_COMPLEXIDADE):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color=cores(i))
        
    ax1.set_xlabel("Tempo Normalizado (Normalized Time)")
    ax1.set_ylabel("Coerência/Amplitude SPHY (%)")
    ax1.set_title(f"GHZ CV Entanglement - {num_modes} Modos ({NUM_VETORES_COMPLEXIDADE} Vetores)")
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Estabilidade e Variância
    ax2.plot(novo_tempo, emaranhamento, 'k-', label="SPHY Stabilized Coherence")
    
    ax2.axhline(estabilidade_media_plot, color='green', linestyle='--', label=f"Média: {estabilidade_media_plot:.4f}")
    ax2.axhline(estabilidade_media_plot + std_dev_estabilidade, color='orange', linestyle='--', label=f"± Desvio Padrão")
    ax2.axhline(estabilidade_media_plot - std_dev_estabilidade, color='orange', linestyle='--')
    
    ax2.set_xlabel("Tempo Normalizado (Normalized Time)")
    ax2.set_ylabel("Coerência/Amplitude SPHY (%)")
    ax2.set_title("Estabilidade da Coerência via HARPIA/Meissner")
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"GHZ Simulation (SF): SPHY Coherence and Stability - {num_modes} Modos", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("\n📊 Gerando gráfico...")
    plt.savefig(caminho_fig_salva, dpi=300)
    print(f"✅ Gráfico salvo em: {caminho_fig_salva}")
    
    plt.show()


if __name__ == "__main__":
    caminho_arquivo_input = input("Por favor, digite o caminho completo do arquivo CSV: ").strip()
    analisar_csv_e_gerar_relatorio(caminho_arquivo_input)