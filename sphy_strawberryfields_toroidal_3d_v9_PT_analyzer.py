# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# Arquivo: sphy_strawberryfields_toroidal_3d_v9_PT_analisador.py
# Objetivo: ANALISADOR DE TESTE DE DESEMPENHO E GRÁFICOS (WIGNER, HISTOGRAMA, ESTABILIDADE) A PARTIR DE CSV.
# Autor: Gemini AI
# ───────────────────────────────────────────────────────────────

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
import os
import sys

# === Configuração e Variáveis (Deve corresponder ao simulador) ===
MODO_ALVO = 0 
LIMITE_TUNELAMENTO = 0.05 
DIR_LOG = "resultados_analise_sphy_v9"
os.makedirs(DIR_LOG, exist_ok=True)

# Definição das colunas CSV necessárias (Nomes de coluna mantidos em inglês por convenção de dados)
COV_HEADERS = ["Vqq_0", "Vqp_0", "Vpq_0", "Vpp_0"]
MEANS_HEADERS = ["mu_q_0", "mu_p_0"]
PROXY_MAG_COL = "Proxy_Mag"
SPHY_COHERENCE_COL = "SPHY (%)"
ACCEPTED_COL = "Accepted"
FRAME_COL = "Frame"

# === 1. Plotagem da Função de Wigner (Estado Final) ===

def plot_funcao_wigner(cov_alvo, meios_alvo, nome_arquivo_wigner, total_frames):
    """Gera a Função de Wigner (Visualização do Estado CV) para o estado do último frame."""
    if cov_alvo is None or meios_alvo is None or not any(cov_alvo):
        print("❌ Erro: Dados do estado quântico (Wigner) não disponíveis no CSV.")
        return

    # Reconstroi a Matriz 2x2 de Covariância
    cov = np.array([[cov_alvo[0], cov_alvo[1]], [cov_alvo[2], cov_alvo[3]]])
    # Vetor de Deslocamento [mu_q, mu_p]
    meios = np.array([meios_alvo[0], meios_alvo[1]])
    
    q_lim = max(3.0, np.max(np.abs(meios))) + 1.0 
    q_grid = np.linspace(-q_lim, q_lim, 100)
    Q, P = np.meshgrid(q_grid, q_grid)
    coordenadas = np.vstack([Q.flatten(), P.flatten()]).T
    
    try:
        # A Função de Wigner para o estado Gaussiano é modelada pela PDF multivariada.
        pdf_wigner = multivariate_normal.pdf(coordenadas, mean=meios, cov=cov)
    except np.linalg.LinAlgError:
        print("⚠️ Erro de Álgebra Linear na Wigner. Matriz de covariância singular.")
        return
        
    W = pdf_wigner.reshape(Q.shape)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    contorno = ax.contourf(Q, P, W, 100, cmap='RdBu_r')
    ax.scatter(meios[0], meios[1], marker='x', color='black', s=100, label='Centro ($\mu_q, \mu_p$)')
    
    ax.set_title(f'Função de Wigner do Modo Alvo (Frame Final: {total_frames})', fontsize=14)
    ax.set_xlabel('Quadratura de Posição ($q$)')
    ax.set_ylabel('Quadratura de Momento ($p$)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(contorno, label='Amplitude W(q, p)')
    
    plt.savefig(nome_arquivo_wigner, dpi=300)
    plt.show(block=False) # Mostra o gráfico sem bloquear o restante do script
    print(f"🖼️ Função de Wigner salva: {nome_arquivo_wigner}")


# === 2. Plotagem do Histograma de Desempenho ===

def plot_histograma_tunelamento(df, limite, nome_arquivo_hist, total_frames):
    """Gera o Histograma da Magnitude do Proxy de Tunelamento (|Delta n_barra|)."""
    if df.empty: return

    dados_proxy = df[PROXY_MAG_COL].astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(dados_proxy, bins=30, edgecolor='black', alpha=0.7, color='skyblue', 
            label='Magnitude do Proxy de Tunelamento ( |$\\Delta \\bar{n}$| )')
    
    ax.axvline(limite, color='red', linestyle='--', linewidth=2, 
               label=f'Limite de Tunelamento ({limite})')
    
    contagem_sucesso = (dados_proxy >= limite).sum()
    taxa_sucesso = 100 * (contagem_sucesso / total_frames)
    
    ax.text(0.95, 0.90, f'Taxa de Sucesso Total: {taxa_sucesso:.2f}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
            horizontalalignment='right', fontsize=12, color='darkgreen', weight='bold')

    ax.set_title(f'Distribuição de Desempenho em {total_frames} Frames', fontsize=14)
    ax.set_xlabel('Magnitude do Proxy de Tunelamento ( |$\\Delta \\bar{n}$| )')
    ax.set_ylabel('Frequência de Ocorrência (Frames)')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    plt.savefig(nome_arquivo_hist, dpi=300)
    plt.show(block=False)
    print(f"🖼️ Histograma de Tunelamento salvo: {nome_arquivo_hist}")


# === 3. Plotagem da Evolução de Estabilidade SPHY (2D) ===

def plot_evolucao_sphy(df, nome_arquivo):
    """Gera o gráfico 2D de estabilidade SPHY ao longo do tempo (baseado na Coerência SPHY)."""
    lista_evolucao_sphy = df[SPHY_COHERENCE_COL].astype(float).tolist()
    if not lista_evolucao_sphy: return

    evolucao_sphy = np.array(lista_evolucao_sphy)
    pontos_tempo = np.linspace(0, 1, len(evolucao_sphy))
    
    # Reproduz a lógica de interpolação e redundância do simulador
    n_redundancias = 2 
    sinais = [interp1d(pontos_tempo, np.roll(evolucao_sphy, i), kind='cubic') for i in range(n_redundancias)]
    novo_tempo = np.linspace(0, 1, 2000)
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, n_redundancias)
    estabilidade_tunelamento = np.average(dados, axis=0, weights=pesos)

    media_estabilidade_2 = np.mean(dados[1]) 
    variancia_estabilidade_2 = np.var(dados[1])

    total_frames = len(lista_evolucao_sphy)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Gráfico 1: Sinal de Coerência SPHY (Amplitude)
    ax1.set_title("Evolução da Coerência SPHY (Sinal 1: Amplitude)")
    for i in range(n_redundancias):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue')  
    ax1.plot(novo_tempo, estabilidade_tunelamento, 'k--', linewidth=2, label="Estabilidade Média Ponderada")
    ax1.set_xlabel("Tempo Normalizado")
    ax1.set_ylabel("Coerência/Amplitude")
    ax1.legend()
    ax1.grid()

    # Gráfico 2: Sinal de Coerência SPHY (Estabilidade)
    ax2.set_title("Evolução da Coerência SPHY (Sinal 2: Estabilidade)")
    ax2.plot(novo_tempo, dados[1], color='red', alpha=0.7, label='Sinal de Coerência (2)')
    
    ax2.axhline(media_estabilidade_2, color='green', linestyle='--', label=f"Média: {media_estabilidade_2:.2f}")
    ax2.axhline(media_estabilidade_2 + np.sqrt(variancia_estabilidade_2), color='orange', linestyle='--', label=f"± Variância")
    ax2.axhline(media_estabilidade_2 - np.sqrt(variancia_estabilidade_2), color='orange', linestyle='--')

    ax2.set_xlabel("Tempo Normalizado")
    ax2.set_ylabel("Coerência/Amplitude")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Análise de Tunelamento Quântico (SF CV): {total_frames} Frames (Estabilidade SPHY)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(nome_arquivo, dpi=300)
    plt.show(block=False)
    print(f"🖼️ Gráfico de Estabilidade 2D salvo: {nome_arquivo}")

# === Função Principal de Análise ===

def rodar_analise(caminho_arquivo_csv):
    """Carrega o CSV, calcula métricas e gera os gráficos."""
    try:
        df = pd.read_csv(caminho_arquivo_csv)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo CSV não encontrado: {caminho_arquivo_csv}")
        return
    except Exception as e:
        print(f"❌ Erro ao ler o CSV: {e}")
        return

    print("=" * 60)
    print(f" 🔍 Iniciando Análise SPHY-CV para: {os.path.basename(caminho_arquivo_csv)}")
    print("=" * 60)
    
    # 1. CÁLCULO DE MÉTRICAS 
    total_frames = len(df)
    
    # Taxa de Sucesso (Tunelamento)
    frames_aceitos = df[df[ACCEPTED_COL] == '✅']
    taxa_sucesso = 100 * (len(frames_aceitos) / total_frames)
    
    # Estabilidade SPHY
    dados_sphy = df[SPHY_COHERENCE_COL].astype(float)
    media_estabilidade = dados_sphy.mean()
    variancia_estabilidade = dados_sphy.var()
    
    # Métricas Quânticas (Último Frame)
    cov_alvo_plana = df.iloc[-1][COV_HEADERS].values.astype(float)
    pureza, espremedura_minima, wigner_maxima = float('nan'), float('nan'), float('nan')
    
    try:
        V = np.array([[cov_alvo_plana[0], cov_alvo_plana[1]], [cov_alvo_plana[2], cov_alvo_plana[3]]])
        det_2V = np.linalg.det(2 * V)
        pureza = 1.0 / np.sqrt(det_2V)
        
        traco_V = np.trace(V)
        det_V = np.linalg.det(V)
        espremedura_minima = 0.5 * (traco_V - np.sqrt(traco_V**2 - 4 * det_V))
        
        wigner_maxima = pureza / np.pi
    except Exception:
        pass # Mantém NaN se o cálculo falhar
        

    # 2. IMPRESSÃO DO RELATÓRIO DE MÉTRICAS
    
    print("      📊 RELATÓRIO DE DESEMPENHO SPHY-CV")
    print("-" * 60)
    print(f"| Total de Frames Analisados: {total_frames:,}")
    print(f"| Taxa de Sucesso (Túnel Aceito): {len(frames_aceitos)}/{total_frames} | **{taxa_sucesso:.2f}%**")
    print("-" * 60)
    print(f"| ⭐ Estabilidade SPHY Média: {media_estabilidade:.4f}")
    print(f"| 🌊 Variância da Estabilidade: {variancia_estabilidade:.6f}")
    print("-" * 60)
    print(f"| ⚛️ Pureza Final (μ): {pureza:.4f}")
    print(f"| 🔬 Espremedura Mínima (λ_min): {espremedura_minima:.4f}")
    print(f"| 📈 Wigner Máxima (W_max): {wigner_maxima:.4f}")
    print("=" * 60)
    
    
    # 3. GERAÇÃO DE NOMES DE ARQUIVO
    nome_base = os.path.splitext(os.path.basename(caminho_arquivo_csv))[0]
    nome_arquivo_wigner = os.path.join(DIR_LOG, f"{nome_base}_WIGNER_ANALISE.png")
    nome_arquivo_hist = os.path.join(DIR_LOG, f"{nome_base}_HISTOGRAMA_ANALISE.png")
    nome_arquivo_estabilidade = os.path.join(DIR_LOG, f"{nome_base}_ESTABILIDADE_ANALISE.png")


    # 4. PLOTAGEM E EXIBIÇÃO
    
    # A. FUNÇÃO DE WIGNER
    meios_alvo = df.iloc[-1][MEANS_HEADERS].values.astype(float)
    plot_funcao_wigner(cov_alvo_plana, meios_alvo, nome_arquivo_wigner, total_frames)
    
    # B. HISTOGRAMA DE TUNELAMENTO
    plot_histograma_tunelamento(df, LIMITE_TUNELAMENTO, nome_arquivo_hist, total_frames)
    
    # C. EVOLUÇÃO DE ESTABILIDADE SPHY
    plot_evolucao_sphy(df, nome_arquivo_estabilidade)

    # Bloqueia a execução APENAS no final para manter as janelas do Matplotlib abertas
    print("\nVisualizando gráficos... Feche as janelas do Matplotlib para finalizar.")
    plt.show(block=True) 


if __name__ == "__main__":
    
    arquivo_csv = None
    
    if len(sys.argv) > 1:
        arquivo_csv = sys.argv[1]
    
    if arquivo_csv is None:
        print("\n--- ANALISADOR CSV SPHY ---")
        arquivo_csv = input("Por favor, insira o caminho completo ou nome do arquivo CSV de log: ")
        
    if not arquivo_csv:
        print("❌ Operação cancelada. Nenhum caminho de arquivo fornecido.")
        sys.exit(1)
        
    rodar_analise(arquivo_csv)