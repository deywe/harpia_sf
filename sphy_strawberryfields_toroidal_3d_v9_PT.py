# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# Arquivo: sphy_strawberryfields_toroidal_3d_v9_PT.py 
# Objetivo: TUNELAMENTO QUÂNTICO EM UMA ESTRUTURA TOROIDAL + ENGENHARIA DE CAMPO SPHY (COLETA DE DADOS)
# Autor: deywe@QLZ | Convertido para SF por Gemini AI
# No paradigma SPHY, o qubit é um sensor universal, então o grau de pureza (μ) é uma medida
# das interações, onde 1 representa desconexão total do universo.
# ───────────────────────────────────────────────────────────────

# 1️⃣ Importar módulos necessários
# ASSUMPTION: 'meissner_core.py' deve estar disponível
try:
    from meissner_core import meissner_correction_step 
except ImportError:
    print("⚠️ AVISO: 'meissner_core.py' não encontrado. Usando função dummy.")
    def meissner_correction_step(H, S, C, I, T, psi_state):
        return 0.1, 0.0, psi_state # Retorno Dummy

# ⚛️ Importações Strawberry Fields & CV
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

# Fallback para mean_photon_number
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

# === Configuração da Estrutura Toroidal SPHY ===
TAMANHO_GRADE = 2 
NUM_MODOS = TAMANHO_GRADE * TAMANHO_GRADE # 4 Modos

# Modo Alvo (o "partícula" sujeita ao tunelamento)
MODO_ALVO = 0 

DIRECAO_TUNELAMENTO = 'either' # 'either' = ou aumento ou diminuição excede o limite
LIMITE_TUNELAMENTO = 0.05     

# === Diretório de Log
DIR_LOG = "logs_sphy_toroidal_v9"
os.makedirs(DIR_LOG, exist_ok=True)

BACKEND_MOTOR = "gaussian"
COERENCIA_INICIAL = 90.0 # Valor padrão

# === Configuração e Funções Auxiliares ===

def obter_parametros_usuario():
    """Obtém os parâmetros de simulação do usuário."""
    try:
        num_modos = NUM_MODOS
        print(f"🔢 Número de Modos (Grade {TAMANHO_GRADE}x{TAMANHO_GRADE}): {num_modos}")
        total_frames = int(input("🔁 Total de Tentativas de Tunelamento (Frames) a simular: "))
        
        forca_barreira_input = float(input("🚧 Força da Barreira (0.0 a 1.0): "))
        if not (0.0 <= forca_barreira_input <= 1.0):
             print("❌ A Força da Barreira deve estar entre 0.0 e 1.0.")
             exit(1)
             
        # Converte a força para um ângulo de rotação para o Rgate
        angulo_barreira_theta = forca_barreira_input * np.pi / 2 
        
        return num_modos, total_frames, angulo_barreira_theta
    except ValueError:
        print("❌ Entrada inválida. Por favor, insira números inteiros/decimais.")
        exit(1)

def programa_tunelamento_toroidal(angulo_barreira_theta, angulo_perturbacao_sphy):
    """Define o programa quântico para o tunelamento toroidal."""
    prog = sf.Program(NUM_MODOS)
    with prog.context as q:
        # Preparação do Estado (Estado coerente comprimido em todos os modos)
        for mode in range(NUM_MODOS):
            ops.Sgate(0.5) | q[mode]
            ops.Dgate(0.5, 0) | q[mode]

        # Acoplamento Toroidal (CZ Gates para as conexões)
        # Atenção: Mantendo a estrutura original do circuito.
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
        
        # Barreira (Rgate no modo alvo)
        ops.Rgate(angulo_barreira_theta) | q[MODO_ALVO]
        # Perturbação do Campo SPHY (Ruído/Modulação no modo alvo)
        ops.Rgate(angulo_perturbacao_sphy) | q[MODO_ALVO]
        
        # Aplicando uma perturbação menor nos modos adjacentes
        for mode in [1, 2, 3]:
             ops.Rgate(angulo_perturbacao_sphy / 2) | q[mode]
             
    return prog

def simular_frame(dados_frame):
    """Simula um único frame (tentativa) no processo multiprocessing."""
    frame, num_modes, total_frames, prob_ruido, coerencia_sphy, angulo_barreira_theta = dados_frame
    
    random.seed(os.getpid() * frame) 
    
    angulo_perturbacao_sphy = 0.0
    # Aplica ruído/perturbação aleatória baseada na probabilidade
    if random.random() < prob_ruido:
        angulo_perturbacao_sphy = random.uniform(-np.pi/8, np.pi/8)
    
    programa = programa_tunelamento_toroidal(angulo_barreira_theta, angulo_perturbacao_sphy)
    
    try:
        eng = sf.Engine(BACKEND_MOTOR) 
        resultado = eng.run(programa)
        estado = resultado.state
    except Exception as e:
        return None, None, None, f"\nErro Crítico ao rodar o Motor SF no frame {frame}: {e}"

    # --- COLETA DE VALOR ESPERADO E DADOS DE ESTADO ---
    try:
        numeros_medios_fotons = mean_photon_number(estado.cov(), estado.means())
        
        # Coleta a Matriz de Covariância e o Vetor de Deslocamento para o modo alvo (Plano)
        q_idx = 2 * MODO_ALVO
        cov_alvo = estado.cov()[q_idx:q_idx+2, q_idx:q_idx+2].flatten()
        meios_alvo = estado.means()[q_idx:q_idx+2].flatten()
        
    except Exception as e:
        return None, None, None, f"\nErro ao calcular observáveis no frame {frame}: {e}"
    
    # Proxy para Pauli Z (deslocamento relativo/mudança de população)
    proxy_base_z = 0.25 
    proxies_z = [(n - proxy_base_z) for n in numeros_medios_fotons]

    valor_esperado_z_alvo = proxies_z[MODO_ALVO]
    magnitude_proxy = abs(valor_esperado_z_alvo)

    # Lógica de Aceitação de Tunelamento
    if DIRECAO_TUNELAMENTO == 'decrease':
        resultado_bruto = 1 if valor_esperado_z_alvo < -LIMITE_TUNELAMENTO else 0
    elif DIRECAO_TUNELAMENTO == 'increase':
        resultado_bruto = 1 if valor_esperado_z_alvo > LIMITE_TUNELAMENTO else 0
    else:
        # Tunelamento é bem-sucedido se a magnitude da mudança exceder o limite
        resultado_bruto = 1 if magnitude_proxy > LIMITE_TUNELAMENTO else 0

    estado_ideal = 1

    # === Lógica SPHY/Meissner ===
    # Estes representam os fatores de estabilidade do ambiente/sistema
    H = random.uniform(0.95, 1.0) 
    S = random.uniform(0.95, 1.0) 
    C = coerencia_sphy / 100    # Coerência Atual (normalizada)
    I = abs(H - S)             # Índice de Instabilidade (análogo a Phi(t))
    T = frame                   
    vetor_estado_psi = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] # Placeholder para vetor fase/estado

    try:
        # Aplica o passo de correção da Modulação de Gravidade Quântica (QGM)
        impulso, impacto_fase, vetor_estado_psi = meissner_correction_step(H, S, C, I, T, vetor_estado_psi)
    except Exception as e:
        return None, None, None, f"\nErro Crítico no frame {frame} (Meissner IA): {e}"

    # Calcula a nova coerência baseada no impulso
    delta = impulso * 0.7
    nova_coerencia = min(100, coerencia_sphy + delta)
    ativado = delta > 0 
    aceito = (resultado_bruto == estado_ideal) and ativado
    
    # Entrada de Log
    timestamp_atual = datetime.utcnow().isoformat()
    dados_para_hash = f"{frame}:{resultado_bruto}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{impulso:.4f}:{nova_coerencia:.4f}:{timestamp_atual}"
    assinatura_sha256 = hashlib.sha256(dados_para_hash.encode('utf-8')).hexdigest()

    logs_fase = [round(z, 4) for z in proxies_z] 
    sinal_proxy = '-' if valor_esperado_z_alvo < 0 else '+'
    
    entrada_log = [
        frame, resultado_bruto,
        *logs_fase,
        round(magnitude_proxy, 4), sinal_proxy,
        *cov_alvo, # Vqq, Vqp, Vpq, Vpp
        *meios_alvo, # mu_q, mu_p
        round(H, 4), round(S, 4), round(C, 4), round(I, 4), round(impulso, 4),
        round(nova_coerencia, 4), "✅" if aceito else "❌",
        assinatura_sha256, timestamp_atual
    ]
    return entrada_log, nova_coerencia, (cov_alvo, meios_alvo), None

# --- NOVAS FUNÇÕES DE PLOTAGEM CIENTÍFICA ---

def plot_funcao_wigner(cov_alvo, meios_alvo, nome_arquivo_wigner):
    """Plota a Função de Wigner para o modo alvo no estado final."""
    if cov_alvo is None or meios_alvo is None:
        print("❌ Erro: Dados do estado quântico (Wigner) não disponíveis para plotagem.")
        return

    # Matriz de Covariância 2x2
    cov = np.array([[cov_alvo[0], cov_alvo[1]], [cov_alvo[2], cov_alvo[3]]])
    meios = np.array([meios_alvo[0], meios_alvo[1]])
    
    q_lim = max(3.0, np.max(np.abs(meios)))
    q_grid = np.linspace(-q_lim, q_lim, 100)
    Q, P = np.meshgrid(q_grid, q_grid)
    
    coordenadas = np.vstack([Q.flatten(), P.flatten()]).T
    
    try:
        pdf_wigner = multivariate_normal.pdf(coordenadas, mean=meios, cov=cov)
    except np.linalg.LinAlgError:
        print("⚠️ Erro de Álgebra Linear na Wigner. Plotando estado de vácuo (Cov=I).")
        cov_id = np.identity(2) * 0.5 
        pdf_wigner = multivariate_normal.pdf(coordenadas, mean=[0,0], cov=cov_id)
        
    W = pdf_wigner.reshape(Q.shape)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    contorno = ax.contourf(Q, P, W, 100, cmap='RdBu_r')
    ax.scatter(meios[0], meios[1], marker='x', color='black', s=100, label='Centro ($\mu_q, \mu_p$)')
    
    ax.set_title(f'Função de Wigner do Modo Alvo (Frame Final)', fontsize=14)
    ax.set_xlabel('Quadratura de Posição ($q$)')
    ax.set_ylabel('Quadratura de Momento ($p$)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(contorno, label='Amplitude W(q, p)')
    
    plt.savefig(nome_arquivo_wigner, dpi=300)
    print(f"🖼️ Função de Wigner salva: {nome_arquivo_wigner}")


def plot_histograma_tunelamento(df, limite, nome_arquivo_hist):
    """Plota o Histograma da Magnitude do Proxy de Tunelamento."""
    if df.empty:
        print("❌ Erro: DataFrame vazio para plotagem do Histograma.")
        return

    dados_proxy = df['Proxy_Mag']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(dados_proxy, bins=30, edgecolor='black', alpha=0.7, color='skyblue', 
            label='Magnitude do Proxy de Tunelamento ( |$\\Delta \\bar{n}$| )')
    
    ax.axvline(limite, color='red', linestyle='--', linewidth=2, 
               label=f'Limite de Tunelamento ({limite})')
    
    contagem_sucesso = (dados_proxy >= limite).sum()
    contagem_total = len(dados_proxy)
    taxa_sucesso = 100 * (contagem_sucesso / contagem_total)
    
    ax.text(0.95, 0.90, f'Taxa de Sucesso Total: {taxa_sucesso:.2f}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
            horizontalalignment='right', fontsize=12, color='darkgreen', weight='bold')

    ax.set_title(f'Distribuição de Desempenho (Proxy de Tunelamento) em {contagem_total} Frames', fontsize=14)
    ax.set_xlabel('Magnitude do Proxy de Tunelamento ( |$\\Delta \\bar{n}$| )')
    ax.set_ylabel('Frequência de Ocorrência (Frames)')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    plt.savefig(nome_arquivo_hist, dpi=300)
    print(f"🖼️ Histograma de Tunelamento salvo: {nome_arquivo_hist}")


# === Função Principal de Simulação com Novas Métricas ===

def executar_simulacao_multiprocessing(num_modos, total_frames, angulo_barreira_theta, prob_ruido=1.0, num_processos=4):
    """Executa a simulação completa, calcula todas as métricas e gera os gráficos."""
    print("=" * 60)
    print(f" ⚛️ ONDAS SPHY (SF): Tunelamento Toroidal ({TAMANHO_GRADE}x{TAMANHO_GRADE}) • {total_frames:,} Frames")
    print(f" 🚧 Força da Barreira: {angulo_barreira_theta*180/np.pi:.2f} graus Rgate (Análogo CV)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo_csv = os.path.join(DIR_LOG, f"toroidal_{num_modos}q_log_{timecode}.csv")
    nome_arquivo_fig = os.path.join(DIR_LOG, f"toroidal_{num_modos}q_grafico_2D_{timecode}.png")
    nome_arquivo_wigner = os.path.join(DIR_LOG, f"toroidal_{num_modos}q_FUNCAO_WIGNER_{timecode}.png")
    nome_arquivo_hist = os.path.join(DIR_LOG, f"toroidal_{num_modos}q_HISTOGRAMA_{timecode}.png")


    manager = Manager()
    coerencia_sphy = manager.Value('f', COERENCIA_INICIAL)
    dados_log, evolucao_sphy = manager.list(), manager.list()
    estados_validos = manager.Value('i', 0)
    # Guarda os dados de estado do último frame (Matriz Covariância e Vetor de Meios)
    dados_ultimo_estado = manager.dict({'cov': None, 'means': None}) 

    entradas_frame = [
        (f, num_modos, total_frames, prob_ruido, coerencia_sphy.value, angulo_barreira_theta) 
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Usando {num_processos} processos para a simulação...")
    with Pool(processes=num_processos) as pool:
        for entrada_log, nova_coerencia, dados_estado, erro in tqdm(pool.imap_unordered(simular_frame, entradas_frame),
                                            total=total_frames, desc="⏳ Simulando SPHY Toroidal (SF)"):
            if erro:
                print(f"\n{erro}", file=sys.stderr)
                pool.terminate()
                break
            if entrada_log:
                dados_log.append(entrada_log)
                evolucao_sphy.append(nova_coerencia)
                coerencia_sphy.value = nova_coerencia 
                if entrada_log[-3] == "✅":
                    estados_validos.value += 1
                
                if dados_estado:
                    # Atualiza o último estado quântico coletado
                    dados_ultimo_estado['cov'] = dados_estado[0]
                    dados_ultimo_estado['means'] = dados_estado[1]


    # --- Cálculo de Métricas (Incluindo Pureza, Espremedura e Wigner Máxima) ---
    taxa_aceitacao = 100 * (estados_validos.value / total_frames) if total_frames > 0 else 0
    
    lista_evolucao_sphy = list(evolucao_sphy)
    
    # 1. Preparação e Cálculo das Métricas SPHY Clássicas
    if not lista_evolucao_sphy:
        print("❌ Não há dados para calcular métricas SPHY ou plotar gráfico 2D.")
        return

    evolucao_sphy_np = np.array(lista_evolucao_sphy)
    pontos_tempo = np.linspace(0, 1, len(evolucao_sphy_np))
    
    n_redundancias = 2 
    sinais = [interp1d(pontos_tempo, np.roll(evolucao_sphy_np, i), kind='cubic') for i in range(n_redundancias)]
    novo_tempo = np.linspace(0, 1, 2000)
    
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, n_redundancias)
    estabilidade_tunelamento = np.average(dados, axis=0, weights=pesos) 
    
    media_estabilidade_sphy = np.mean(dados[1]) 
    variancia_estabilidade = np.var(dados[1])
    
    # 2. Cálculo das Métricas Quânticas (Pureza, Espremedura, Wigner Máxima)
    cov_alvo_plana = dados_ultimo_estado['cov']
    pureza = float('nan')
    espremedura_minima = float('nan')
    wigner_maxima = float('nan')
    
    if cov_alvo_plana is not None:
        try:
            # Reconstrói a Matriz de Covariância 2x2 para o Modo Alvo
            V = np.array([[cov_alvo_plana[0], cov_alvo_plana[1]], 
                          [cov_alvo_plana[2], cov_alvo_plana[3]]])
            
            # Pureza (mu = 1 / sqrt(det(2V)))
            det_2V = np.linalg.det(2 * V)
            pureza = 1.0 / np.sqrt(det_2V)
            
            # Espremedura (Autovalor mínimo de V)
            traco_V = np.trace(V)
            det_V = np.linalg.det(V)
            espremedura_minima = 0.5 * (traco_V - np.sqrt(traco_V**2 - 4 * det_V))
            
            # Wigner Máxima (Wmax) no centro do espaço de fase
            wigner_maxima = pureza / np.pi
            
        except np.linalg.LinAlgError:
            print("⚠️ Erro de Álgebra Linear ao calcular Pureza/Espremedura/Wigner. Matriz inválida.")
        except ValueError:
            print("⚠️ Erro no cálculo da Espremedura. Valor negativo sob a raiz.")

    # 3. Imprime as Métricas no Console (Relatório Completo)
    print("\n" + "=" * 60)
    print("           📊 RELATÓRIO DE DESEMPENHO SPHY")
    print("-" * 60)
    print(f"| ✅ Taxa de Sucesso de Tunelamento (SPHY Toroidal): {estados_validos.value}/{total_frames} | **{taxa_aceitacao:.2f}%**")
    print("-" * 60)
    print(f"| ⭐ Estabilidade Média SPHY: {media_estabilidade_sphy:.4f}")
    print(f"| 🌊 Variância da Estabilidade: {variancia_estabilidade:.6f}")
    print("-" * 60)
    # NOVAS MÉTRICAS QUÂNTICAS
    print(f"| ⚛️ Pureza do Estado Final (μ): {pureza:.4f} (Ideal SPHY < 1.0 | QEC Ideal = 1.0)")
    print(f"| 🔬 Espremedura Mínima (λ_min): {espremedura_minima:.4f} (Espremido < 0.5)")
    print(f"| 📈 Wigner Máxima (W_max): {wigner_maxima:.4f} (W_max <= 1/π)")
    print("=" * 60)
    
    # ... (Escrita do CSV)
    cols_fase = [f"Qubit_{i+1}_Fase" for i in range(NUM_MODOS)]
    headers_cov = ["Vqq_0", "Vqp_0", "Vpq_0", "Vpp_0"]
    headers_meios = ["mu_q_0", "mu_p_0"]

    cabecalho = [
        "Frame", "Resultado", 
        *cols_fase,
        "Proxy_Mag", "Proxy_Sinal",
        *headers_cov, 
        *headers_meios,
        "H", "S", "C", "I", "Impulso", "SPHY (%)", "Aceito", 
        "Assinatura_SHA256", "Timestamp"
    ]
    with open(nome_arquivo_csv, mode="w", newline="", encoding="utf-8") as file:
        escritor = csv.writer(file)
        escritor.writerow(cabecalho)
        escritor.writerows(list(dados_log))
    print(f"🧾 CSV salvo: {nome_arquivo_csv}")

    # --- Plotagem de Gráficos ---
    # Plota a Função de Wigner com o estado quântico final
    plot_funcao_wigner(dados_ultimo_estado['cov'], dados_ultimo_estado['means'], nome_arquivo_wigner)
    
    # Plota o Histograma
    df_resultados = pd.DataFrame(list(dados_log), columns=cabecalho)
    plot_histograma_tunelamento(df_resultados, LIMITE_TUNELAMENTO, nome_arquivo_hist)

    # === CÓDIGO DE PLOTAGEM DE ESTABILIDADE 2D ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Gráfico 1: Sinal de Coerência SPHY (Amplitude)
    ax1.set_title("Evolução da Coerência SPHY (Sinal 1: Amplitude) 📈")
    for i in range(n_redundancias): 
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue')  
    ax1.plot(novo_tempo, estabilidade_tunelamento, 'k--', linewidth=2, label="Estabilidade Média Ponderada")
    ax1.set_xlabel("Tempo Normalizado")
    ax1.set_ylabel("Coerência/Amplitude")
    ax1.legend()
    ax1.grid()

    # Gráfico 2: Sinal de Coerência SPHY (Estabilidade)
    ax2.set_title("Evolução da Coerência SPHY (Sinal 2: Estabilidade) 🌊")
    ax2.plot(novo_tempo, dados[1], color='red', alpha=0.7, label='Sinal de Coerência (2)')
    
    ax2.axhline(media_estabilidade_sphy, color='green', linestyle='--', label=f"Média: {media_estabilidade_sphy:.2f}")
    ax2.axhline(media_estabilidade_sphy + np.sqrt(variancia_estabilidade), color='orange', linestyle='--', label=f"± Variância")
    ax2.axhline(media_estabilidade_sphy - np.sqrt(variancia_estabilidade), color='orange', linestyle='--')

    ax2.set_xlabel("Tempo Normalizado")
    ax2.set_ylabel("Coerência/Amplitude")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Simulação de Tunelamento Quântico (SF CV): {total_frames} Tentativas (SPHY Toroidal)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(nome_arquivo_fig, dpi=300)
    print(f"🖼️ Gráfico de Estabilidade 2D salvo: {nome_arquivo_fig}")
    plt.show(block=True) 


if __name__ == "__main__":
    modes, pairs, barrier_theta = obter_parametros_usuario()
    
    executar_simulacao_multiprocessing(num_modos=modes, total_frames=pairs, angulo_barreira_theta=barrier_theta, prob_ruido=1.0, num_processos=4)