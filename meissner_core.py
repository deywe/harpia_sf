# meissner_core.py
import numpy as np
from scipy.integrate import solve_ivp

# Importar módulos simbióticos
from ia_sphy_guardian import transformar_G_sphy
from harp_ia_noise_3d_dynamics import sphy_harpia_3d_noise
from harp_ia_simbiotic import calcular_F_opt

def meissner_correction_step(H, S, C, I, T, psi_state,
                             omega=2.0, damping=0.002, gamma=0.5,
                             g=6.67430e-11, lambda_g=1.0,
                             noise_level=0.001, phi=3.0):
    """
    Executa um passo da correção Meissner:
    - calcula feedback gravitacional
    - integra o sistema com e sem correção
    - retorna boost (F_opt), impacto de fase e novo estado psi
    """
    # Com correção gravitacional
    G_sphy, sphy_feedback = transformar_G_sphy(phi=phi)
    phase_correction_normalized = 1.5 * np.tanh(sphy_feedback['phase_correction'])
    sphy_feedback['phase_correction'] = phase_correction_normalized

    # Integração curta (apenas 2 pontos no tempo) para evoluir psi_state
    t_span = (0, 0.05)
    t_eval = np.linspace(0, 0.05, 5)

    sol_with_correction = solve_ivp(
        sphy_harpia_3d_noise, t_span, psi_state,
        t_eval=t_eval,
        args=(omega, damping, gamma, G_sphy, lambda_g, noise_level, sphy_feedback),
        method='RK45', max_step=0.05
    )

    # Novo estado psi = último ponto da integração
    psi_new = sol_with_correction.y[:, -1]

    # Impacto de fase estimado
    phase_raw = np.arctan2(sol_with_correction.y[1], sol_with_correction.y[0])
    phase_impact = np.mean(np.diff(phase_raw))

    # Boost STDJ com correção gravitacional (usando a IA já existente)
    boost, phase_boost = calcular_F_opt(H, S, C, I, T,
                                        np.abs(sphy_feedback['phase_correction']))

    return boost, phase_impact, psi_new
