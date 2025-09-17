# config.py
"""
Arquivo de configuração para o otimizador WPT NSGA-II.
Contém constantes baseadas no artigo 'Optimal design of a Low-Cost SAE JA2954 compliant WPT system'.
"""

import numpy as np

# ===========================================================================
# Constantes Físicas e Materiais
# ===========================================================================
mu_0 = 4 * np.pi * 1e-7  # (μ0) Permeabilidade do vácuo [H/m]
rho_cobre = 1.68e-8      # Resistividade do cobre (@ 20°C) [Ohm*m]

# ===========================================================================
# Parâmetros Fixos de Projeto (Baseado na Tabela 2 do artigo)
# ===========================================================================
# Estes são os parâmetros definidos pelo padrão SAE J2954 para o caso de
# uma bobina WPT3Z3 e não são modificados durante a otimização.
p_d = 11000.0  # (PD) Potência de projeto [W]

# Dimensões geométricas das bobinas (para WPT3Z3)
PRIMARY_COIL_DIMENSIONS = {
    'a_p': 0.65,  # Lado 'a' da bobina primária [m]
    'b_p': 0.5   # Lado 'b' da bobina primária [m]
}
SECONDARY_COIL_DIMENSIONS = {
    'a_s': 0.38,  # Lado 'a' da bobina secundária [m]
    'b_s': 0.38   # Lado 'b' da bobina secundária [m]
}
# A distância entre indutores (0.25m) é usada nos cálculos de indutância mútua.
distancia_indutor = 0.25


# ===========================================================================
# Limites das Variáveis de Otimização (Baseado na Tabela 3 do artigo)
# ===========================================================================
# Estas são as variáveis que o algoritmo NSGA-II irá ajustar para encontrar
# a solução ótima. Cada variável tem um valor mínimo e máximo definido.
VAR_LIMITS = {
    'S_p': (1, 80),      # (Sp) Seção do condutor primário [mm^2]
    'N_p': (1, 20),      # (Np) Número de espiras do primário
    'S_s': (1, 80),      # (Ss) Seção do condutor secundário [mm^2]
    'N_s': (1, 20),      # (Ns) Número de espiras do secundário
    'V_p': (300, 600),   # (Vp) Tensão de entrada [V]
    'V_s': (300, 600),   # (Vs) Tensão de saída [V]
}

# ===========================================================================
# Restrições do Problema de Otimização (Baseado na Tabela 8 do artigo)
# ===========================================================================
# As soluções geradas pelo otimizador devem respeitar estas restrições
# para serem consideradas válidas (viáveis).
CONSTRAINTS = {
    'frequency_kHz': (79.0, 90.0),      # Faixa de frequência permitida pela Norma
    'efficiency_min': 0.95,             # Eficiência mínima de 95%
    'V_capacitor_max_kV': 3.5,          # Tensão máxima nos capacitores [KV]
    'current_density_max': 4.0,         # Densidade de corrente máxima [A/mm^2]
    # Outras restrições como estabilidade (Qp > ...) e distribuição de perdas
    # são implementadas diretamente nas funções de avaliação.
}
k_penalizacao = 2e9 # Constante de penalidade (K) para soluções que violam as restrições.

# ===========================================================================
# Parâmetros do Algoritmo Genético NSGA-II
# ===========================================================================
POPULATION_SIZE = 100
MAX_GENERATIONS = 200
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 1.0 / len(VAR_LIMITS) # Probabilidade de mutação por variável

# ===========================================================================
# Parâmetros de Métodos Numéricos Auxiliares
# ===========================================================================
# Parâmetros para o Método da Secante (usado para encontrar a frequência correta)
SECANT_METHOD_TOLERANCE = 1.0  # Tolerância de potência em Watts para convergência
SECANT_METHOD_MAX_ITER = 50

# ===========================================================================
# Símbolos e Variáveis do Circuito (Documentação Centralizada)
# ===========================================================================
# Estes são placeholders para as variáveis calculadas durante a modelagem
# eletromagnética e a análise do circuito para cada indivíduo da população.
# A nomenclatura segue os símbolos do artigo, convertidos para snake_case.
# ---------------------------------------------------------------------------

# --- Elétricas ---
v_p = None      # (Vp) Tensão de entrada primária [V]
v_s = None      # (Vs) Tensão de saída secundária [V]
i_p = None      # (Ip) Corrente primária [A]
i_s = None      # (Is) Corrente secundária [A]
l_p = None      # (Lp) Indutância primária [H]
l_s = None      # (Ls) Indutância secundária [H]
c_p = None      # (Cp) Capacitor ressonante primário [F]
c_s = None      # (Cs) Capacitor ressonante secundário [F]
r_p = None      # (Rp) Resistência primária [Ohm]
r_s = None      # (Rs) Resistência secundária [Ohm]
r_l = None      # (RL) Resistência que modela a bateria [Ohm]
m = None        # (M) Indutância mútua [H]
w = None        # (w) Frequência angular [rad/s]
f = None        # (f) Frequência de operação [Hz]
p_s = None      # (Ps) Potência entregue ao secundário [W]
z_t = None      # (Zt) Impedância total equivalente [Ohm]
v_cp = None     # (VCp) Tensão no capacitor primário [V]
v_cs = None     # (VCs) Tensão no capacitor secundário [V]
q_p = None      # (Qp) Fator de qualidade primário [-]
q_s = None      # (Qs) Fator de qualidade secundário [-]
eta = None      # (η) Eficiência [-]
v_bat = None    # (Vbat) Tensão da bateria [V]
k = None        # (k) Coeficiente de acoplamento [-]
b = None        # (B) Densidade de fluxo magnético [T]

# --- Geométricas e Físicas ---
n_p = None      # (Np) Número de espiras do primário [-]
n_s = None      # (Ns) Número de espiras do secundário [-]
s_p = None      # (Sp) Seção do condutor primário [mm^2]
s_s = None      # (Ss) Seção do condutor secundário [mm^2]
a_p = PRIMARY_COIL_DIMENSIONS['a_p']   # Lado 'a' da bobina primária [m]
b_p = PRIMARY_COIL_DIMENSIONS['b_p']   # Lado 'b' da bobina primária [m]
a_s = SECONDARY_COIL_DIMENSIONS['a_s'] # Lado 'a' da bobina secundária [m]
b_s = SECONDARY_COIL_DIMENSIONS['b_s'] # Lado 'b' da bobina secundária [m]
d = None        # (d) Diâmetro [m]
d_0 = None      # (d0) Diâmetro do fio (strand) [mm]
vol_cu_p = None   # (Volcup) Volume de cobre no primário [m^3]
vol_cu_s = None   # (Volcus) Volume de cobre no secundário [m^3]
dl_p = None     # (dlp) Elemento infinitesimal no contorno da bobina primária [m]
dl_s = None     # (dls) Elemento infinitesimal no contorno da bobina secundária [m]
dist_r = None   # (r) Distância entre elementos de corrente [m]
r_dc = None     # (Rdc) Resistência DC [Ohm]
r_skin = None   # (Rskin) Resistência por efeito pelicular [Ohm]
r_prox = None   # (Rprox) Resistência por efeito de proximidade [Ohm]
delta_skin = None # (δ) Profundidade de penetração [m]
h = None        # (H) Intensidade do campo magnético [A/m]
psi1_xi = None  # (ψ1(ξ)) Conjunto de funções de Bessel para R_skin [-]
psi2_xi = None  # (ψ2(ξ)) Conjunto de funções de Bessel para R_prox [-]

# --- Otimização e Algoritmo ---
j_p = None      # (Jp) Função objetivo primária (volume de cobre) [m^3]
j_s = None      # (Js) Função objetivo secundária (volume de cobre) [m^3]
xi = None       # (xi) Vetor de decisão [-]
j_xi = None     # (J(xi)) Vetor objetivo [-]
pais_j_i = None   # (pJ,i) Conjunto de pais
filho_c = None    # (ci) Filho gerado por crossover
filho_m = None    # (mi) Filho gerado por mutação
offspring_o = None # (O) Conjunto de descendentes (filhos)
beta_k = None     # (βk) Fator de espalhamento no crossover (SBX) [-]
crowding_dx_j = None # (dX,j) Distância de aglomeração (crowding distance) [-]
