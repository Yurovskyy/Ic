import numpy as np
import math

from resistencia.din_en_iec_2000_sm import calcular_area_strand,calcular_comprimento_fio,calcular_diametro_interno,calcular_resistencia_litz
from indutancia.aebicher_spain import calculate_rectangular_spiral_inductance
from mutua.bivot_sm import calcular_indutancia_mutua

def calcular_resistencias(Parametros_fixos_projeto,individual,Constantes_fisicas):
    """
    Calcula as resistências usando as formulas de Santa Maria.
    """
    
    # Diâmetro externo (m)
    dp = 2*np.sqrt((Parametros_fixos_projeto["A_p"]*Parametros_fixos_projeto["B_p"])/np.pi)       
    d_sp = np.sqrt(individual.variables["S_p"])       # Diâmetro do fio Litz condutor (m)
    N_p = individual.variables["N_p"]       # Número de espiras
    T_dp = Parametros_fixos_projeto["T_dp"] # Espaçamento entre espiras (m)
    d_0 = Parametros_fixos_projeto["d_0"]   # Diâmetro do strand
    n0 = d_sp/d_0 # Número de filamentos, strand, talvez ta errado

    d_ip = calcular_diametro_interno(dp, d_sp, N_p, T_dp)

    Lp = calcular_comprimento_fio(N_p, dp, d_ip)

    A_0 = calcular_area_strand(d_0)

    R_p = calcular_resistencia_litz(
        l=Lp,
        A_str=A_0,
        N_strands=n0,
        n_b=Parametros_fixos_projeto["n_b"],
        rho=Constantes_fisicas["Rho_Cobre"]
    )

    return R_p


def calculate_inductances(Parametros_fixos_projeto,individual):
    """
    Calcula as auto-indutâncias (Lp, Ls) e a indutância mútua (M).
    """
    
    d_sp = math.sqrt(individual.variables["S_p"])
    d_ss = math.sqrt(individual.variables["S_s"])
    
    L_p = calculate_rectangular_spiral_inductance(Parametros_fixos_projeto["A_p"],Parametros_fixos_projeto["B_p"],
                                                  Parametros_fixos_projeto["N_p"],d_sp,Parametros_fixos_projeto["T_dp"],d_sp)
    L_s = calculate_rectangular_spiral_inductance(Parametros_fixos_projeto["A_s"],Parametros_fixos_projeto["B_s"],
                                                  Parametros_fixos_projeto["N_s"],d_ss,Parametros_fixos_projeto["T_ds"],d_sp)
    
    return L_p, L_s

def calculate_mutual_inductance(Parametros_fixos_projeto):
    
    R_p = math.sqrt((Parametros_fixos_projeto["A_p"]*Parametros_fixos_projeto["B_p"])/math.pi)
    R_s = math.sqrt((Parametros_fixos_projeto["A_s"]*Parametros_fixos_projeto["B_s"])/math.pi)
    
    M = calcular_indutancia_mutua(R_p,R_s,Parametros_fixos_projeto["Distancia_bobinas"],Parametros_fixos_projeto["N_p"],Parametros_fixos_projeto["N_s"])
    
    return M
