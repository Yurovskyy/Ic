# electromagnetic_modeling.py
"""
Módulo para modelagem eletromagnética do sistema WPT.
Referência: Seção 3 do artigo.
"""
import numpy as np

from resistencia.tourkhani_spain import calculate_r_total
from mutua.neumann_spain import calculate_M_analytical_circular
from indutancia.aebicher_spain import calculate_ipt_coil_inductance_approx

def calculate_inductances(individual, g):
    """
    Calcula as auto-indutâncias (Lp, Ls) e a indutância mútua (M).
    """
    
    L_p = calculate_ipt_coil_inductance_approx(individual.variables["S_p"],individual.variables["N_p"],g)
    L_s = calculate_ipt_coil_inductance_approx(individual.variables["S_s"],individual.variables["N_s"],g)
    
    return L_p, L_s

def calculate_mutual_inductance(individual, distancia_bobinas):
    
    M = calculate_M_analytical_circular((np.sqrt(individual.variables["S_p"])/2),(np.sqrt(individual.variables["S_s"])/2),distancia_bobinas)
    
    return M

def calculate_resistances(individual, f, d_0):
    """
    Calcula as resistências AC (Rp, Rs) considerando efeitos de pele e proximidade.
    """
    
    R_pdict = calculate_r_total(individual.variables["S_p"],f,d_0)
    R_p = R_pdict["r_total_per_m"]
    R_sdict = calculate_r_total(individual.variables["S_s"],f,d_0)
    R_s = R_sdict["r_total_per_m"]
    
    return R_p, R_s
