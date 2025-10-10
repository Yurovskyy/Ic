"""
Script principal para executar o algoritmo NSGA-II para otimização de WPT
utilizando a biblioteca Pymoo.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# Importa as constantes e as funções de modelagem dos arquivos anteriores
from config import Limites_variaveis, Restricoes, Parametros_algoritmo, Parametros_fixos_projeto
from modelagem import calculate_inductances
from circuito.wpt_system import secant_method_for_frequency

# --- Definição do Problema para Pymoo ---
class WPTProblem(Problem):
    """
    Esta classe define o problema de otimização do WPT no formato que o pymoo entende.
    Herda da classe pymoo.core.problem.Problem.
    """
    def __init__(self):
        # Mapeia as variáveis para garantir a ordem consistente
        self.var_keys = ['S_p', 'N_p', 'S_s', 'N_s', 'V_p', 'V_s']
        
        # Extrai os limites inferiores (xl) e superiores (xu) do arquivo de configuração
        xl = np.array([Limites_variaveis[key][0] for key in self.var_keys])
        xu = np.array([Limites_variaveis[key][1] for key in self.var_keys])

        super().__init__(
            n_var=len(self.var_keys),     # Número de variáveis de decisão
            n_obj=2,                      # Número de funções objetivo (Vol_p, Vol_s)
            n_constr=7,                   # Número de restrições
            xl=xl,                        # Limites inferiores das variáveis
            xu=xu                         # Limites superiores das variáveis
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Este é o método central que avalia um conjunto de soluções.
        'x' é um array NumPy onde cada linha é um indivíduo.
        'out' é um dicionário que deve ser preenchido com os resultados 'F' (objetivos) e 'G' (restrições).
        """
        n_solutions = x.shape[0]
        
        # Inicializa os arrays de saída para objetivos (F) e restrições (G)
        f = np.full((n_solutions, self.n_obj), np.inf)
        g = np.full((n_solutions, self.n_constr), np.inf)

        # Loop para avaliar cada solução (cada linha em 'x')
        for i in range(n_solutions):
            # Cria um objeto 'Indivíduo' temporário para compatibilidade com as funções existentes
            # Isso encapsula a lógica de avaliação de forma limpa
            variables = {key: val for key, val in zip(self.var_keys, x[i, :])}
            
            # --- Etapa 2: Avaliação da Solução (Referência: Seção 4 do artigo) ---
            
            # Etapa 2.1: Cálculo das indutâncias
            # Este objeto temporário facilita a passagem de parâmetros
            temp_individual = type('obj', (object,), {'variables': variables})()
            temp_individual.L_p, temp_individual.L_s, temp_individual.M = calculate_inductances(temp_individual)
            
            # Etapa 2.2: Encontrar frequência de operação via método da secante
            frequency_hz, params = secant_method_for_frequency(temp_individual)
            
            # Se o método não convergir, a solução é infactível e já está penalizada com 'np.inf'
            if frequency_hz is None or params is None:
                continue
                
            # --- Etapa 2.3: Cálculo dos Objetivos e Restrições ---
            
            # (Eq. 24, 25) Cálculo dos volumes de cobre (Funções Objetivo)
            # ? corrigir aki
            len_p = 2 * (Parametros_fixos_projeto['A_p'] + Parametros_fixos_projeto['B_p']) * variables['N_p']
            len_s = 2 * (Parametros_fixos_projeto['A_s'] + Parametros_fixos_projeto['B_s']) * variables['N_s']
            vol_cu_p = variables['S_p'] * len_p
            vol_cu_s = variables['S_s'] * len_s
            
            f[i, 0] = vol_cu_p
            f[i, 1] = vol_cu_s
            
            # Cálculo das Restrições (no formato g(x) <= 0)
            # Referência: Tabela 8 do artigo
            
            # Restrição 1: Frequência Mínima (79 - f_kHz <= 0)
            g[i, 0] = Restricoes['frequency_kHz'][0] - (frequency_hz / 1000.0)
            
            # Restrição 2: Frequência Máxima (f_kHz - 90 <= 0)
            g[i, 1] = (frequency_hz / 1000.0) - Restricoes['frequency_kHz'][1]
            
            # Restrição 3: Eficiência Mínima (0.95 - eff <= 0)
            g[i, 2] = Restricoes['efficiency_min'] - params['efficiency']
            
            # Restrição 4: Tensão Máx. no Capacitor Primário (VC_p - 3.5kV <= 0)
            g[i, 3] = params['VC_p'] - (Restricoes['V_capacitor_max_kV'] * 1000.0)
            
            # Restrição 5: Tensão Máx. no Capacitor Secundário (VC_s - 3.5kV <= 0)
            g[i, 4] = params['VC_s'] - (Restricoes['V_capacitor_max_kV'] * 1000.0)
            
            # Restrição 6: Densidade de Corrente Máx. no Primário (J_p - 4 <= 0)
            current_density_p = params['I_p'] / variables['S_p']
            g[i, 5] = current_density_p - Restricoes['current_density_max']
            
            # Restrição 7: Densidade de Corrente Máx. no Secundário (J_s - 4 <= 0)
            current_density_s = params['I_s'] / variables['S_s']
            g[i, 6] = current_density_s - Restricoes['current_density_max']

        # Define os resultados no dicionário de saída do Pymoo
        out["F"] = f
        out["G"] = g

# --- Execução do Algoritmo ---
if __name__ == '__main__':
    # 1. Instancia o problema
    problem = WPTProblem()
    
    # 2. Instancia o algoritmo NSGA-II
    # Define os operadores de cruzamento e mutação conforme descrito no artigo
    algorithm = NSGA2(
        pop_size=Parametros_algoritmo["Tamanho_populacao"],
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=Parametros_algoritmo["Probabilidade_crossover"], eta=15), # SBX é o crossover da Eq. 53-55
        mutation=PM(prob=Parametros_algoritmo["Probabilidade_mutacao"], eta=20),    # PM é a mutação polinomial
        eliminate_duplicates=True
    )
    
    # 3. Define o critério de parada (número de gerações)
    termination = get_termination("n_gen", Parametros_algoritmo["Maximo_geracoes"])
    
    # 4. Executa a otimização
    print("Iniciando o algoritmo NSGA-II com Pymoo para otimização de WPT...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True  # Mostra o progresso a cada geração
    )
    
    print("\nOtimização concluída.")
    
    # 5. Processa e exibe os resultados
    print("\n--- Frente de Pareto Final ---")
    
    if res.F is not None and len(res.F) > 0:
        # Extrai as variáveis (X) e os objetivos (F) da frente de Pareto
        pareto_solutions = res.X
        pareto_objectives = res.F

        for i in range(len(pareto_solutions)):
            print(f"\nSolução {i+1}:")
            print(f"  - Objetivos (Vol_p, Vol_s): ({pareto_objectives[i, 0]:.4f}, {pareto_objectives[i, 1]:.4f})")
            
            solution_vars = {key: val for key, val in zip(problem.var_keys, pareto_solutions[i, :])}
            for key, val in solution_vars.items():
                print(f"  - {key}: {val:.2f}")

        # 6. Visualização da Frente de Pareto
        plt.figure(figsize=(10, 6))
        plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], s=40, facecolors='none', edgecolors='blue', label='Soluções Ótimas de Pareto')
        plt.title('Frente de Pareto - Otimização com Pymoo para WPT')
        plt.xlabel('Volume de Cobre Primário ($Vol_{cup}$)')
        plt.ylabel('Volume de Cobre Secundário ($Vol_{cus}$)')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Nenhuma solução factível encontrada na frente de Pareto.")
