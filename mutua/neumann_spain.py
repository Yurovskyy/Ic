import numpy as np
from scipy.special import ellipk, ellipe # Para as integrais elípticas
try:
    from ..config import Constantes_fisicas, Parametros_numericos
except ImportError:
    # Suporte a execução direta do arquivo (sem pacote pai)
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import Constantes_fisicas, Parametros_numericos


def calculate_M_analytical_circular(r_a: float, r_b: float, distancia_bobinas: float) -> float:
    """
    Calcula a indutância mútua entre duas espiras circulares coaxiais
    usando a fórmula analítica exata com integrais elípticas.

    Args:
        r_a (float): Raio da primeira espira (a) [m]
        r_b (float): Raio da segunda espira (b) [m]
        distancia_bobina (float): Distância axial entre as espiras (d) [m]

    Returns:
        m (float): A indutância mútua [H].
    """
    if distancia_bobinas == 0 and r_a == r_b:
        # Caso especial: autoindutância de uma espira fina (tende ao infinito), não Mútua.
        # Mas para M, se d=0, o campo é no plano.
        # A fórmula diverge se os fios se tocam. Retorna um valor simbólico.
        return float('inf')
        
    # Parâmetro k^2 da fórmula
    k_squared = (4 * r_a * r_b) / ((r_a + r_b)**2 + distancia_bobinas**2)
    
    # k é a raiz quadrada do módulo para as funções elípticas
    k = np.sqrt(k_squared)

    # K(k) e E(k) são as integrais elípticas completas
    K_k = ellipk(k_squared)
    E_k = ellipe(k_squared)
    
    # Aplica a fórmula completa
    term1 = (2 / k - k) * K_k
    term2 = (2 / k) * E_k
    
    m = Constantes_fisicas["Mu_0"] * np.sqrt(r_a * r_b) * (term1 - term2)
    
    return m

# ==============================================================================
# 2. IMPLEMENTAÇÃO NUMÉRICA (MÉTODO USADO NO SEU CÓDIGO, ADAPTADO)
# ==============================================================================

# Funções auxiliares (as mesmas da primeira resposta, mas para espiras circulares)
def create_circular_loop_points(radius: float, num_segments: int, center_z: float = 0.0) -> np.ndarray:
    points = []
    delta_angle = 2 * np.pi / num_segments
    for i in range(num_segments):
        angle = i * delta_angle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append(np.array([x, y, center_z]))
    return np.array(points)

def get_segment_vectors_and_midpoints(loop_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_points = len(loop_points)
    segment_vectors, segment_midpoints = [], []
    for i in range(num_points):
        p1 = loop_points[i]
        p2 = loop_points[(i + 1) % num_points]
        dl = p2 - p1
        midpoint = (p1 + p2) / 2.0
        segment_vectors.append(dl)
        segment_midpoints.append(midpoint)
    return np.array(segment_vectors), np.array(segment_midpoints)

def calculate_M_numerical_circular(r_a: float, r_b: float, distancia_bobina: float, numero_segmentos: int) -> float:
    """
    Calcula a indutância mútua entre duas espiras circulares coaxiais
    usando a aproximação numérica da fórmula de Neumann.

    Args:
        r_a (float): Raio da primeira espira (a) [m]
        r_b (float): Raio da segunda espira (b) [m]
        distancia_bobina (float): Distância axial entre as espiras (d) [m]
        numero_segmentos (int): Número de segmentos para o loop circular iterar

    Returns:
        m (float): A indutância mútua [H].
    """
    # Cria as geometrias das duas espiras
    points1 = create_circular_loop_points(r_a, numero_segmentos, center_z=0.0)
    points2 = create_circular_loop_points(r_b, numero_segmentos, center_z=distancia_bobina)

    # Discretiza as espiras em segmentos dl e pontos médios
    segments1, midpoints1 = get_segment_vectors_and_midpoints(points1)
    segments2, midpoints2 = get_segment_vectors_and_midpoints(points2)
    
    # Aplica a soma de Neumann
    M_numerical = 0.0
    for i in range(numero_segmentos):
        dl1, mid1 = segments1[i], midpoints1[i]
        for j in range(numero_segmentos):
            dl2, mid2 = segments2[j], midpoints2[j]
            r_vec = mid1 - mid2
            r = np.linalg.norm(r_vec)
            if r > Parametros_numericos["Eps_distance"]:
                M_numerical += np.dot(dl1, dl2) / r
    
    return (Constantes_fisicas["Mu_0"] / (4 * np.pi)) * M_numerical

# ==============================================================================
# 3. COMPARAÇÃO E VALIDAÇÃO
# ==============================================================================

if __name__ == '__main__':
    # --- Parâmetros do Teste ---
    r1 = 0.1   # Raio da espira 1 (10 cm)
    r2 = 0.15  # Raio da espira 2 (15 cm)
    d = 0.05   # Distância entre as espiras (5 cm)
    
    # Parâmetro de precisão para o método numérico
    # Quanto maior, mais preciso (e mais lento) será o cálculo.
    n_segments = 400

    print("--- Validação da Implementação da Indutância Mútua ---")
    print(f"Parâmetros: Raio 1 = {r1} m, Raio 2 = {r2} m, Distância = {d} m")
    print("-" * 55)

    # Calcula usando a fórmula analítica (gabarito)
    M_analytical = calculate_M_analytical_circular(r1, r2, d)
    print(f"Resultado (Fórmula Analítica Exata): M = {M_analytical:.6e} H")

    # Calcula usando a implementação numérica
    M_numerical = calculate_M_numerical_circular(r1, r2, d, Parametros_numericos["Numero_segmentos"])
    print(f"Resultado (Implementação Numérica):   M = {M_numerical:.6e} H (com {n_segments} segmentos)")
    
    print("-" * 55)

    # Calcula e exibe o erro percentual
    error_percent = abs((M_numerical - M_analytical) / M_analytical) * 100
    print(f"Erro Percentual: {error_percent:.4f}%")
    
    if error_percent < 0.1:
        print("\nVALIDAÇÃO BEM-SUCEDIDA: O resultado numérico é extremamente próximo do analítico.")
    else:
        print("\nAVISO: O erro é maior que o esperado. Verifique os parâmetros.")
