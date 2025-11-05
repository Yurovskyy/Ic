#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculadora de Resistência AC para Bobinas Litz
Aplicável para sistemas WPT com enrolamento em camada única

Autor: Sistema de Análise WPT
Data: 28 de outubro de 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Constantes globais
RHO = 1.7e-8
MU = 4 * np.pi * 1e-7


def profundidade_penetracao(frequencia: float) -> float:
    omega = 2 * np.pi * frequencia
    delta = np.sqrt(2 * RHO / (omega * MU))
    return delta


def funcao_bessel_psi1(zeta: float, tipo_litz: bool = True) -> float:
    psi1 = 1 + zeta**2/8 + zeta**4/192 + zeta**6/9216
    if tipo_litz:
        psi1 *= 1.02
    return psi1


def funcao_bessel_psi2(zeta: float) -> float:
    return zeta**2 / 16 + zeta**4 / 768


# Pesquisar isso
def fator_geometrico_camada_unica(Ta: float, ds_bundle: float) -> float:
    razao = ds_bundle / Ta
    if razao < 0.5:
        return 0.1
    elif razao < 1.0:
        return 0.3
    elif razao < 2.0:
        return 0.5
    else:
        return 0.7


def resistencia_dc_espira(comprimento: float, num_fios: int, diametro_fio: float, fator_empacotamento: float) -> float:
    area_fio = np.pi * (diametro_fio / 2)**2
    area_total = num_fios * area_fio
    area_efetiva = fator_empacotamento * area_total
    return RHO * comprimento / area_efetiva


def calcular_resistencia_completa(a: float, b: float, N: int, n: int, ds: float,
                                 eta: float, f: float, Ta: float,
                                 geometria: str = 'retangular') -> Dict[str, float]:
    delta = profundidade_penetracao(f)
    zeta = ds / delta
    if geometria == 'retangular':
        L_espira = 2 * (a + b)
    elif geometria == 'circular':
        L_espira = np.pi * a
    else:
        raise ValueError("Geometria deve ser 'retangular' ou 'circular'")
    
    Rdc_espira = resistencia_dc_espira(L_espira, n, ds, eta)
    Psi1 = funcao_bessel_psi1(zeta, tipo_litz=True)
    Psi2 = funcao_bessel_psi2(zeta)
    area_fio = np.pi * (ds/2)**2
    area_bundle = n * area_fio / eta
    ds_bundle = 2 * np.sqrt(area_bundle / np.pi)
    fator_geo = fator_geometrico_camada_unica(Ta, ds_bundle)
    Rskin_espira = Psi1 * Rdc_espira
    Rprox_espira = Psi2 * Rdc_espira * fator_geo
    Rtotal_espira = Rskin_espira + Rprox_espira
    Rtotal_bobina = N * Rtotal_espira
    area_efetiva = eta * n * np.pi * (ds/2)**2

    return {
        'frequencia_hz': f,
        'profundidade_penetracao_mm': delta * 1e3,
        'zeta': zeta,
        'comprimento_espira_m': L_espira,
        'area_efetiva_mm2': area_efetiva * 1e6,
        'diametro_bundle_mm': ds_bundle * 1e3,
        'Rdc_espira_ohm': Rdc_espira,
        'Psi1': Psi1,
        'Psi2': Psi2,
        'fator_geometrico': fator_geo,
        'Rskin_espira_ohm': Rskin_espira,
        'Rprox_espira_ohm': Rprox_espira,
        'Rtotal_espira_ohm': Rtotal_espira,
        'Rtotal_bobina_ohm': Rtotal_bobina,
        'razao_skin_total': Rskin_espira / Rtotal_espira,
        'razao_prox_total': Rprox_espira / Rtotal_espira
    }


def analise_sensibilidade_frequencia(params: Dict, freq_min=70e3, freq_max=95e3, num_pontos=50):
    frequencias = np.linspace(freq_min, freq_max, num_pontos)
    resistencias = []
    for f in frequencias:
        p = params.copy()
        p['f'] = f
        r = calcular_resistencia_completa(**p)
        resistencias.append(r['Rtotal_bobina_ohm'])
    return frequencias, np.array(resistencias)


def analise_sensibilidade_empacotamento(params: Dict, eta_min=0.3, eta_max=0.7, num_pontos=20):
    etas = np.linspace(eta_min, eta_max, num_pontos)
    resistencias = []
    for eta in etas:
        p = params.copy()
        p['eta'] = eta
        r = calcular_resistencia_completa(**p)
        resistencias.append(r['Rtotal_bobina_ohm'])
    return etas, np.array(resistencias)


def imprimir_resultados(r: Dict):
    print("=" * 70)
    print("RESULTADOS DO CÁLCULO DE RESISTÊNCIA AC - BOBINA LITZ")
    print("=" * 70)
    print()
    print(f"Frequência: {r['frequencia_hz']/1e3:.2f} kHz")
    print(f"Profundidade (δ): {r['profundidade_penetracao_mm']:.3f} mm")
    print(f"ζ (ds/δ): {r['zeta']:.3f}")
    print(f"Comprimento espira: {r['comprimento_espira_m']:.3f} m")
    print()
    print(f"Área efetiva: {r['area_efetiva_mm2']:.3f} mm²")
    print(f"Diâmetro bundle: {r['diametro_bundle_mm']:.2f} mm")
    print()
    print(f"Ψ₁: {r['Psi1']:.4f} | Ψ₂: {r['Psi2']:.4f} | Fator geom.: {r['fator_geometrico']:.2f}")
    print()
    print(f"Rdc: {r['Rdc_espira_ohm']*1e3:.2f} mΩ | Rskin: {r['Rskin_espira_ohm']*1e3:.2f} mΩ | Rprox: {r['Rprox_espira_ohm']*1e3:.2f} mΩ")
    print(f"Rtotal espira: {r['Rtotal_espira_ohm']*1e3:.2f} mΩ | Rtotal bobina: {r['Rtotal_bobina_ohm']:.3f} Ω")
    print(f"Efeito pelicular: {r['razao_skin_total']*100:.1f}% | Efeito proximidade: {r['razao_prox_total']*100:.1f}%")
    print("=" * 70)


def exemplo_bobina_primaria():
    print("\nEXEMPLO: BOBINA PRIMÁRIA - SISTEMA WPT SAE J2954\n")
    parametros = {
        'a': 0.65, 'b': 0.5, 'N': 15, 'n': 280,
        'ds': 0.1e-3, 'eta': 0.5, 'f': 85e3,
        'Ta': 1.5e-3, 'geometria': 'retangular'
    }
    resultados = calcular_resistencia_completa(**parametros)
    imprimir_resultados(resultados)
    return parametros, resultados


def exemplo_bobina_secundaria():
    print("\nEXEMPLO: BOBINA SECUNDÁRIA - SISTEMA WPT SAE J2954\n")
    parametros = {
        'a': 0.45, 'b': 0.35, 'N': 20, 'n': 280,
        'ds': 0.1e-3, 'eta': 0.5, 'f': 85e3,
        'Ta': 1.0e-3, 'geometria': 'retangular'
    }
    resultados = calcular_resistencia_completa(**parametros)
    imprimir_resultados(resultados)
    return parametros, resultados


def plotar_analise_sensibilidade(parametros):
    print("\nGerando gráficos de análise de sensibilidade...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análise de Sensibilidade - Resistência da Bobina Litz', fontsize=16, fontweight='bold')

    freqs, Rs_freq = analise_sensibilidade_frequencia(parametros, 70e3, 95e3, 50)
    axes[0, 0].plot(freqs/1e3, Rs_freq, 'b-', linewidth=2)
    axes[0, 0].axvline(85, color='r', linestyle='--')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Frequência (kHz)')
    axes[0, 0].set_ylabel('Resistência (Ω)')
    axes[0, 0].set_title('Variação com Frequência')

    etas, Rs_eta = analise_sensibilidade_empacotamento(parametros, 0.3, 0.7, 20)
    axes[0, 1].plot(etas, Rs_eta, 'g-', linewidth=2)
    axes[0, 1].axvline(0.5, color='r', linestyle='--')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Fator η')
    axes[0, 1].set_ylabel('Resistência (Ω)')
    axes[0, 1].set_title('Variação com Empacotamento')

    Ns = np.arange(10, 31, 1)
    Rs_N = [calcular_resistencia_completa(**{**parametros, 'N': N})['Rtotal_bobina_ohm'] for N in Ns]
    axes[1, 0].plot(Ns, Rs_N, 'm-', linewidth=2)
    axes[1, 0].axvline(15, color='r', linestyle='--')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Espiras (N)')
    axes[1, 0].set_ylabel('Resistência (Ω)')
    axes[1, 0].set_title('Variação com N')

    Tas = np.linspace(0.5e-3, 3.0e-3, 20)
    Rs_Ta = [calcular_resistencia_completa(**{**parametros, 'Ta': Ta})['Rtotal_bobina_ohm'] for Ta in Tas]
    axes[1, 1].plot(Tas*1e3, Rs_Ta, 'c-', linewidth=2)
    axes[1, 1].axvline(1.5, color='r', linestyle='--')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('Espaçamento (mm)')
    axes[1, 1].set_ylabel('Resistência (Ω)')
    axes[1, 1].set_title('Variação com Espaçamento')

    plt.tight_layout()
    plt.savefig('/home/sandbox/analise_sensibilidade_resistencia.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos salvos em: /home/sandbox/analise_sensibilidade_resistencia.png")
    return fig


def plotar_decomposicao_resistencia(resultados):
    print("\nGerando gráfico de decomposição da resistência...")
    labels = ['Efeito Pelicular', 'Efeito de Proximidade']
    sizes = [resultados['razao_skin_total']*100, resultados['razao_prox_total']*100]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12})
    ax.set_title(f"Decomposição da Resistência\nRtotal = {resultados['Rtotal_espira_ohm']*1e3:.2f} mΩ")
    plt.tight_layout()
    plt.savefig('/home/sandbox/decomposicao_resistencia.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico salvo em: /home/sandbox/decomposicao_resistencia.png")
    return fig


def main():
    print("\n" + "="*70)
    print("CALCULADORA DE RESISTÊNCIA AC PARA BOBINAS LITZ")
    print("Sistema WPT com Enrolamento em Camada Única")
    print("="*70)
    params1, results1 = exemplo_bobina_primaria()
    params2, results2 = exemplo_bobina_secundaria()
    # plotar_analise_sensibilidade(params1)
    # plotar_decomposicao_resistencia(results1)
    print("\nCálculos concluídos com sucesso!")


if __name__ == "__main__":
    main()
