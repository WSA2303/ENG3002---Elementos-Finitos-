import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# TESTE DE VALIDAÇÃO DO PROGRAMA DE VIGA/PÓRTICO 2D
# Viga em balanço com carga concentrada perpendicular na ponta
# Elemento de Euler-Bernoulli 2D com 2 nós e 3 GDL por nó:
# {u, v, theta}
#
# Unidades:
# coordenadas: m
# E: Pa = N/m²
# A: m²
# I: m^4
# força: N
# deslocamento: m
# rotação: rad
# ============================================================

# -------------------------
# Configurações gerais
# -------------------------
MOSTRAR_GRAFICOS = False
NPTS_ELEMENTO = 100
PASTA_SAIDA = Path("outputs_teste_balanco_viga2D")
PASTA_SAIDA.mkdir(exist_ok=True)

# Graus de liberdade
DOF_U = 0
DOF_V = 1
DOF_THETA = 2
DFREEDOM = 3

# -------------------------
# Dados do teste
# -------------------------
E = 205e9          # Pa
A = 0.00146373     # m²
I = 8.0e-6         # m^4
L = 1.0            # m
P = 10000.0        # N, módulo da carga transversal na ponta

# Convenção adotada no teste:
# A força aplicada é sempre perpendicular à viga e negativa no eixo local v.
# Logo, em coordenadas locais, a força nodal no nó livre é:
# [N2, V2, M2] = [0, -P, 0]

# ============================================================
# Funções do programa de viga 2D
# ============================================================

def matriz_rigidez_local(E, A, I, L):
    """Matriz de rigidez local do elemento de viga/pórtico 2D Euler-Bernoulli.
    Ordem dos GDL locais: [u1, v1, theta1, u2, v2, theta2].
    """
    EA_L = E * A / L
    EI_L3 = E * I / L**3

    k = np.array([
        [ EA_L,          0.0,          0.0, -EA_L,          0.0,          0.0],
        [ 0.0,     12*EI_L3,  6*L*EI_L3,  0.0,    -12*EI_L3,  6*L*EI_L3],
        [ 0.0,   6*L*EI_L3, 4*L**2*EI_L3, 0.0, -6*L*EI_L3, 2*L**2*EI_L3],
        [-EA_L,          0.0,          0.0,  EA_L,          0.0,          0.0],
        [ 0.0,    -12*EI_L3, -6*L*EI_L3,  0.0,     12*EI_L3, -6*L*EI_L3],
        [ 0.0,   6*L*EI_L3, 2*L**2*EI_L3, 0.0, -6*L*EI_L3, 4*L**2*EI_L3]
    ], dtype=float)

    return k


def matriz_transformacao(c, s):
    """Transforma deslocamentos globais em locais: u_local = T u_global."""
    T = np.array([
        [ c,  s, 0.0, 0.0, 0.0, 0.0],
        [-s,  c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c,  s, 0.0],
        [0.0, 0.0, 0.0, -s,  c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=float)
    return T


def funcoes_hermite(x, L):
    xi = x / L
    N1 = 1.0 - 3.0*xi**2 + 2.0*xi**3
    N2 = L * (xi - 2.0*xi**2 + xi**3)
    N3 = 3.0*xi**2 - 2.0*xi**3
    N4 = L * (-xi**2 + xi**3)
    return N1, N2, N3, N4


def deslocamento_transversal_hermite(x, L, v1, th1, v2, th2):
    N1, N2, N3, N4 = funcoes_hermite(x, L)
    return N1*v1 + N2*th1 + N3*v2 + N4*th2


def esforcos_internos_elemento(E, A, I, L, u_local, npts=NPTS_ELEMENTO):
    """Calcula N(x), M(x) e V(x).
    Convenções:
    N = EA du/dx
    M = EI d²v/dx²
    V = -dM/dx = -EI d³v/dx³
    """
    u1, v1, th1, u2, v2, th2 = u_local

    x = np.linspace(0.0, L, npts)
    xi = x / L

    N = E * A * (u2 - u1) / L
    N_array = N * np.ones_like(x)

    d2v_dx2 = (
        (-6.0 + 12.0*xi) * v1
        + L * (-4.0 + 6.0*xi) * th1
        + (6.0 - 12.0*xi) * v2
        + L * (-2.0 + 6.0*xi) * th2
    ) / L**2

    d3v_dx3 = (
        12.0*v1
        + 6.0*L*th1
        - 12.0*v2
        + 6.0*L*th2
    ) / L**3

    M = E * I * d2v_dx2
    V = -E * I * d3v_dx3 * np.ones_like(x)

    return x, N_array, M, V


def resolver_balanco(nome, coordenadas, E, A, I, P):
    """Resolve uma viga em balanço com 1 elemento.

    Nó 1: engastado.
    Nó 2: livre.
    Carga: perpendicular à viga, aplicada no nó 2, no sentido -v local.
    """
    nnods = 2
    ndof_total = nnods * DFREEDOM

    dx = coordenadas[1, 0] - coordenadas[0, 0]
    dy = coordenadas[1, 1] - coordenadas[0, 1]
    L_elem = np.hypot(dx, dy)
    c = dx / L_elem
    s = dy / L_elem

    k_local = matriz_rigidez_local(E, A, I, L_elem)
    T = matriz_transformacao(c, s)
    k_global = T.T @ k_local @ T

    # Carga local transversal na ponta: [0, 0, 0, 0, -P, 0]
    F_local = np.array([0.0, 0.0, 0.0, 0.0, -P, 0.0])
    F_global = T.T @ F_local

    K = k_global.copy()
    F = F_global.copy()
    K_original = K.copy()
    F_original = F.copy()

    # Engaste no nó 1: ux1 = uy1 = theta1 = 0
    restricoes = [0, 1, 2]
    for gdlf in restricoes:
        K[gdlf, :] = 0.0
        K[:, gdlf] = 0.0
        K[gdlf, gdlf] = 1.0
        F[gdlf] = 0.0

    desloc = np.linalg.solve(K, F)
    reacoes = K_original @ desloc - F_original

    u_local = T @ desloc
    f_local_nodal = k_local @ u_local
    xloc, N, M, V = esforcos_internos_elemento(E, A, I, L_elem, u_local)

    # Solução analítica local para viga em balanço com carga transversal -P
    v2_exato = -P * L_elem**3 / (3.0 * E * I)
    theta2_exato = -P * L_elem**2 / (2.0 * E * I)
    V_exato = P
    M_engaste_exato = P * L_elem
    M_ponta_exato = 0.0

    erro_v = abs((u_local[4] - v2_exato) / v2_exato) * 100.0
    erro_th = abs((u_local[5] - theta2_exato) / theta2_exato) * 100.0

    return {
        "nome": nome,
        "coordenadas": coordenadas,
        "L": L_elem,
        "c": c,
        "s": s,
        "forca_global": F_global,
        "desloc_global": desloc,
        "desloc_local": u_local,
        "reacoes_global": reacoes,
        "forcas_locais_nodais": f_local_nodal,
        "x": xloc,
        "N": N,
        "M": M,
        "V": V,
        "v2_exato": v2_exato,
        "theta2_exato": theta2_exato,
        "V_exato": V_exato,
        "M_engaste_exato": M_engaste_exato,
        "M_ponta_exato": M_ponta_exato,
        "erro_v_percent": erro_v,
        "erro_theta_percent": erro_th,
    }


def plotar_deformada(resultado, fator_escala=80.0):
    nome = resultado["nome"]
    coordenadas = resultado["coordenadas"]
    L_elem = resultado["L"]
    c = resultado["c"]
    s = resultado["s"]
    u_local = resultado["desloc_local"]
    u1, v1, th1, u2, v2, th2 = u_local

    xloc = np.linspace(0.0, L_elem, NPTS_ELEMENTO)
    xi = xloc / L_elem
    u_axial = (1.0 - xi)*u1 + xi*u2
    v_trans = deslocamento_transversal_hermite(xloc, L_elem, v1, th1, v2, th2)

    x_def_local = xloc + fator_escala * u_axial
    y_def_local = fator_escala * v_trans

    x1_global, y1_global = coordenadas[0]
    x_def_global = x1_global + c*x_def_local - s*y_def_local
    y_def_global = y1_global + s*x_def_local + c*y_def_local

    plt.figure(figsize=(6, 5))
    plt.plot(coordenadas[:, 0], coordenadas[:, 1], "k--o", label="Indeformada")
    plt.plot(x_def_global, y_def_global, linewidth=2.0, label=f"Deformada x{fator_escala:g}")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{nome} - deformada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / f"deformada_{nome}.png", dpi=200)
    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close()


def plotar_diagramas(resultado):
    nome = resultado["nome"]
    x = resultado["x"]

    plt.figure(figsize=(6, 4))
    plt.plot(x, resultado["V"] / 1e3)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("x local [m]")
    plt.ylabel("V [kN]")
    plt.title(f"{nome} - esforço cortante")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / f"V_{nome}.png", dpi=200)
    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(x, resultado["M"] / 1e3)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("x local [m]")
    plt.ylabel("M [kN.m]")
    plt.title(f"{nome} - momento fletor")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / f"M_{nome}.png", dpi=200)
    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close()


# ============================================================
# Definição dos três casos
# ============================================================
casos = [
    ("horizontal", np.array([[0.0, 0.0], [L, 0.0]], dtype=float)),
    ("vertical",   np.array([[0.0, 0.0], [0.0, L]], dtype=float)),
    ("inclinado_45", np.array([[0.0, 0.0], [L/np.sqrt(2.0), L/np.sqrt(2.0)]], dtype=float)),
]

resultados = [resolver_balanco(nome, coords, E, A, I, P) for nome, coords in casos]

# ============================================================
# Relatório
# ============================================================
relatorio = PASTA_SAIDA / "relatorio_teste_balanco_viga2D.txt"
with open(relatorio, "w", encoding="utf-8") as arq:
    arq.write("============================================================\n")
    arq.write("TESTE DE VALIDAÇÃO - VIGA EM BALANÇO COM 1 ELEMENTO\n")
    arq.write("Elemento de Euler-Bernoulli 2D | Sistema SI\n")
    arq.write("============================================================\n\n")

    arq.write("1. Dados adotados\n")
    arq.write(f"E = {E:.6e} Pa\n")
    arq.write(f"A = {A:.6e} m²\n")
    arq.write(f"I = {I:.6e} m^4\n")
    arq.write(f"L = {L:.6e} m\n")
    arq.write(f"P = {P:.6e} N\n\n")

    arq.write("2. Solução analítica local\n")
    arq.write("Para carga transversal aplicada na ponta de uma viga em balanço:\n")
    arq.write("v_L = -P L^3/(3 E I)\n")
    arq.write("theta_L = -P L^2/(2 E I)\n")
    arq.write("V = P\n")
    arq.write("M_engaste = P L\n\n")

    for res in resultados:
        arq.write("------------------------------------------------------------\n")
        arq.write(f"CASO: {res['nome']}\n")
        arq.write("------------------------------------------------------------\n")
        arq.write(f"Comprimento = {res['L']:.6e} m\n")
        arq.write(f"c = {res['c']:.6e}, s = {res['s']:.6e}\n")
        arq.write("Força global aplicada [Fx1, Fy1, M1, Fx2, Fy2, M2]:\n")
        arq.write("  " + "  ".join(f"{v:.6e}" for v in res["forca_global"]) + "\n")
        arq.write("Deslocamentos globais [ux1, uy1, th1, ux2, uy2, th2]:\n")
        arq.write("  " + "  ".join(f"{v:.6e}" for v in res["desloc_global"]) + "\n")
        arq.write("Deslocamentos locais [u1, v1, th1, u2, v2, th2]:\n")
        arq.write("  " + "  ".join(f"{v:.6e}" for v in res["desloc_local"]) + "\n\n")

        arq.write("Comparação no nó livre em coordenadas locais:\n")
        arq.write(f"v2 programa  = {res['desloc_local'][4]: .10e} m\n")
        arq.write(f"v2 analítico = {res['v2_exato']: .10e} m\n")
        arq.write(f"erro v2      = {res['erro_v_percent']: .6e} %\n")
        arq.write(f"theta2 programa  = {res['desloc_local'][5]: .10e} rad\n")
        arq.write(f"theta2 analítico = {res['theta2_exato']: .10e} rad\n")
        arq.write(f"erro theta2      = {res['erro_theta_percent']: .6e} %\n\n")

        arq.write("Esforços internos obtidos pelo programa:\n")
        arq.write(f"N = {res['N'][0]: .10e} N\n")
        arq.write(f"V = {res['V'][0]: .10e} N\n")
        arq.write(f"M(0) = {res['M'][0]: .10e} N.m\n")
        arq.write(f"M(L) = {res['M'][-1]: .10e} N.m\n")
        arq.write("Forças nodais locais resistentes [N1, V1, M1, N2, V2, M2]:\n")
        arq.write("  " + "  ".join(f"{v:.6e}" for v in res["forcas_locais_nodais"]) + "\n\n")

# CSV resumo
csv_resumo = PASTA_SAIDA / "resumo_comparacao.csv"
with open(csv_resumo, "w", encoding="utf-8") as arq:
    arq.write("caso,c,s,Fx2_N,Fy2_N,ux2_m,uy2_m,u2_local_m,v2_local_m,theta2_rad,v2_exato_m,theta_exato_rad,erro_v_percent,erro_theta_percent,V_N,M0_Nm,ML_Nm\n")
    for res in resultados:
        arq.write(
            f"{res['nome']},{res['c']:.10e},{res['s']:.10e},"
            f"{res['forca_global'][3]:.10e},{res['forca_global'][4]:.10e},"
            f"{res['desloc_global'][3]:.10e},{res['desloc_global'][4]:.10e},"
            f"{res['desloc_local'][3]:.10e},{res['desloc_local'][4]:.10e},{res['desloc_local'][5]:.10e},"
            f"{res['v2_exato']:.10e},{res['theta2_exato']:.10e},"
            f"{res['erro_v_percent']:.10e},{res['erro_theta_percent']:.10e},"
            f"{res['V'][0]:.10e},{res['M'][0]:.10e},{res['M'][-1]:.10e}\n"
        )

# Gráficos
for res in resultados:
    plotar_deformada(res)
    plotar_diagramas(res)

# Saída no terminal
print("Cálculo finalizado.")
print(f"Relatório salvo em: {relatorio.resolve()}")
print(f"Resumo CSV salvo em: {csv_resumo.resolve()}")
print(f"Gráficos salvos em: {PASTA_SAIDA.resolve()}")
print("\nResumo dos resultados locais no nó livre:")
for res in resultados:
    print(
        f"{res['nome']:12s} | v2 = {res['desloc_local'][4]: .6e} m "
        f"| v_exato = {res['v2_exato']: .6e} m "
        f"| erro = {res['erro_v_percent']: .3e}% "
        f"| theta2 = {res['desloc_local'][5]: .6e} rad "
        f"| erro_theta = {res['erro_theta_percent']: .3e}%"
    )
