import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Programa para cálculo de vigas/pórticos planos 2D
# Elemento de Euler-Bernoulli com 2 nós e 3 GDL por nó:
# {u, v, theta}
#
# Unidades recomendadas:
# coordenadas: m
# E: Pa = N/m²
# A: m²
# I: m^4
# forças: N
# momentos: N.m
# deslocamentos: m
# rotações: rad
# ============================================================

# -------------------------
# Configurações gerais
# -------------------------
FATOR_ESCALA = 300.0          # fator de escala para visualização da deformada
MOSTRAR_GRAFICOS = False       # coloque True se quiser abrir as figuras durante a execução
NPTS_ELEMENTO = 50            # pontos para calcular diagramas internos
PASTA_SAIDA = Path("outputs_viga_EB2D")
PASTA_SAIDA.mkdir(exist_ok=True)

# Identificação dos graus de liberdade por nó
DOF_U = 0       # deslocamento global X
DOF_V = 1       # deslocamento global Y
DOF_THETA = 2   # rotação em torno de Z

# ============================================================
# DADOS DE ENTRADA
# ============================================================

# Geometria: coordenadas nodais [x, y]
coordenadas = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [0.0, 2.0],
    [0.5, 1.5],
    [1.0, 1.0],
    [0.5, 0.5]
], dtype=float)

# Conectividade dos elementos [nó inicial, nó final]
# Entrada em numeração humana, começando em 1.
connections = np.array([
    [1, 2],
    [1, 6],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=int) - 1

nnods = coordenadas.shape[0]
nelem = connections.shape[0]
dfreedom = 3
ndof_total = nnods * dfreedom

# Propriedades por elemento
# Modifique livremente estes vetores para usar materiais/seções diferentes.
E_elem = np.array([
    205e9, 205e9, 205e9, 205e9, 205e9,
    205e9, 205e9, 205e9, 205e9
], dtype=float)

A_elem = np.array([
    0.00146373, 0.00146373, 0.00146373,
    0.00146373, 0.00146373, 0.00146373,
    0.00146373, 0.00146373, 0.00146373
], dtype=float)

I_elem = np.array([
    8.0e-6, 8.0e-6, 8.0e-6,
    8.0e-6, 8.0e-6, 8.0e-6,
    8.0e-6, 8.0e-6, 8.0e-6
], dtype=float)

# Carregamentos nodais: [nó, GDL, valor]
# GDL: DOF_U = 0, DOF_V = 1, DOF_THETA = 2
loads = np.array([
    [3, DOF_U,      10000.0],
    [6, DOF_V,     -10000.0]
], dtype=float)

# Condições de contorno: [nó, GDL, valor_prescrito]
# Se o valor_prescrito for omitido, o código assume valor zero.
restrictions = np.array([
    [1, DOF_U,     0.0],
    [1, DOF_V,     0.0],
    [5, DOF_V,     0.0]
], dtype=float)

# ============================================================
# FUNÇÕES AUXILIARES
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
    """Matriz T que transforma deslocamentos globais em locais: u_local = T u_global."""
    T = np.array([
        [ c,  s, 0.0, 0.0, 0.0, 0.0],
        [-s,  c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c,  s, 0.0],
        [0.0, 0.0, 0.0, -s,  c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=float)
    return T


def gdl_elemento(no1, no2):
    """Retorna os índices dos GDL globais de um elemento."""
    return np.array([
        no1*dfreedom + DOF_U,
        no1*dfreedom + DOF_V,
        no1*dfreedom + DOF_THETA,
        no2*dfreedom + DOF_U,
        no2*dfreedom + DOF_V,
        no2*dfreedom + DOF_THETA
    ], dtype=int)


def funcoes_hermite(x, L):
    """Funções de forma de Hermite para deslocamento transversal."""
    xi = x / L
    N1 = 1.0 - 3.0*xi**2 + 2.0*xi**3
    N2 = L * (xi - 2.0*xi**2 + xi**3)
    N3 = 3.0*xi**2 - 2.0*xi**3
    N4 = L * (-xi**2 + xi**3)
    return N1, N2, N3, N4


def esforcos_internos_elemento(E, A, I, L, u_local, npts=NPTS_ELEMENTO):
    """Calcula N(x), M(x) e V(x) a partir dos deslocamentos locais do elemento.

    Convenções usadas:
    N = EA du/dx
    M = EI d²v/dx²
    V = -dM/dx = -EI d³v/dx³
    """
    u1, v1, th1, u2, v2, th2 = u_local

    x = np.linspace(0.0, L, npts)
    xi = x / L

    # Esforço normal constante
    N = E * A * (u2 - u1) / L
    N_array = N * np.ones_like(x)

    # Segunda derivada do campo transversal interpolado por Hermite
    d2v_dx2 = (
        (-6.0 + 12.0*xi) * v1
        + L * (-4.0 + 6.0*xi) * th1
        + (6.0 - 12.0*xi) * v2
        + L * (-2.0 + 6.0*xi) * th2
    ) / L**2

    # Terceira derivada do campo transversal interpolado por Hermite
    d3v_dx3 = (
        12.0*v1
        + 6.0*L*th1
        - 12.0*v2
        + 6.0*L*th2
    ) / L**3

    M = E * I * d2v_dx2
    V = -E * I * d3v_dx3 * np.ones_like(x)

    return x, N_array, M, V


def deslocamento_transversal_hermite(x, L, v1, th1, v2, th2):
    N1, N2, N3, N4 = funcoes_hermite(x, L)
    return N1*v1 + N2*th1 + N3*v2 + N4*th2


# ============================================================
# PRÉ-PROCESSAMENTO: GEOMETRIA DOS ELEMENTOS
# ============================================================

comprimentos = np.zeros(nelem, dtype=float)
angulos = np.zeros(nelem, dtype=float)
cossenos = np.zeros(nelem, dtype=float)
senos = np.zeros(nelem, dtype=float)

for e in range(nelem):
    no1, no2 = connections[e]
    dx = coordenadas[no2, 0] - coordenadas[no1, 0]
    dy = coordenadas[no2, 1] - coordenadas[no1, 1]
    L = np.sqrt(dx**2 + dy**2)

    if L <= 0.0:
        raise ValueError(f"Elemento {e+1} possui comprimento nulo.")

    comprimentos[e] = L
    angulos[e] = np.arctan2(dy, dx)
    cossenos[e] = dx / L
    senos[e] = dy / L

# ============================================================
# MONTAGEM DA MATRIZ DE RIGIDEZ GLOBAL E VETOR DE CARGAS
# ============================================================

Kglobal = np.zeros((ndof_total, ndof_total), dtype=float)
Fglobal = np.zeros(ndof_total, dtype=float)

# Guardar matrizes locais e transformações para pós-processamento
Klocais = []
T_elementos = []

for e in range(nelem):
    no1, no2 = connections[e]
    L = comprimentos[e]
    c = cossenos[e]
    s = senos[e]

    k_local = matriz_rigidez_local(E_elem[e], A_elem[e], I_elem[e], L)
    T = matriz_transformacao(c, s)
    k_global = T.T @ k_local @ T

    Klocais.append(k_local)
    T_elementos.append(T)

    dofs = gdl_elemento(no1, no2)
    Kglobal[np.ix_(dofs, dofs)] += k_global

# Aplicação dos carregamentos nodais
for carga in loads:
    node = int(carga[0]) - 1
    dof = int(carga[1])
    value = carga[2]
    Fglobal[node*dfreedom + dof] += value

# Cópias antes das condições de contorno, para cálculo das reações
K_original = Kglobal.copy()
F_original = Fglobal.copy()

# ============================================================
# APLICAÇÃO DAS CONDIÇÕES DE CONTORNO
# ============================================================

for restr in restrictions:
    node = int(restr[0]) - 1
    dof = int(restr[1])
    valor = restr[2] if len(restr) >= 3 else 0.0

    gdlf = node*dfreedom + dof

    # Ajuste para deslocamento prescrito não nulo
    Fglobal -= Kglobal[:, gdlf] * valor

    # Método direto
    Kglobal[gdlf, :] = 0.0
    Kglobal[:, gdlf] = 0.0
    Kglobal[gdlf, gdlf] = 1.0
    Fglobal[gdlf] = valor

# ============================================================
# SOLUÇÃO DO SISTEMA
# ============================================================

try:
    desloc = np.linalg.solve(Kglobal, Fglobal)
except np.linalg.LinAlgError as erro:
    raise RuntimeError(
        "A matriz global ficou singular. Verifique se há vínculos suficientes "
        "para impedir movimentos de corpo rígido."
    ) from erro

reacoes = K_original @ desloc - F_original

# ============================================================
# PÓS-PROCESSAMENTO: ESFORÇOS INTERNOS
# ============================================================

resultados_elementos = []

for e in range(nelem):
    no1, no2 = connections[e]
    dofs = gdl_elemento(no1, no2)
    u_global_elem = desloc[dofs]
    u_local_elem = T_elementos[e] @ u_global_elem

    xloc, N, M, V = esforcos_internos_elemento(
        E_elem[e], A_elem[e], I_elem[e], comprimentos[e], u_local_elem
    )

    f_local_nodal = Klocais[e] @ u_local_elem

    resultados_elementos.append({
        "u_local": u_local_elem,
        "f_local_nodal": f_local_nodal,
        "x": xloc,
        "N": N,
        "M": M,
        "V": V
    })

# ============================================================
# RELATÓRIO DE RESULTADOS
# ============================================================

relatorio_path = PASTA_SAIDA / "relatorio_resultados_viga_EB2D.txt"

nomes_gdl = {
    DOF_U: "u_x",
    DOF_V: "u_y",
    DOF_THETA: "theta_z"
}

with open(relatorio_path, "w", encoding="utf-8") as arq:
    arq.write("RELATÓRIO DE RESULTADOS - VIGA/PÓRTICO 2D EULER-BERNOULLI\n")
    arq.write("="*72 + "\n\n")

    arq.write("1. DADOS GERAIS\n")
    arq.write(f"Número de nós: {nnods}\n")
    arq.write(f"Número de elementos: {nelem}\n")
    arq.write(f"GDL por nó: {dfreedom} -> [u_x, u_y, theta_z]\n")
    arq.write(f"Número total de GDL: {ndof_total}\n\n")

    arq.write("2. DESLOCAMENTOS NODAIS\n")
    arq.write("Nó        ux [m]          uy [m]        theta [rad]       ux [mm]        uy [mm]\n")
    for n in range(nnods):
        ux = desloc[n*dfreedom + DOF_U]
        uy = desloc[n*dfreedom + DOF_V]
        th = desloc[n*dfreedom + DOF_THETA]
        arq.write(f"{n+1:3d}  {ux:14.6e} {uy:14.6e} {th:14.6e} {ux*1e3:12.6f} {uy*1e3:12.6f}\n")
    arq.write("\n")

    arq.write("3. REAÇÕES DE APOIO\n")
    arq.write("Nó      GDL         Reação\n")
    for restr in restrictions:
        node = int(restr[0]) - 1
        dof = int(restr[1])
        gdlf = node*dfreedom + dof
        unidade = "N" if dof in [DOF_U, DOF_V] else "N.m"
        arq.write(f"{node+1:3d}   {nomes_gdl[dof]:8s}  {reacoes[gdlf]:14.6e} {unidade}\n")
    arq.write("\n")

    arq.write("4. PROPRIEDADES DOS ELEMENTOS\n")
    arq.write("Elem  nó_i nó_j      L [m]      ang [graus]       E [Pa]       A [m²]       I [m^4]\n")
    for e in range(nelem):
        no1, no2 = connections[e]
        arq.write(
            f"{e+1:4d} {no1+1:5d} {no2+1:5d} "
            f"{comprimentos[e]:10.5f} {np.rad2deg(angulos[e]):14.5f} "
            f"{E_elem[e]:12.5e} {A_elem[e]:12.5e} {I_elem[e]:12.5e}\n"
        )
    arq.write("\n")

    arq.write("5. ESFORÇOS INTERNOS POR ELEMENTO\n")
    arq.write("Convenções: N = EA du/dx; M = EI d²v/dx²; V = -dM/dx\n")
    arq.write("N positivo indica tração no eixo local do elemento.\n\n")

    for e, res in enumerate(resultados_elementos):
        no1, no2 = connections[e]
        N = res["N"]
        M = res["M"]
        V = res["V"]
        u_local = res["u_local"]
        f_local = res["f_local_nodal"]

        arq.write(f"Elemento {e+1} | nós {no1+1}-{no2+1}\n")
        arq.write("  Deslocamentos locais [u1, v1, theta1, u2, v2, theta2]:\n")
        arq.write("  " + "  ".join(f"{valor: .6e}" for valor in u_local) + "\n")
        arq.write("  Forças nodais locais resistentes [N1, V1, M1, N2, V2, M2]:\n")
        arq.write("  " + "  ".join(f"{valor: .6e}" for valor in f_local) + "\n")
        arq.write(f"  N constante = {N[0]: .6e} N = {N[0]/1e3: .6f} kN\n")
        arq.write(f"  V constante = {V[0]: .6e} N = {V[0]/1e3: .6f} kN\n")
        arq.write(f"  M(0) = {M[0]: .6e} N.m = {M[0]/1e3: .6f} kN.m\n")
        arq.write(f"  M(L) = {M[-1]: .6e} N.m = {M[-1]/1e3: .6f} kN.m\n")
        arq.write(f"  M mínimo = {np.min(M): .6e} N.m = {np.min(M)/1e3: .6f} kN.m\n")
        arq.write(f"  M máximo = {np.max(M): .6e} N.m = {np.max(M)/1e3: .6f} kN.m\n\n")

# Arquivos CSV dos deslocamentos nodais
csv_desloc = PASTA_SAIDA / "deslocamentos_nodais.csv"
with open(csv_desloc, "w", encoding="utf-8") as arq:
    arq.write("no,ux_m,uy_m,theta_rad,ux_mm,uy_mm\n")
    for n in range(nnods):
        ux = desloc[n*dfreedom + DOF_U]
        uy = desloc[n*dfreedom + DOF_V]
        th = desloc[n*dfreedom + DOF_THETA]
        arq.write(f"{n+1},{ux:.10e},{uy:.10e},{th:.10e},{ux*1e3:.10e},{uy*1e3:.10e}\n")

# Arquivos CSV dos esforços internos por elemento
for e, res in enumerate(resultados_elementos):
    csv_elem = PASTA_SAIDA / f"esforcos_elemento_{e+1:02d}.csv"
    with open(csv_elem, "w", encoding="utf-8") as arq:
        arq.write("x_local_m,N_N,M_Nm,V_N,N_kN,M_kNm,V_kN\n")
        for xval, nval, mval, vval in zip(res["x"], res["N"], res["M"], res["V"]):
            arq.write(
                f"{xval:.10e},{nval:.10e},{mval:.10e},{vval:.10e},"
                f"{nval/1e3:.10e},{mval/1e3:.10e},{vval/1e3:.10e}\n"
            )

# ============================================================
# GRÁFICOS
# ============================================================

# Estrutura original e deformada
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

# Estrutura original
for e in range(nelem):
    no1, no2 = connections[e]
    x_plot = [coordenadas[no1, 0], coordenadas[no2, 0]]
    y_plot = [coordenadas[no1, 1], coordenadas[no2, 1]]
    plt.plot(x_plot, y_plot, "k--", linewidth=1.5)
    plt.plot(x_plot, y_plot, "ko", markersize=4)

# Estrutura deformada usando a interpolação de viga em cada elemento
for e in range(nelem):
    no1, no2 = connections[e]
    L = comprimentos[e]
    c = cossenos[e]
    s = senos[e]
    x1_global, y1_global = coordenadas[no1]

    u_local = resultados_elementos[e]["u_local"]
    u1, v1, th1, u2, v2, th2 = u_local

    xloc = np.linspace(0.0, L, NPTS_ELEMENTO)
    xi = xloc / L

    # Interpolação axial linear e transversal cúbica
    u_axial = (1.0 - xi)*u1 + xi*u2
    v_trans = deslocamento_transversal_hermite(xloc, L, v1, th1, v2, th2)

    x_def_local = xloc + FATOR_ESCALA * u_axial
    y_def_local = FATOR_ESCALA * v_trans

    # Retorno para coordenadas globais
    x_def_global = x1_global + c*x_def_local - s*y_def_local
    y_def_global = y1_global + s*x_def_local + c*y_def_local

    plt.plot(x_def_global, y_def_global, linewidth=2.0)

for n in range(nnods):
    plt.text(coordenadas[n, 0], coordenadas[n, 1], f" {n+1}", fontsize=9)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Estrutura original e deformada | escala = {FATOR_ESCALA:g}x")
plt.grid(True)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "estrutura_deformada.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

# Diagramas por elemento: N, M e V em função de x local
plt.figure(figsize=(8, 5))
for e, res in enumerate(resultados_elementos):
    plt.plot(res["x"], res["N"]/1e3, label=f"Elem. {e+1}")
plt.xlabel("x local [m]")
plt.ylabel("N [kN]")
plt.title("Esforço normal por elemento")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_N.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(8, 5))
for e, res in enumerate(resultados_elementos):
    plt.plot(res["x"], res["M"]/1e3, label=f"Elem. {e+1}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [m]")
plt.ylabel("M [kN.m]")
plt.title("Momento fletor por elemento")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_M.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(8, 5))
for e, res in enumerate(resultados_elementos):
    plt.plot(res["x"], res["V"]/1e3, label=f"Elem. {e+1}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [m]")
plt.ylabel("V [kN]")
plt.title("Esforço cortante por elemento")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_V.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

# ============================================================
# SAÍDA NO TERMINAL
# ============================================================

print("Cálculo finalizado.")
print(f"Relatório salvo em: {relatorio_path.resolve()}")
print(f"CSV de deslocamentos salvo em: {csv_desloc.resolve()}")
print(f"Gráficos e CSVs dos esforços salvos em: {PASTA_SAIDA.resolve()}")
print("\nDeslocamentos nodais:")
for n in range(nnods):
    ux = desloc[n*dfreedom + DOF_U]
    uy = desloc[n*dfreedom + DOF_V]
    th = desloc[n*dfreedom + DOF_THETA]
    print(f"Nó {n+1:2d}: ux = {ux: .6e} m | uy = {uy: .6e} m | theta = {th: .6e} rad")
