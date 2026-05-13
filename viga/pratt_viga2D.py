import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# TRELIÇA DE PRATT MODELADA COM ELEMENTOS DE VIGA/PÓRTICO 2D
# Elemento de Euler-Bernoulli 2D:
#   GDL por nó = {u_x, u_y, theta_z}
#
# SISTEMA DE UNIDADES: kN - mm
#   comprimento: mm
#   força: kN
#   momento: kN.mm
#   E: kN/mm²
#   A: mm²
#   I: mm^4
# ============================================================

MOSTRAR_GRAFICOS = False
NPTS_ELEMENTO = 50
FRACAO_VISUAL = 0.15
PASTA_SAIDA = Path("Output_Pratt_Viga2D")
PASTA_SAIDA.mkdir(exist_ok=True)

DOF_U = 0
DOF_V = 1
DOF_THETA = 2

# ============================================================
# DADOS DE ENTRADA DA TRELIÇA PRATT
# ============================================================

COORD = np.array([
    [0.0,     0.0],
    [2500.0,  0.0],
    [5000.0,  0.0],
    [7500.0,  0.0],
    [10000.0, 0.0],
    [2500.0,  2500.0],
    [5000.0,  2500.0],
    [7500.0,  2500.0],
], dtype=float)

CONEC = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 6],
    [3, 7],
    [4, 8],
    [1, 6],
    [6, 3],
    [7, 4],
    [8, 5],
], dtype=int) - 1

NNOS = COORD.shape[0]
NEL = CONEC.shape[0]
NGLN = 3
NGL = NNOS * NGLN

E_ELEM = 200.0 * np.ones(NEL)     # kN/mm²
A_ELEM = 400.0 * np.ones(NEL)     # mm²

# Como o elemento de viga 2D precisa de I, adotou-se seção quadrada equivalente:
# A = 400 mm² -> 20 mm x 20 mm -> I = b*h³/12.
I_EQ = 20.0 * 20.0**3 / 12.0
I_ELEM = I_EQ * np.ones(NEL)      # mm^4

FORCAS = np.array([
    [6, DOF_V, -25.0],
    [7, DOF_V, -50.0],
    [8, DOF_V, -25.0],
], dtype=float)

VINC = np.array([
    [1, DOF_U, 0.0],
    [1, DOF_V, 0.0],
    [5, DOF_V, 0.0],
], dtype=float)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def matriz_rigidez_local(E, A, I, L):
    """Matriz local 6x6 do elemento Euler-Bernoulli 2D.
    Ordem local: [u1, v1, theta1, u2, v2, theta2].
    """
    EA_L = E * A / L
    EI = E * I
    return np.array([
        [ EA_L,          0.0,          0.0, -EA_L,          0.0,          0.0],
        [ 0.0,     12*EI/L**3,   6*EI/L**2,  0.0,    -12*EI/L**3,   6*EI/L**2],
        [ 0.0,      6*EI/L**2,     4*EI/L,  0.0,     -6*EI/L**2,     2*EI/L],
        [-EA_L,          0.0,          0.0,  EA_L,          0.0,          0.0],
        [ 0.0,    -12*EI/L**3,  -6*EI/L**2,  0.0,     12*EI/L**3,  -6*EI/L**2],
        [ 0.0,      6*EI/L**2,     2*EI/L,  0.0,     -6*EI/L**2,     4*EI/L],
    ], dtype=float)


def matriz_transformacao(c, s):
    """Transforma deslocamentos globais em locais: u_local = T @ u_global."""
    return np.array([
        [ c,  s, 0.0, 0.0, 0.0, 0.0],
        [-s,  c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c,  s, 0.0],
        [0.0, 0.0, 0.0, -s,  c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def gdl_elemento(no1, no2):
    return np.array([
        no1*NGLN + DOF_U,
        no1*NGLN + DOF_V,
        no1*NGLN + DOF_THETA,
        no2*NGLN + DOF_U,
        no2*NGLN + DOF_V,
        no2*NGLN + DOF_THETA,
    ], dtype=int)


def deslocamento_transversal_hermite(x, L, v1, th1, v2, th2):
    xi = x / L
    N1 = 1.0 - 3.0*xi**2 + 2.0*xi**3
    N2 = L * (xi - 2.0*xi**2 + xi**3)
    N3 = 3.0*xi**2 - 2.0*xi**3
    N4 = L * (-xi**2 + xi**3)
    return N1*v1 + N2*th1 + N3*v2 + N4*th2


def esforcos_internos_elemento(E, A, I, L, u_local, npts=NPTS_ELEMENTO):
    """Calcula N(x), M(x) e V(x) a partir dos deslocamentos locais."""
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


def calcula_escala_automatica(coord, desloc, fracao=FRACAO_VISUAL):
    dx = np.ptp(coord[:, 0])
    dy = np.ptp(coord[:, 1])
    lref = max(dx, dy)
    desloc_xy = np.column_stack((desloc[0::3], desloc[1::3]))
    umax = np.max(np.linalg.norm(desloc_xy, axis=1))
    if umax <= 1.0e-14 or lref <= 1.0e-14:
        return 1.0
    return fracao * lref / umax

# ============================================================
# GEOMETRIA DOS ELEMENTOS
# ============================================================

comprimentos = np.zeros(NEL)
angulos = np.zeros(NEL)
cossenos = np.zeros(NEL)
senos = np.zeros(NEL)

for e in range(NEL):
    no1, no2 = CONEC[e]
    dx = COORD[no2, 0] - COORD[no1, 0]
    dy = COORD[no2, 1] - COORD[no1, 1]
    L = np.hypot(dx, dy)
    if L <= 0.0:
        raise ValueError(f"Elemento {e+1} possui comprimento nulo.")
    comprimentos[e] = L
    angulos[e] = np.arctan2(dy, dx)
    cossenos[e] = dx / L
    senos[e] = dy / L

# ============================================================
# MONTAGEM DO SISTEMA GLOBAL
# ============================================================

KG = np.zeros((NGL, NGL), dtype=float)
FG = np.zeros(NGL, dtype=float)
K_LOCAIS = []
T_ELEM = []

for e in range(NEL):
    no1, no2 = CONEC[e]
    L = comprimentos[e]
    c = cossenos[e]
    s = senos[e]
    k_local = matriz_rigidez_local(E_ELEM[e], A_ELEM[e], I_ELEM[e], L)
    T = matriz_transformacao(c, s)
    k_global = T.T @ k_local @ T
    K_LOCAIS.append(k_local)
    T_ELEM.append(T)
    dofs = gdl_elemento(no1, no2)
    KG[np.ix_(dofs, dofs)] += k_global

# Cargas nodais
for no, gl, valor in FORCAS:
    no = int(no) - 1
    gl = int(gl)
    FG[no*NGLN + gl] += valor

KG_ORIG = KG.copy()
FG_ORIG = FG.copy()

# Condições de contorno pelo método direto
for no, gl, valor in VINC:
    no = int(no) - 1
    gl = int(gl)
    valor = float(valor)
    idx = no*NGLN + gl
    FG -= KG[:, idx] * valor
    KG[idx, :] = 0.0
    KG[:, idx] = 0.0
    KG[idx, idx] = 1.0
    FG[idx] = valor

# ============================================================
# SOLUÇÃO
# ============================================================

U = np.linalg.solve(KG, FG)      # deslocamentos [mm] e rotações [rad]
R = KG_ORIG @ U - FG_ORIG       # reações [kN] e [kN.mm]
ESCALA = calcula_escala_automatica(COORD, U)

# ============================================================
# ESFORÇOS INTERNOS
# ============================================================

resultados_elementos = []
for e in range(NEL):
    no1, no2 = CONEC[e]
    dofs = gdl_elemento(no1, no2)
    u_global_elem = U[dofs]
    u_local_elem = T_ELEM[e] @ u_global_elem
    x, N, M, V = esforcos_internos_elemento(
        E_ELEM[e], A_ELEM[e], I_ELEM[e], comprimentos[e], u_local_elem
    )
    f_local_nodal = K_LOCAIS[e] @ u_local_elem
    resultados_elementos.append({
        "u_local": u_local_elem,
        "f_local_nodal": f_local_nodal,
        "x": x,
        "N": N,
        "M": M,
        "V": V,
    })

# ============================================================
# RELATÓRIO E CSVs
# ============================================================

relatorio_path = PASTA_SAIDA / "relatorio_pratt_viga2D.txt"
csv_desloc_path = PASTA_SAIDA / "deslocamentos_pratt_viga2D.csv"
nomes_gdl = {DOF_U: "u_x", DOF_V: "u_y", DOF_THETA: "theta_z"}

with open(relatorio_path, "w", encoding="utf-8") as arq:
    arq.write("=====================================================\n")
    arq.write("RELATÓRIO DE RESULTADOS - PRATT COM ELEMENTO DE VIGA 2D\n")
    arq.write("Elemento Euler-Bernoulli 2D | Sistema kN - mm\n")
    arq.write("=====================================================\n\n")
    arq.write("1. DADOS GERAIS\n")
    arq.write(f"Número de nós: {NNOS}\n")
    arq.write(f"Número de elementos: {NEL}\n")
    arq.write(f"GDL por nó: {NGLN} -> [u_x, u_y, theta_z]\n")
    arq.write(f"Número total de GDL: {NGL}\n")
    arq.write(f"E adotado: {E_ELEM[0]:.6f} kN/mm²\n")
    arq.write(f"A adotada: {A_ELEM[0]:.6f} mm²\n")
    arq.write(f"I adotado: {I_ELEM[0]:.6f} mm^4\n")
    arq.write(f"Escala automática da deformada: {ESCALA:.6f}\n\n")

    arq.write("2. DESLOCAMENTOS NODAIS\n")
    arq.write("Nó        ux [mm]        uy [mm]       theta [rad]\n")
    for n in range(NNOS):
        ux = U[n*NGLN + DOF_U]
        uy = U[n*NGLN + DOF_V]
        th = U[n*NGLN + DOF_THETA]
        arq.write(f"{n+1:3d}  {ux:14.6e} {uy:14.6e} {th:14.6e}\n")
    arq.write("\n")

    arq.write("3. REAÇÕES DE APOIO\n")
    arq.write("Nó      GDL          Reação\n")
    for no, gl, valor in VINC:
        no = int(no) - 1
        gl = int(gl)
        idx = no*NGLN + gl
        unidade = "kN" if gl in [DOF_U, DOF_V] else "kN.mm"
        arq.write(f"{no+1:3d}   {nomes_gdl[gl]:8s}  {R[idx]:14.6e} {unidade}\n")
    arq.write("\n")

    arq.write("4. PROPRIEDADES DOS ELEMENTOS\n")
    arq.write("Elem  nó_i nó_j   L [mm]    ang [graus]   E [kN/mm²]   A [mm²]   I [mm^4]\n")
    for e in range(NEL):
        no1, no2 = CONEC[e]
        arq.write(
            f"{e+1:4d} {no1+1:5d} {no2+1:5d} "
            f"{comprimentos[e]:10.4f} {np.rad2deg(angulos[e]):12.4f} "
            f"{E_ELEM[e]:12.6f} {A_ELEM[e]:10.4f} {I_ELEM[e]:12.4f}\n"
        )
    arq.write("\n")

    arq.write("5. ESFORÇOS INTERNOS POR ELEMENTO\n")
    arq.write("Convenções: N = EA du/dx; M = EI d²v/dx²; V = -dM/dx.\n")
    arq.write("N positivo indica tração no eixo local do elemento.\n\n")
    for e, res in enumerate(resultados_elementos):
        no1, no2 = CONEC[e]
        N = res["N"]
        M = res["M"]
        V = res["V"]
        f_local = res["f_local_nodal"]
        arq.write(f"Elemento {e+1} | nós {no1+1}-{no2+1}\n")
        arq.write("  Forças nodais locais resistentes [N1, V1, M1, N2, V2, M2]:\n")
        arq.write("  " + "  ".join(f"{valor: .6e}" for valor in f_local) + "\n")
        arq.write(f"  N constante = {N[0]: .6e} kN\n")
        arq.write(f"  V constante = {V[0]: .6e} kN\n")
        arq.write(f"  M(0) = {M[0]: .6e} kN.mm = {M[0]/1000.0: .6f} kN.m\n")
        arq.write(f"  M(L) = {M[-1]: .6e} kN.mm = {M[-1]/1000.0: .6f} kN.m\n")
        arq.write(f"  M mínimo = {np.min(M): .6e} kN.mm = {np.min(M)/1000.0: .6f} kN.m\n")
        arq.write(f"  M máximo = {np.max(M): .6e} kN.mm = {np.max(M)/1000.0: .6f} kN.m\n\n")

with open(csv_desloc_path, "w", encoding="utf-8") as arq:
    arq.write("no,ux_mm,uy_mm,theta_rad\n")
    for n in range(NNOS):
        arq.write(f"{n+1},{U[n*NGLN+DOF_U]:.10e},{U[n*NGLN+DOF_V]:.10e},{U[n*NGLN+DOF_THETA]:.10e}\n")

for e, res in enumerate(resultados_elementos):
    csv_elem = PASTA_SAIDA / f"esforcos_elemento_{e+1:02d}.csv"
    with open(csv_elem, "w", encoding="utf-8") as arq:
        arq.write("x_local_mm,N_kN,M_kNmm,V_kN,M_kNm\n")
        for xval, nval, mval, vval in zip(res["x"], res["N"], res["M"], res["V"]):
            arq.write(f"{xval:.10e},{nval:.10e},{mval:.10e},{vval:.10e},{mval/1000.0:.10e}\n")

# ============================================================
# GRÁFICOS
# ============================================================

plt.figure(figsize=(11, 6))
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

for e in range(NEL):
    no1, no2 = CONEC[e]
    x = [COORD[no1, 0], COORD[no2, 0]]
    y = [COORD[no1, 1], COORD[no2, 1]]
    plt.plot(x, y, "k--", linewidth=1.4)
    plt.plot(x, y, "ko", markersize=4)

for e in range(NEL):
    no1, no2 = CONEC[e]
    L = comprimentos[e]
    c = cossenos[e]
    s = senos[e]
    x1g, y1g = COORD[no1]
    u_local = resultados_elementos[e]["u_local"]
    u1, v1, th1, u2, v2, th2 = u_local
    xloc = np.linspace(0.0, L, NPTS_ELEMENTO)
    xi = xloc / L
    u_axial = (1.0 - xi)*u1 + xi*u2
    v_trans = deslocamento_transversal_hermite(xloc, L, v1, th1, v2, th2)
    x_def_local = xloc + ESCALA * u_axial
    y_def_local = ESCALA * v_trans
    x_def_global = x1g + c*x_def_local - s*y_def_local
    y_def_global = y1g + s*x_def_local + c*y_def_local
    plt.plot(x_def_global, y_def_global, linewidth=2.0)

for n in range(NNOS):
    plt.text(COORD[n, 0], COORD[n, 1], f" {n+1}", fontsize=9)

plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(f"Treliça Pratt modelada como viga/pórtico 2D | escala automática = {ESCALA:.2f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "pratt_deformada_viga2D.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(9, 5))
for e, res in enumerate(resultados_elementos):
    plt.plot(res["x"], res["N"], label=f"Elem. {e+1}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [mm]")
plt.ylabel("N [kN]")
plt.title("Esforço normal por elemento - Pratt")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_N_pratt_viga2D.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(9, 5))
for e, res in enumerate(resultados_elementos):
    plt.plot(res["x"], res["V"], label=f"Elem. {e+1}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [mm]")
plt.ylabel("V [kN]")
plt.title("Esforço cortante por elemento - Pratt")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_V_pratt_viga2D.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(9, 5))
for e, res in enumerate(resultados_elementos):
    plt.plot(res["x"], res["M"]/1000.0, label=f"Elem. {e+1}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [mm]")
plt.ylabel("M [kN.m]")
plt.title("Momento fletor por elemento - Pratt")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_M_pratt_viga2D.png", dpi=200)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

print("Cálculo finalizado - Pratt com elemento de viga/pórtico 2D.")
print(f"Relatório salvo em: {relatorio_path.resolve()}")
print(f"CSV de deslocamentos salvo em: {csv_desloc_path.resolve()}")
print(f"Arquivos de saída salvos em: {PASTA_SAIDA.resolve()}")
print(f"Escala automática da deformada: {ESCALA:.6f}")

print("\nDESLOCAMENTOS NODAIS")
print("Nó        ux [mm]        uy [mm]       theta [rad]")
for n in range(NNOS):
    print(f"{n+1:2d}  {U[n*NGLN+DOF_U]:14.6e} {U[n*NGLN+DOF_V]:14.6e} {U[n*NGLN+DOF_THETA]:14.6e}")

print("\nREAÇÕES NOS VÍNCULOS")
for no, gl, valor in VINC:
    no = int(no) - 1
    gl = int(gl)
    idx = no*NGLN + gl
    unidade = "kN" if gl in [DOF_U, DOF_V] else "kN.mm"
    print(f"Nó {no+1}, {nomes_gdl[gl]}: {R[idx]: .6e} {unidade}")

print("\nESFORÇOS RESUMIDOS POR ELEMENTO")
print("Elem      N [kN]       V [kN]      M(0) [kN.m]    M(L) [kN.m]")
for e, res in enumerate(resultados_elementos):
    print(
        f"{e+1:2d}  "
        f"{res['N'][0]:12.6f} "
        f"{res['V'][0]:12.6f} "
        f"{res['M'][0]/1000.0:14.6f} "
        f"{res['M'][-1]/1000.0:14.6f}"
    )
