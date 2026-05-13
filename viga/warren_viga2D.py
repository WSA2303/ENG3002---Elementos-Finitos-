import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# WARREN ANALISADA COM ELEMENTOS DE VIGA/PÓRTICO 2D
# Elemento de Euler-Bernoulli 2D:
# GDL por nó = {u_x, u_y, theta_z}
#
# SISTEMA DE UNIDADES:
# comprimento: mm
# força: kN
# E: kN/mm²
# A: mm²
# I: mm^4
# momentos: kN.mm
# ============================================================

# -------------------------
# Configurações gerais
# -------------------------
MOSTRAR_GRAFICOS = True
NPTS_ELEMENTO = 50
PASTA_SAIDA = Path("Output_Warren_Viga2D")
PASTA_SAIDA.mkdir(exist_ok=True)

# Graus de liberdade
GLX = 0
GLY = 1
GLT = 2  # theta_z

# ============================================================
# DADOS DA MALHA DA WARREN
# ============================================================

COORD = np.array([
    [0.0,    0.0],     # nó 1
    [3000.0, 0.0],     # nó 2
    [6000.0, 0.0],     # nó 3
    [9000.0, 0.0],     # nó 4
    [1500.0, 2500.0],  # nó 5
    [4500.0, 2500.0],  # nó 6
    [7500.0, 2500.0]   # nó 7
], dtype=float)

CONEC = np.array([
    [1, 2],   # elemento 1
    [2, 3],   # elemento 2
    [3, 4],   # elemento 3
    [5, 6],   # elemento 4
    [6, 7],   # elemento 5
    [1, 5],   # elemento 6
    [5, 2],   # elemento 7
    [2, 6],   # elemento 8
    [6, 3],   # elemento 9
    [3, 7],   # elemento 10
    [7, 4]    # elemento 11
], dtype=int) - 1

NNOS = COORD.shape[0]
NEL = CONEC.shape[0]
NGLN = 3
NGL = NNOS * NGLN

# ============================================================
# PROPRIEDADES DOS ELEMENTOS
# ============================================================
# ATENÇÃO:
# Como agora os elementos são de viga/pórtico 2D, além de E e A,
# é obrigatório fornecer I.
#
# Para manter coerência com A = 400 mm², adotei inicialmente
# uma seção quadrada 20 mm x 20 mm:
#
# A = 20*20 = 400 mm²
# I = b*h^3/12 = 20*20^3/12 = 13333.333 mm^4
#
# Altere I_ELEM se a seção real for outra.
# ============================================================

E_ELEM = 200.0 * np.ones(NEL)        # kN/mm²
A_ELEM = 400.0 * np.ones(NEL)        # mm²
I_ELEM = (20.0 * 20.0**3 / 12.0) * np.ones(NEL)  # mm^4

# ============================================================
# FORÇAS NODAIS
# cada linha: [nó, GDL, valor]
# força em kN; momento em kN.mm
# ============================================================

FORCAS = np.array([
    [2, GLY, -40.0],
    [3, GLY, -40.0]
], dtype=float)

# ============================================================
# VÍNCULOS
# cada linha: [nó, GDL, valor_prescrito]
# Nó 1: apoio fixo em x e y
# Nó 4: apoio móvel em y
#
# As rotações ficam livres, como em um apoio simples.
# ============================================================

VINC = np.array([
    [1, GLX, 0.0],
    [1, GLY, 0.0],
    [4, GLY, 0.0]
], dtype=float)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def matriz_rigidez_local(E, A, I, L):
    """Matriz de rigidez local 6x6 do elemento de pórtico 2D.
    Ordem local: [u1, v1, theta1, u2, v2, theta2]
    """
    EA_L = E * A / L
    EI = E * I

    k = np.array([
        [ EA_L,          0.0,          0.0, -EA_L,          0.0,          0.0],
        [ 0.0,    12*EI/L**3,   6*EI/L**2,  0.0,   -12*EI/L**3,   6*EI/L**2],
        [ 0.0,     6*EI/L**2,     4*EI/L,  0.0,    -6*EI/L**2,     2*EI/L],
        [-EA_L,          0.0,          0.0,  EA_L,          0.0,          0.0],
        [ 0.0,   -12*EI/L**3,  -6*EI/L**2,  0.0,    12*EI/L**3,  -6*EI/L**2],
        [ 0.0,     6*EI/L**2,     2*EI/L,  0.0,    -6*EI/L**2,     4*EI/L],
    ], dtype=float)

    return k


def matriz_transformacao(c, s):
    """Transforma deslocamentos globais em deslocamentos locais: u_local = T @ u_global."""
    T = np.array([
        [ c,  s, 0.0, 0.0, 0.0, 0.0],
        [-s,  c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c,  s, 0.0],
        [0.0, 0.0, 0.0, -s,  c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    return T


def gdl_elemento(no1, no2):
    """Retorna os 6 GDL globais de um elemento de 2 nós."""
    return np.array([
        no1*NGLN + GLX,
        no1*NGLN + GLY,
        no1*NGLN + GLT,
        no2*NGLN + GLX,
        no2*NGLN + GLY,
        no2*NGLN + GLT,
    ], dtype=int)


def escala_automatica(coord, desloc, fracao=0.15):
    """Calcula fator de ampliação automático para a deformada."""
    dx = np.ptp(coord[:, 0])
    dy = np.ptp(coord[:, 1])
    Lref = max(dx, dy)

    ux = desloc[0::3]
    uy = desloc[1::3]
    desloc_nodal = np.sqrt(ux**2 + uy**2)
    umax = np.max(desloc_nodal)

    if umax < 1.0e-14:
        return 1.0

    return fracao * Lref / umax


def hermite_viga(x, L, v1, th1, v2, th2):
    """Campo transversal v(x) usando Hermite cúbico."""
    xi = x / L
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L * (xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L * (-xi**2 + xi**3)
    return N1*v1 + N2*th1 + N3*v2 + N4*th2


def esforcos_internos(E, A, I, L, u_local, npts=NPTS_ELEMENTO):
    """Calcula N(x), M(x), V(x) do elemento a partir dos deslocamentos locais."""
    u1, v1, th1, u2, v2, th2 = u_local

    x = np.linspace(0.0, L, npts)
    xi = x / L

    # Esforço normal constante
    N = E * A * (u2 - u1) / L
    N_array = N * np.ones_like(x)

    # Curvatura pela interpolação de Hermite
    d2v_dx2 = (
        (-6.0 + 12.0*xi)*v1
        + L*(-4.0 + 6.0*xi)*th1
        + (6.0 - 12.0*xi)*v2
        + L*(-2.0 + 6.0*xi)*th2
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


# ============================================================
# PRÉ-PROCESSAMENTO GEOMÉTRICO
# ============================================================

L_ELEM = np.zeros(NEL)
ANG_ELEM = np.zeros(NEL)
C_ELEM = np.zeros(NEL)
S_ELEM = np.zeros(NEL)

for e in range(NEL):
    no1, no2 = CONEC[e]
    dx = COORD[no2, 0] - COORD[no1, 0]
    dy = COORD[no2, 1] - COORD[no1, 1]
    L = np.hypot(dx, dy)

    if L <= 0.0:
        raise ValueError(f"Elemento {e+1} possui comprimento nulo.")

    L_ELEM[e] = L
    ANG_ELEM[e] = np.arctan2(dy, dx)
    C_ELEM[e] = dx / L
    S_ELEM[e] = dy / L

# ============================================================
# MONTAGEM DA MATRIZ GLOBAL
# ============================================================

KG = np.zeros((NGL, NGL), dtype=float)
FG = np.zeros(NGL, dtype=float)

K_LOCAL_LISTA = []
T_LISTA = []

for e in range(NEL):
    no1, no2 = CONEC[e]
    L = L_ELEM[e]
    c = C_ELEM[e]
    s = S_ELEM[e]

    k_local = matriz_rigidez_local(E_ELEM[e], A_ELEM[e], I_ELEM[e], L)
    T = matriz_transformacao(c, s)
    k_global = T.T @ k_local @ T

    K_LOCAL_LISTA.append(k_local)
    T_LISTA.append(T)

    dofs = gdl_elemento(no1, no2)
    KG[np.ix_(dofs, dofs)] += k_global

# Vetor de forças
for carga in FORCAS:
    no = int(carga[0]) - 1
    gl = int(carga[1])
    valor = carga[2]
    FG[no*NGLN + gl] += valor

KG_ORIG = KG.copy()
FG_ORIG = FG.copy()

# ============================================================
# APLICAÇÃO DOS VÍNCULOS
# ============================================================

for vinc in VINC:
    no = int(vinc[0]) - 1
    gl = int(vinc[1])
    valor = float(vinc[2])

    idx = no*NGLN + gl

    # permite deslocamento prescrito não-nulo
    FG -= KG[:, idx] * valor

    KG[idx, :] = 0.0
    KG[:, idx] = 0.0
    KG[idx, idx] = 1.0
    FG[idx] = valor

# ============================================================
# SOLUÇÃO
# ============================================================

try:
    U = np.linalg.solve(KG, FG)
except np.linalg.LinAlgError as erro:
    raise RuntimeError(
        "A matriz global ficou singular. Verifique vínculos, rigidez à flexão I "
        "e se existem modos de corpo rígido."
    ) from erro

R = KG_ORIG @ U - FG_ORIG

ESCALA = escala_automatica(COORD, U, fracao=0.15)

# ============================================================
# PÓS-PROCESSAMENTO
# ============================================================

RESULTADOS = []

for e in range(NEL):
    no1, no2 = CONEC[e]
    dofs = gdl_elemento(no1, no2)

    u_global = U[dofs]
    u_local = T_LISTA[e] @ u_global

    f_local = K_LOCAL_LISTA[e] @ u_local

    xloc, N, M, V = esforcos_internos(
        E_ELEM[e], A_ELEM[e], I_ELEM[e], L_ELEM[e], u_local
    )

    RESULTADOS.append({
        "u_local": u_local,
        "f_local": f_local,
        "x": xloc,
        "N": N,
        "M": M,
        "V": V,
    })

# ============================================================
# SAÍDA NA TELA
# ============================================================

print("\n=====================================================")
print("RESULTADOS - WARREN COM ELEMENTOS DE VIGA/PÓRTICO 2D")
print("Unidades: kN, mm, rad")
print("=====================================================")

print(f"\nFator de ampliação automático da deformada: {ESCALA:.3f}")

print("\nDESLOCAMENTOS NODAIS")
print("Nó        ux [mm]        uy [mm]      theta [rad]")
for n in range(NNOS):
    ux = U[n*NGLN + GLX]
    uy = U[n*NGLN + GLY]
    th = U[n*NGLN + GLT]
    print(f"{n+1:2d}  {ux:14.6e} {uy:14.6e} {th:14.6e}")

print("\nREAÇÕES DE APOIO")
print("Nó      GDL        Reação")
nomes = {GLX: "Rx [kN]", GLY: "Ry [kN]", GLT: "Mz [kN.mm]"}
for vinc in VINC:
    no = int(vinc[0]) - 1
    gl = int(vinc[1])
    idx = no*NGLN + gl
    print(f"{no+1:2d}   {nomes[gl]:10s} {R[idx]:14.6e}")

print("\nESFORÇOS LOCAIS NODAIS RESISTENTES POR ELEMENTO")
print("Elem        N1          V1          M1          N2          V2          M2")
for e, res in enumerate(RESULTADOS, start=1):
    print(f"{e:2d}  " + " ".join(f"{v:11.4e}" for v in res["f_local"]))

print("\nRESUMO DOS ESFORÇOS INTERNOS")
print("Elem       N [kN]      V [kN]      M_i [kN.mm]   M_j [kN.mm]")
for e, res in enumerate(RESULTADOS, start=1):
    N0 = res["N"][0]
    V0 = res["V"][0]
    M0 = res["M"][0]
    ML = res["M"][-1]
    print(f"{e:2d}  {N0:12.4f} {V0:12.4f} {M0:14.4f} {ML:14.4f}")

# ============================================================
# RELATÓRIO
# ============================================================

relatorio = PASTA_SAIDA / "relatorio_warren_viga2D.txt"

with open(relatorio, "w", encoding="utf-8") as arq:
    arq.write("=====================================================\n")
    arq.write("RELATÓRIO - WARREN ANALISADA COMO VIGA/PÓRTICO 2D\n")
    arq.write("Sistema de unidades: kN - mm\n")
    arq.write("=====================================================\n\n")

    arq.write("OBSERVAÇÃO IMPORTANTE\n")
    arq.write("Este modelo usa elementos de viga/pórtico Euler-Bernoulli 2D.\n")
    arq.write("Portanto, cada nó possui ux, uy e theta_z, e as ligações entre barras são rígidas.\n")
    arq.write("Isso NÃO é exatamente igual a uma treliça ideal rotulada, a menos que sejam usadas liberações de momento.\n\n")

    arq.write("1. DADOS GERAIS\n")
    arq.write(f"NNOS = {NNOS}\n")
    arq.write(f"NEL  = {NEL}\n")
    arq.write(f"NGLN = {NGLN}\n")
    arq.write(f"NGL  = {NGL}\n")
    arq.write(f"Escala automática da deformada = {ESCALA:.6f}\n\n")

    arq.write("2. DESLOCAMENTOS NODAIS\n")
    arq.write("Nó        ux [mm]        uy [mm]      theta [rad]\n")
    for n in range(NNOS):
        ux = U[n*NGLN + GLX]
        uy = U[n*NGLN + GLY]
        th = U[n*NGLN + GLT]
        arq.write(f"{n+1:2d}  {ux:14.6e} {uy:14.6e} {th:14.6e}\n")
    arq.write("\n")

    arq.write("3. REAÇÕES DE APOIO\n")
    arq.write("Nó      GDL        Reação\n")
    for vinc in VINC:
        no = int(vinc[0]) - 1
        gl = int(vinc[1])
        idx = no*NGLN + gl
        arq.write(f"{no+1:2d}   {nomes[gl]:10s} {R[idx]:14.6e}\n")
    arq.write("\n")

    arq.write("4. PROPRIEDADES DOS ELEMENTOS\n")
    arq.write("Elem  nó_i nó_j     L [mm]       ang [graus]     E [kN/mm²]     A [mm²]      I [mm^4]\n")
    for e in range(NEL):
        no1, no2 = CONEC[e]
        arq.write(
            f"{e+1:4d} {no1+1:5d} {no2+1:5d} "
            f"{L_ELEM[e]:12.6f} {np.rad2deg(ANG_ELEM[e]):14.6f} "
            f"{E_ELEM[e]:14.6f} {A_ELEM[e]:12.6f} {I_ELEM[e]:14.6f}\n"
        )
    arq.write("\n")

    arq.write("5. ESFORÇOS LOCAIS NODAIS RESISTENTES POR ELEMENTO\n")
    arq.write("Elem        N1          V1          M1          N2          V2          M2\n")
    for e, res in enumerate(RESULTADOS, start=1):
        arq.write(f"{e:2d}  " + " ".join(f"{v:14.6e}" for v in res["f_local"]) + "\n")
    arq.write("\n")

    arq.write("6. RESUMO DOS ESFORÇOS INTERNOS\n")
    arq.write("Elem       N [kN]      V [kN]      M_i [kN.mm]   M_j [kN.mm]\n")
    for e, res in enumerate(RESULTADOS, start=1):
        N0 = res["N"][0]
        V0 = res["V"][0]
        M0 = res["M"][0]
        ML = res["M"][-1]
        arq.write(f"{e:2d}  {N0:12.4f} {V0:12.4f} {M0:14.4f} {ML:14.4f}\n")

print(f"\nRelatório salvo em: {relatorio.resolve()}")

# CSV dos deslocamentos
csv_desloc = PASTA_SAIDA / "deslocamentos_warren_viga2D.csv"
with open(csv_desloc, "w", encoding="utf-8") as arq:
    arq.write("no,ux_mm,uy_mm,theta_rad\n")
    for n in range(NNOS):
        arq.write(f"{n+1},{U[n*NGLN+GLX]:.10e},{U[n*NGLN+GLY]:.10e},{U[n*NGLN+GLT]:.10e}\n")

# CSV de esforços por elemento
for e, res in enumerate(RESULTADOS, start=1):
    csv_elem = PASTA_SAIDA / f"esforcos_elemento_{e:02d}.csv"
    with open(csv_elem, "w", encoding="utf-8") as arq:
        arq.write("x_local_mm,N_kN,V_kN,M_kNmm\n")
        for x, n, v, m in zip(res["x"], res["N"], res["V"], res["M"]):
            arq.write(f"{x:.10e},{n:.10e},{v:.10e},{m:.10e}\n")

# ============================================================
# GRÁFICO DA ESTRUTURA ORIGINAL E DEFORMADA
# ============================================================

plt.figure(figsize=(11, 5))
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

# Indeformada
for e in range(NEL):
    no1, no2 = CONEC[e]
    x = COORD[[no1, no2], 0]
    y = COORD[[no1, no2], 1]
    plt.plot(x, y, "k--", linewidth=1.2)
    plt.plot(x, y, "ko", markersize=4)

# Deformada com interpolação de viga
for e in range(NEL):
    no1, no2 = CONEC[e]

    L = L_ELEM[e]
    c = C_ELEM[e]
    s = S_ELEM[e]
    x0, y0 = COORD[no1]

    u1, v1, th1, u2, v2, th2 = RESULTADOS[e]["u_local"]

    xloc = np.linspace(0.0, L, NPTS_ELEMENTO)
    xi = xloc / L

    u_axial = (1.0 - xi)*u1 + xi*u2
    v_trans = hermite_viga(xloc, L, v1, th1, v2, th2)

    x_def_local = xloc + ESCALA * u_axial
    y_def_local = ESCALA * v_trans

    x_def_global = x0 + c*x_def_local - s*y_def_local
    y_def_global = y0 + s*x_def_local + c*y_def_local

    plt.plot(x_def_global, y_def_global, linewidth=2.0)

for n in range(NNOS):
    plt.text(COORD[n, 0], COORD[n, 1], f" {n+1}", fontsize=9)

plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(f"Warren original e deformada com elemento de viga 2D | escala = {ESCALA:.2f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "warren_deformada_viga2D.png", dpi=300)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

# ============================================================
# GRÁFICOS DOS ESFORÇOS POR ELEMENTO
# ============================================================

plt.figure(figsize=(9, 5))
for e, res in enumerate(RESULTADOS, start=1):
    plt.plot(res["x"], res["N"], label=f"E{e}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [mm]")
plt.ylabel("N [kN]")
plt.title("Esforço normal por elemento")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_N_warren_viga2D.png", dpi=300)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(9, 5))
for e, res in enumerate(RESULTADOS, start=1):
    plt.plot(res["x"], res["V"], label=f"E{e}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [mm]")
plt.ylabel("V [kN]")
plt.title("Esforço cortante por elemento")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_V_warren_viga2D.png", dpi=300)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(9, 5))
for e, res in enumerate(RESULTADOS, start=1):
    plt.plot(res["x"], res["M"], label=f"E{e}")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("x local [mm]")
plt.ylabel("M [kN.mm]")
plt.title("Momento fletor por elemento")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "diagrama_M_warren_viga2D.png", dpi=300)
if MOSTRAR_GRAFICOS:
    plt.show()
else:
    plt.close()

print(f"Arquivos salvos na pasta: {PASTA_SAIDA.resolve()}")
