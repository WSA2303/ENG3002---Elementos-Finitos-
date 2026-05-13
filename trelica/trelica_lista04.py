import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# TRELIÇA PLANA 2D - MÉTODO DOS ELEMENTOS FINITOS
# ============================================================

# Graus de liberdade por nó
GLX = 0
GLY = 1

# ============================================================
# PARÂMETROS DA MALHA
# ============================================================
NEL   = 9   # número de elementos
NGLN  = 2   # número de graus de liberdade por nó
NNOE  = 2   # número de nós por elemento
NNOS  = 6   # número de nós
NVINC = 3   # número de vínculos
NFORC = 2   # número de forças

# Número total de graus de liberdade da estrutura
NGL = NNOS * NGLN

# ============================================================
# PROPRIEDADES DOS ELEMENTOS
# Agora cada elemento pode ter seu próprio E e sua própria A
# ============================================================
# Para usar materiais/seções diferentes, basta mudar os valores.

E_ELEM = np.array([
    205e9,
    205e9,
    205e9,
    205e9,
    205e9,
    205e9,
    205e9,
    205e9,
    205e9
], dtype=float)

A_ELEM = np.array([
    0.00146373,
    0.00146373,
    0.00146373,
    0.00146373,
    0.00146373,
    0.00146373,
    0.00146373,
    0.00146373,
    0.00146373
], dtype=float)

# ============================================================
# COORDENADAS NODAIS
# cada linha: [x, y]
# ============================================================
COORD = np.array([
    [0.0, 0.0],   # nó 1
    [0.0, 1.0],   # nó 2
    [0.0, 2.0],   # nó 3
    [0.5, 1.5],   # nó 4
    [1.0, 1.0],   # nó 5
    [0.5, 0.5]    # nó 6
], dtype=float)

# ============================================================
# CONECTIVIDADE DOS ELEMENTOS
# cada linha: [nó inicial, nó final]
# observação: subtrai-se 1 para usar índices do Python
# ============================================================
CONEC = np.array([
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

# ============================================================
# FORÇAS NODAIS
# cada linha: [nó, grau de liberdade, valor]
# ============================================================
FORCAS = np.array([
    [3, GLX,  10000.0],
    [6, GLY, -10000.0]
], dtype=float)

# ============================================================
# VÍNCULOS
# cada linha: [nó, grau de liberdade]
# ============================================================
VINC = np.array([
    [1, GLX],
    [1, GLY],
    [5, GLY]
], dtype=int)

# ============================================================
# DADOS GEOMÉTRICOS DOS ELEMENTOS
# DADOS_ELEM[e,0] = ângulo
# DADOS_ELEM[e,1] = comprimento
# ============================================================
DADOS_ELEM = np.zeros((NEL, 2), dtype=float)

for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    dx = COORD[no2, 0] - COORD[no1, 0]
    dy = COORD[no2, 1] - COORD[no1, 1]

    ANG = np.arctan2(dy, dx)
    COMP = np.sqrt(dx**2 + dy**2)

    DADOS_ELEM[e, 0] = ANG
    DADOS_ELEM[e, 1] = COMP

# ============================================================
# INICIALIZAÇÃO DA MATRIZ GLOBAL E DO VETOR GLOBAL
# ============================================================
KG = np.zeros((NGL, NGL), dtype=float)
FG = np.zeros(NGL, dtype=float)

# ============================================================
# MONTAGEM DA MATRIZ DE RIGIDEZ GLOBAL
# ============================================================
for e in range(NEL):
    ANG = DADOS_ELEM[e, 0]
    COMP = DADOS_ELEM[e, 1]

    c = np.cos(ANG)
    s = np.sin(ANG)

    E = E_ELEM[e]
    A = A_ELEM[e]

    k = (E * A) / COMP

    KE = k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ], dtype=float)

    # Superposição de KE em KG
    for j in range(NNOE):
        for gl_j in range(NGLN):
            col_local = j * NGLN + gl_j
            col_global = CONEC[e, j] * NGLN + gl_j

            for l in range(NNOE):
                for gl_l in range(NGLN):
                    lin_local = l * NGLN + gl_l
                    lin_global = CONEC[e, l] * NGLN + gl_l

                    KG[lin_global, col_global] += KE[lin_local, col_local]

# ============================================================
# MONTAGEM DO VETOR GLOBAL DE FORÇAS
# ============================================================
for i in range(NFORC):
    no = int(FORCAS[i, 0]) - 1
    gl = int(FORCAS[i, 1])
    valor = FORCAS[i, 2]

    indice_global = no * NGLN + gl
    FG[indice_global] += valor

# Guardar sistema original para cálculo das reações
KG_ORIG = KG.copy()
FG_ORIG = FG.copy()

# ============================================================
# APLICAÇÃO DOS VÍNCULOS
# ============================================================
for i in range(NVINC):
    no = VINC[i, 0] - 1
    gl = VINC[i, 1]

    indice_global = no * NGLN + gl

    KG[indice_global, :] = 0.0
    KG[:, indice_global] = 0.0
    KG[indice_global, indice_global] = 1.0
    FG[indice_global] = 0.0

# ============================================================
# SOLUÇÃO DO SISTEMA
# ============================================================
U = np.linalg.solve(KG, FG)   # deslocamentos [m]
U_MM = U * 1e3                # deslocamentos [mm]

# ============================================================
# FATOR DE AMPLIAÇÃO AUTOMÁTICO DA DEFORMADA
# ============================================================

# dimensões características da estrutura
xmin = np.min(COORD[:, 0])
xmax = np.max(COORD[:, 0])
ymin = np.min(COORD[:, 1])
ymax = np.max(COORD[:, 1])

LX = xmax - xmin
LY = ymax - ymin

# comprimento de referência da estrutura
LREF = max(LX, LY)

# deslocamento resultante em cada nó
U_NODAL = np.zeros(NNOS)
for no in range(NNOS):
    ux = U[NGLN * no]
    uy = U[NGLN * no + 1]
    U_NODAL[no] = np.sqrt(ux**2 + uy**2)

UMAX = np.max(U_NODAL)

# percentual do tamanho da estrutura que a deformada deve ocupar visualmente
FRACAO_VISUAL = 0.15  # 15% do tamanho da estrutura

# evita divisão por zero
if UMAX > 1e-14:
    ESCALA = FRACAO_VISUAL * LREF / UMAX
else:
    ESCALA = 1.0

print(f"\nFator de ampliação automático da deformada: {ESCALA:.3f}")

# ============================================================
# REAÇÕES DE APOIO
# ============================================================
R = KG_ORIG @ U - FG_ORIG

# ============================================================
# CÁLCULO DOS RESULTADOS EM CADA BARRA
# ============================================================
# RESULT_ELEM[e,0] = deformação axial
# RESULT_ELEM[e,1] = tensão axial [Pa]
# RESULT_ELEM[e,2] = esforço normal [N]
RESULT_ELEM = np.zeros((NEL, 3), dtype=float)

for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    ANG = DADOS_ELEM[e, 0]
    COMP = DADOS_ELEM[e, 1]

    c = np.cos(ANG)
    s = np.sin(ANG)

    E = E_ELEM[e]
    A = A_ELEM[e]

    # vetor de deslocamentos do elemento em coordenadas globais
    Ue = np.array([
        U[NGLN * no1],
        U[NGLN * no1 + 1],
        U[NGLN * no2],
        U[NGLN * no2 + 1]
    ], dtype=float)

    # alongamento axial do elemento
    delta = np.array([-c, -s, c, s]) @ Ue

    # deformação axial
    eps = delta / COMP

    # tensão axial
    sig = E * eps

    # esforço normal
    N = A * sig

    RESULT_ELEM[e, 0] = eps
    RESULT_ELEM[e, 1] = sig
    RESULT_ELEM[e, 2] = N

# ============================================================
# SAÍDA NA TELA
# ============================================================
print("Deslocamentos nodais [m]:")
print(U)

print("\nDeslocamentos nodais [mm]:")
print(U_MM)

print("\nResultados por elemento:")
for e in range(NEL):
    eps = RESULT_ELEM[e, 0]
    sig = RESULT_ELEM[e, 1]
    N = RESULT_ELEM[e, 2]

    estado = "Tração" if N > 0 else "Compressão" if N < 0 else "Nulo"

    print(f"Elemento {e+1:2d}: "
          f"eps = {eps: .6e} | "
          f"sigma = {sig/1e6: .6f} MPa | "
          f"N = {N: .3f} N | {estado}")

print("\nReações de apoio:")
for i in range(NVINC):
    no = VINC[i, 0] - 1
    gl = VINC[i, 1]
    idx = no * NGLN + gl

    direcao = "X" if gl == GLX else "Y"
    print(f"Nó {no+1}, GL {direcao}: {R[idx]: .3f} N")

# ============================================================
# GERAÇÃO DO RELATÓRIO
# ============================================================
PASTA_OUTPUT = Path("Output")
PASTA_OUTPUT.mkdir(exist_ok=True)  # cria a pasta se não existir

nome_arquivo = PASTA_OUTPUT / "relatorio_trelica.txt"

with open(nome_arquivo, "w", encoding="utf-8") as arq:
    arq.write("=====================================================\n")
    arq.write("RELATÓRIO DE RESULTADOS - TRELIÇA PLANA 2D\n")
    arq.write("=====================================================\n\n")

    arq.write("1. DADOS GERAIS\n")
    arq.write(f"NEL   = {NEL}\n")
    arq.write(f"NNOS  = {NNOS}\n")
    arq.write(f"NGLN  = {NGLN}\n")
    arq.write(f"NNOE  = {NNOE}\n")
    arq.write(f"NFORC = {NFORC}\n")
    arq.write(f"NVINC = {NVINC}\n\n")

    arq.write("2. DESLOCAMENTOS NODAIS\n")
    for no in range(NNOS):
        ux = U[NGLN * no]
        uy = U[NGLN * no + 1]
        arq.write(f"Nó {no+1:2d}: "
                  f"Ux = {ux: .6e} m ({ux*1e3: .6f} mm), "
                  f"Uy = {uy: .6e} m ({uy*1e3: .6f} mm)\n")
    arq.write("\n")

    arq.write("3. REAÇÕES DE APOIO\n")
    for i in range(NVINC):
        no = VINC[i, 0] - 1
        gl = VINC[i, 1]
        idx = no * NGLN + gl
        direcao = "X" if gl == GLX else "Y"
        arq.write(f"Nó {no+1:2d}, GL {direcao}: {R[idx]: .6f} N\n")
    arq.write("\n")

    arq.write("4. RESULTADOS NOS ELEMENTOS\n")
    arq.write("Elem | Nó i | Nó j | Comprimento [m] | E [Pa] | A [m²] | deformação | tensão [MPa] | esforço normal [N] | estado\n")
    for e in range(NEL):
        no1 = CONEC[e, 0] + 1
        no2 = CONEC[e, 1] + 1
        COMP = DADOS_ELEM[e, 1]
        E = E_ELEM[e]
        A = A_ELEM[e]
        eps = RESULT_ELEM[e, 0]
        sig_mpa = RESULT_ELEM[e, 1] / 1e6
        N = RESULT_ELEM[e, 2]
        estado = "Tração" if N > 0 else "Compressão" if N < 0 else "Nulo"

        arq.write(f"{e+1:4d} | {no1:4d} | {no2:4d} | "
                  f"{COMP: .6f} | {E: .6e} | {A: .6e} | "
                  f"{eps: .6e} | {sig_mpa: .6f} | {N: .6f} | {estado}\n")

print(f"\nRelatório gerado com sucesso em: {nome_arquivo}")

# ============================================================
# CÁLCULO DAS COORDENADAS DEFORMADAS
# ============================================================
XDEF = np.zeros(NNOS)
YDEF = np.zeros(NNOS)

for no in range(NNOS):
    XDEF[no] = COORD[no, 0] + ESCALA * U[NGLN * no]
    YDEF[no] = COORD[no, 1] + ESCALA * U[NGLN * no + 1]

COORD_DEF = np.column_stack((XDEF, YDEF))

# ============================================================
# PLOTAGEM DA ESTRUTURA ORIGINAL E DEFORMADA
# ============================================================
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.axis('off')

# Estrutura original
for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    x = [COORD[no1, 0], COORD[no2, 0]]
    y = [COORD[no1, 1], COORD[no2, 1]]

    plt.plot(x, y, linewidth=4)
    plt.plot(x, y, 'o')

# Estrutura deformada
for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    xdef = [COORD_DEF[no1, 0], COORD_DEF[no2, 0]]
    ydef = [COORD_DEF[no1, 1], COORD_DEF[no2, 1]]

    plt.plot(xdef, ydef, linewidth=1)
    plt.plot(xdef, ydef, 'o')

plt.title(f"Treliça original e deformada (escala automática = {ESCALA:.2f})")
plt.show()