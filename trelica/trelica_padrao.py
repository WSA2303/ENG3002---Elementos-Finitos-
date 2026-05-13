import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Programa para cálculo de treliças
# ============================================================

f = 300  # fator de escala para visualização da deformada

# Identificação dos eixos de coordenadas
DOF_X = 0
DOF_Y = 1

# Características da malha
noselem = 2      # número de nós por elemento
dfreedom = 2     # número de graus de liberdade por nó
nnods = 6        # número de nós da treliça
nelem = 9        # número de elementos da treliça

# Propriedades
A = 0.00146373   # área da seção em m^2
E = 205e9        # módulo de elasticidade em Pa

# Geometria
coordenadas = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [0.0, 2.0],
    [0.5, 1.5],
    [1.0, 1.0],
    [0.5, 0.5]
], dtype=float)

# conectividades (convertidas para índice começando em 0)
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

# Condições de contorno
nloads = 2
nrestriction = 3

loads = np.array([
    [3, DOF_X,  10000.0],
    [6, DOF_Y, -10000.0]
], dtype=float)

restrictions = np.array([
    [1, DOF_X],
    [1, DOF_Y],
    [5, DOF_Y]
], dtype=int)

# ============================================================
# Propriedades dos elementos
# barra[i,0] = ângulo
# barra[i,1] = comprimento
# ============================================================
barra = np.zeros((nelem, 2), dtype=float)

for i in range(nelem):
    no1 = connections[i, 0]
    no2 = connections[i, 1]

    deltax = coordenadas[no1, 0] - coordenadas[no2, 0]
    deltay = coordenadas[no1, 1] - coordenadas[no2, 1]

    barra[i, 0] = np.arctan2(deltay, deltax)
    barra[i, 1] = np.sqrt(deltax**2 + deltay**2)

# ============================================================
# Matriz de rigidez global e vetor de cargas
# ============================================================
ndof_total = nnods * dfreedom
Kglobal = np.zeros((ndof_total, ndof_total), dtype=float)
Lglobal = np.zeros(ndof_total, dtype=float)

# ============================================================
# Montagem da matriz de rigidez global
# ============================================================
for i in range(nelem):
    theta = barra[i, 0]
    L = barra[i, 1]

    c = np.cos(theta)
    s = np.sin(theta)
    k = (E * A) / L

    # matriz de rigidez do elemento 2D de treliça
    K = np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ], dtype=float) * k

    # superposição
    for j in range(noselem):
        for k_local in range(dfreedom):
            colE = j * dfreedom + k_local
            colG = connections[i, j] * dfreedom + k_local

            for l in range(noselem):
                for m in range(dfreedom):
                    linE = l * dfreedom + m
                    linG = connections[i, l] * dfreedom + m

                    Kglobal[linG, colG] += K[linE, colE]

# ============================================================
# Vetor de carregamento global
# ============================================================
for i in range(nloads):
    node = int(loads[i, 0]) - 1
    dof = int(loads[i, 1])
    value = loads[i, 2]

    linG = node * dfreedom + dof
    Lglobal[linG] += value

# ============================================================
# Aplicação das condições de contorno
# ============================================================
for i in range(nrestriction):
    node = restrictions[i, 0] - 1
    dof = restrictions[i, 1]

    linG = node * dfreedom + dof

    Kglobal[linG, :] = 0.0
    Kglobal[:, linG] = 0.0
    Kglobal[linG, linG] = 1.0
    Lglobal[linG] = 0.0

# ============================================================
# Resultados
# ============================================================
desloc = np.linalg.solve(Kglobal, Lglobal)   # deslocamentos em m
deslocmm = desloc * 1e3                      # deslocamentos em mm

print("Deslocamentos nodais (m):")
print(desloc)

print("\nDeslocamentos nodais (mm):")
print(deslocmm)

# ============================================================
# Plotagem da estrutura original e deformada
# ============================================================
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.axis('off')

# Estrutura original
for i in range(nelem):
    no1 = connections[i, 0]
    no2 = connections[i, 1]

    x_plot = [coordenadas[no1, 0], coordenadas[no2, 0]]
    y_plot = [coordenadas[no1, 1], coordenadas[no2, 1]]

    plt.plot(x_plot, y_plot, linewidth=4)
    plt.plot(x_plot, y_plot, 'o')

# Coordenadas deformadas
defx = np.zeros(nnods)
defy = np.zeros(nnods)

for i in range(nnods):
    defx[i] = coordenadas[i, 0] + f * desloc[2*i]
    defy[i] = coordenadas[i, 1] + f * desloc[2*i + 1]

deformacao = np.column_stack((defx, defy))

# Estrutura deformada
for i in range(nelem):
    no1 = connections[i, 0]
    no2 = connections[i, 1]

    xdef = [deformacao[no1, 0], deformacao[no2, 0]]
    ydef = [deformacao[no1, 1], deformacao[no2, 1]]

    plt.plot(xdef, ydef, linewidth=1)
    plt.plot(xdef, ydef, 'o')

plt.title("Treliça original e deformada")
plt.show()