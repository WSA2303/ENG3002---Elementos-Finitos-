import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO VISUAL DOS GRÁFICOS
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",      # fonte matemática estilo LaTeX
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "figure.dpi": 120,
})

def formatar_grafico(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.minorticks_on()

    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.8)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

# ============================================================
# DADOS DO PROBLEMA
# ============================================================

L = 2500.0          # comprimento do elemento [mm]
E = 210000.0        # módulo de elasticidade [N/mm²]
I = 9.8e6           # momento de inércia [mm^4]

EI = E * I

# Deslocamentos nodais no sistema local
u1 = 0.0350         # [mm]
v1 = 5.54           # [mm]
theta1_graus = 0.173  # [graus]

u2 = 0.102          # [mm]
v2 = 17.0           # [mm]
theta2_graus = 0.301  # [graus]

# Conversão das rotações para radianos
theta1 = np.deg2rad(theta1_graus)
theta2 = np.deg2rad(theta2_graus)

# ============================================================
# PASTA PARA SALVAR OS GRÁFICOS
# ============================================================

pasta_saida = Path("outputs")
pasta_saida.mkdir(exist_ok=True)

# ============================================================
# COORDENADA AO LONGO DO ELEMENTO
# ============================================================

x = np.linspace(0, L, 300)
xi = x / L

# ============================================================
# ESFORÇO NORMAL
# ============================================================

# Como a área A não foi fornecida, calcula-se N/A
N_por_A = E * (u2 - u1) / L
N_por_A_array = N_por_A * np.ones_like(x)

# ============================================================
# MOMENTO FLETOR
# ============================================================

# Segunda derivada da interpolação cúbica de Hermite
d2v_dx2 = (
    (-6 + 12*xi) * v1
    + L * (-4 + 6*xi) * theta1
    + (6 - 12*xi) * v2
    + L * (-2 + 6*xi) * theta2
) / L**2

# Momento fletor: M = EI * d²v/dx²
M = EI * d2v_dx2  # [N.mm]

# Convertendo para kN.m
M_kNm = M / 1e6

# ============================================================
# ESFORÇO CORTANTE
# ============================================================

# Terceira derivada da interpolação de Hermite
d3v_dx3 = (
    12*v1
    + 6*L*theta1
    - 12*v2
    + 6*L*theta2
) / L**3

# Convenção adotada: V = -dM/dx = -EI * d³v/dx³
V = -EI * d3v_dx3  # [N]

# Convertendo para kN
V_kN = V / 1000
V_kN_array = V_kN * np.ones_like(x)

# ============================================================
# CORES DOS GRÁFICOS
# ============================================================

cor_normal = "#C44E52"    # vermelho
cor_momento = "#55A868"   # verde
cor_cortante = "#8172B2"  # roxo

# ============================================================
# GRÁFICO DO ESFORÇO NORMAL POR ÁREA
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, N_por_A_array, color=cor_normal)

ax.set_xlabel(r"$x\;[\mathrm{mm}]$")
ax.set_ylabel(r"$N/A\;[\mathrm{MPa}]$")
ax.set_title(r"Esforço normal por unidade de área")

formatar_grafico(ax)

fig.tight_layout()
fig.savefig(pasta_saida / "grafico_N_por_A.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# GRÁFICO DO MOMENTO FLETOR
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, M_kNm, color=cor_momento)

ax.set_xlabel(r"$x\;[\mathrm{mm}]$")
ax.set_ylabel(r"$M(x)\;[\mathrm{kN \cdot m}]$")
ax.set_title(r"Momento fletor")

formatar_grafico(ax)

fig.tight_layout()
fig.savefig(pasta_saida / "grafico_M.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# GRÁFICO DO ESFORÇO CORTANTE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, V_kN_array, color=cor_cortante)

ax.set_xlabel(r"$x\;[\mathrm{mm}]$")
ax.set_ylabel(r"$V(x)\;[\mathrm{kN}]$")
ax.set_title(r"Esforço cortante")

formatar_grafico(ax)

fig.tight_layout()
fig.savefig(pasta_saida / "grafico_V.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# IMPRESSÃO DOS RESULTADOS
# ============================================================

print("Resultados:")
print(f"N/A = {N_por_A:.4f} MPa")
print(f"M(0) = {M_kNm[0]:.4f} kN.m")
print(f"M(L) = {M_kNm[-1]:.4f} kN.m")
print(f"V = {V_kN:.4f} kN")

print("\nGráficos salvos na pasta:")
print(pasta_saida.resolve())