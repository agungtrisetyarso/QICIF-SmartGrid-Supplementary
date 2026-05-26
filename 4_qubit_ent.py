import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# ========================== PARAMETERS (tuned to match your figure) ==========================
omega1 = 3.83          # rad/s (inter-area mode)
omega2 = 3.90          # rad/s (local mode)
J_intra = 2.8          # intra-pair coupling
J_inter = 1.5          # inter-pair coupling (non-local task substitution)
R       = 0.15         # reskilling / damping parameter
T = 20.0
n_points = 400
t = np.linspace(0, T, n_points)

np.random.seed(42)     # for reproducibility

# ========================== 4-QUBIT OPERATORS ==========================
# Create tensor operators for 4 qubits
sz = [qt.tensor(*[qt.sigmaz() if i==k else qt.qeye(2) for k in range(4)]) for i in range(4)]
sx = [qt.tensor(*[qt.sigmax() if i==k else qt.qeye(2) for k in range(4)]) for i in range(4)]
sy = [qt.tensor(*[qt.sigmay() if i==k else qt.qeye(2) for k in range(4)]) for i in range(4)]

# Heisenberg Hamiltonian with intra-pair and inter-pair couplings
H = (omega1 * (sz[0] + sz[1]) + omega2 * (sz[2] + sz[3]) +
     J_intra * (sx[0]*sx[1] + sy[0]*sy[1] + sz[0]*sz[1]) +
     J_intra * (sx[2]*sx[3] + sy[2]*sy[3] + sz[2]*sz[3]) +
     J_inter * (sx[0]*sx[2] + sy[0]*sy[2] + sz[0]*sz[2]) +
     R * (sz[0] + sz[1] + sz[2] + sz[3]))

# ========================== SEPARABLE INITIAL STATE ==========================
# Product state: (|0> + |1>)/√2 ⊗ |0> ⊗ (|0> + |1>)/√2 ⊗ |0>   (separable)
psi0 = qt.tensor(
    (qt.basis(2,0) + qt.basis(2,1)).unit(), qt.basis(2,0),
    (qt.basis(2,0) + qt.basis(2,1)).unit(), qt.basis(2,0)
)

# ========================== TIME EVOLUTION ==========================
result = qt.mesolve(H, psi0, t)

# ========================== ENTANGLEMENT MEASURES ==========================
# Concurrence of pair 1-2
conc = []
for state in result.states:
    rho12 = qt.ptrace(state, [0,1])          # reduced density matrix of qubits 1 and 2
    conc.append(qt.concurrence(rho12))

# Von Neumann entropy of qubit 1 (reduced density matrix)
entropy = []
for state in result.states:
    rho1 = qt.ptrace(state, [0])
    entropy.append(qt.entropy_vn(rho1))

# ========================== PLOT (matches your uploaded figure) ==========================
plt.figure(figsize=(10, 6))
plt.plot(t, conc, 'b-', linewidth=2.5, label='Concurrence (pair 1-2)')
plt.plot(t, entropy, 'r--', linewidth=2.5, label='Von Neumann entropy (qubit 1)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Entanglement Measure', fontsize=12)
plt.title('4-Qubit Case: Dynamic Entanglement from Separable Initial State', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
