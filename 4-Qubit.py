import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Parameters (physically motivated)
omega = 3.83
J_intra = 2.8      # intra-pair coupling (task substitution within area)
J_inter = 1.5      # inter-pair coupling (non-local automation effect)
T = 20.0
n_points = 300
t = np.linspace(0, T, n_points)

# 4-qubit Pauli operators
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2), qt.qeye(2))
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2), qt.qeye(2))
sz3 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz(), qt.qeye(2))
sz4 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), qt.sigmaz())

sx1 = qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(2), qt.qeye(2))
sx2 = qt.tensor(qt.qeye(2), qt.sigmax(), qt.qeye(2), qt.qeye(2))
sx3 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmax(), qt.qeye(2))
sx4 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), qt.sigmax())

sy1 = qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(2), qt.qeye(2))
sy2 = qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(2), qt.qeye(2))
sy3 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmay(), qt.qeye(2))
sy4 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), qt.sigmay())

# Hamiltonian
H = omega * (sz1 + sz2 + sz3 + sz4) \
    + J_intra * (sx1*sx2 + sy1*sy2 + sz1*sz2) \
    + J_intra * (sx3*sx4 + sy3*sy4 + sz3*sz4) \
    + J_inter * (sx1*sx3 + sy1*sy3 + sz1*sz3)

# Fully separable initial state
psi0 = qt.tensor(
    (qt.basis(2,0) + qt.basis(2,1))/np.sqrt(2), qt.basis(2,0),
    (qt.basis(2,0) + qt.basis(2,1))/np.sqrt(2), qt.basis(2,0)
)

result = qt.mesolve(H, psi0, t)

# Entanglement measures (pair 1-2)
conc = [qt.concurrence(state.ptrace([0,1])) for state in result.states]
vn   = [qt.entropy_vn(state.ptrace([0,1]).ptrace(0)) for state in result.states]

print(f"Max Concurrence (pair 1-2): {max(conc):.4f}")
print(f"Max von Neumann entropy (qubit 1): {max(vn):.4f}")

plt.figure(figsize=(10,5))
plt.plot(t, conc, 'b-', lw=2.5, label='Concurrence (pair 1-2)')
plt.plot(t, vn, 'r--', lw=2.5, label='Von Neumann entropy (qubit 1)')
plt.xlabel('Time (s)')
plt.ylabel('Entanglement Measure')
plt.title('4-Qubit Case: Dynamic Entanglement from Separable Initial State')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
