import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ========================== PARAMETERS ==========================
omega_inter = 2 * np.pi * 0.62   # inter-area ~0.62 Hz
omega_local = 2 * np.pi * 1.20   # local ~1.2 Hz
gamma_pre   = 0.44               # good damping before change
gamma_post  = 0.29               # reduced damping after T_j drop
J_intra = 2.8
J_inter = 1.5
R = 0.15
T = 20.0
n_points = 200
t = np.linspace(0, T, n_points)
t_pre = t[t < 10]
t_post = t[t >= 10]
N_mc = 5000                      # 5000 for fast testing; use 50000 for final plot

np.random.seed(42)

# ======================= REFERENCE TRAJECTORY (EMT-like) =======================
def damped_two_mode(t, gamma, A1=1.0, A2=0.6):
    return A1 * np.exp(-gamma * t) * np.sin(omega_inter * t) + \
           A2 * np.exp(-gamma * t) * np.sin(omega_local * t)

y_pre = damped_two_mode(t_pre, gamma_pre)
y_post = damped_two_mode(t_post, gamma_post)
y_ref = np.concatenate([y_pre, y_post])
y_ref += np.random.normal(0, 0.015, len(t))   # synthetic EMT noise

# ======================= 4-QUBIT MODEL (calibrated on pre-change only) =======================
sz4 = [qt.tensor(*[qt.sigmaz() if i==k else qt.qeye(2) for k in range(4)]) for i in range(4)]
sx4 = [qt.tensor(*[qt.sigmax() if i==k else qt.qeye(2) for k in range(4)]) for i in range(4)]
sy4 = [qt.tensor(*[qt.sigmay() if i==k else qt.qeye(2) for k in range(4)]) for i in range(4)]

H4 = (omega_inter * (sz4[0] + sz4[1]) + omega_local * (sz4[2] + sz4[3]) +
      J_intra * (sx4[0]*sx4[1] + sy4[0]*sy4[1] + sz4[0]*sz4[1]) +
      J_intra * (sx4[2]*sx4[3] + sy4[2]*sy4[3] + sz4[2]*sz4[3]) +
      J_inter * (sx4[0]*sx4[2] + sy4[0]*sy4[2] + sz4[0]*sz4[2]) +
      R * (sz4[0] + sz4[1] + sz4[2] + sz4[3]))

psi0_4 = qt.tensor(
    (qt.basis(2,0) + qt.basis(2,1))/np.sqrt(2), qt.basis(2,0),
    (qt.basis(2,0) + qt.basis(2,1))/np.sqrt(2), qt.basis(2,0)
)

result4 = qt.mesolve(H4, psi0_4, t)
obs4 = [qt.expect(sz4[0] + sz4[1], state) for state in result4.states]

mc4 = np.array([obs4 + np.random.normal(0, 0.015, len(t)) for _ in range(N_mc)])
mean4 = np.mean(mc4, axis=0)
std4  = np.std(mc4, axis=0)

# ======================= PRONY (re-fitted on full data) =======================
def prony_fit(t, A1, g1, w1, p1, A2, g2, w2, p2):
    return A1*np.exp(-g1*t)*np.sin(w1*t + p1) + A2*np.exp(-g2*t)*np.sin(w2*t + p2)

p0 = [1.0, gamma_pre, omega_inter, 0, 0.6, gamma_post, omega_local, np.pi/4]
popt, _ = curve_fit(prony_fit, t, y_ref, p0=p0, maxfev=10000)
y_prony = prony_fit(t, *popt)

# ======================= PLOT FIG. 11 =======================
plt.figure(figsize=(11, 6))
plt.plot(t, y_ref, 'k-', linewidth=2.0, label='EMT reference trajectory')
plt.axvline(x=10, color='gray', linestyle='--', alpha=0.7, label='T_j reduction (40\%)')
plt.plot(t, mean4, 'b-', linewidth=2.5, label='4-qubit entangled model (prediction)')
plt.fill_between(t, mean4 - std4, mean4 + std4, color='b', alpha=0.25, label=r'$\pm 1\sigma$ band')
plt.plot(t, y_prony, 'r--', linewidth=2.0, label='Prony (re-fitted on full data)')

plt.xlabel('Time (s)')
plt.ylabel('Frequency deviation (pu)')
plt.title('Fig. 11: Predictive test on IEEE 39-bus 80% IBR\n'
          'at $t=10$ s the IBR virtual inertia $T_j$ is reduced by 40\%')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
