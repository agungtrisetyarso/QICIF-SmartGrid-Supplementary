import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
L = np.linspace(0.6, 1.4, 50)   # Normalized Labor
K = np.linspace(0.6, 1.4, 50)   # Normalized Capital
L_grid, K_grid = np.meshgrid(L, K)

# Oscillatory isoquant surface (damped waves from VRE + LC analogy)
omega = 0.1158
damping = 0.02
phase = 0.8
Z = 1.0 + 0.8 * np.sin(omega * (L_grid + K_grid) + phase) * np.exp(-damping * (L_grid + K_grid))

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(L_grid, K_grid, Z, cmap='viridis', alpha=0.85, linewidth=0, antialiased=True)

ax.set_xlabel('Normalized Labor (L)')
ax.set_ylabel('Normalized Capital (K)')
ax.set_zlabel('Output Level (Y)')
ax.set_title('Oscillatory Isoquants in Labor-Capital Plane\nDamped Waves from VRE Intermittency\n(Converges to Leontief at High Automation)')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Output Level (Y)')
plt.tight_layout()
plt.savefig('../figures/fig2_oscillatory_isoquants_3d.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 2 saved.")
