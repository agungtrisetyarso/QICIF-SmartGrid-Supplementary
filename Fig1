import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ========================== FIGURE 2 ==========================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Grid for normalized labor and capital
labor = np.linspace(0.6, 1.4, 50)
capital = np.linspace(0.6, 1.4, 50)
L, K = np.meshgrid(labor, capital)

# Different surfaces for increasing reskilling intensity (R)
def stability_surface(L, K, R):
    # Base oscillatory surface with damping
    base = np.sin(8 * (L - 1)) * np.cos(8 * (K - 1)) * np.exp(-3 * R)
    # Add Leontief-like complementarity ridge
    complementarity = np.exp(-((L - 1)**2 + (K - 1)**2) / 0.15)
    return 1.0 + base * 0.6 + complementarity * 0.8

# Plot surfaces for different reskilling levels
Rs = [0.0, 0.04, 0.08]
colors = ['viridis', 'plasma', 'cividis']
labels = ['Baseline (R=0.0)', 'Moderate Reskilling (R=0.04)', 'High Reskilling (R=0.08)']

for i, R in enumerate(Rs):
    Z = stability_surface(L, K, R)
    surf = ax.plot_surface(L, K, Z, cmap=colors[i], alpha=0.75, linewidth=0, antialiased=True)
    # Add contour lines
    ax.contour(L, K, Z, zdir='z', offset=0.4, levels=15, alpha=0.4)

# Oscillatory trajectories (helical paths showing dynamics)
t = np.linspace(0, 10, 300)
for R, color in zip([0.0, 0.04, 0.08], ['red', 'blue', 'green']):
    # Oscillatory trajectory in labor-capital-stability space
    labor_traj = 1.0 + 0.25 * np.sin(6 * t) * np.exp(-0.8 * R * t)
    capital_traj = 1.0 + 0.22 * np.cos(6 * t) * np.exp(-0.75 * R * t)
    stability_traj = 1.2 + 0.6 * np.sin(8 * t) * np.exp(-1.2 * R * t)
    
    ax.plot(labor_traj, capital_traj, stability_traj, 
            color=color, linewidth=2.5, label=f'Trajectory (R={R})')
    # Add starting and ending points
    ax.scatter(labor_traj[0], capital_traj[0], stability_traj[0], 
               color=color, s=60, marker='o')
    ax.scatter(labor_traj[-1], capital_traj[-1], stability_traj[-1], 
               color=color, s=80, marker='^')

# Labels and title
ax.set_xlabel('Normalized Labor', fontsize=12, labelpad=10)
ax.set_ylabel('Normalized Capital', fontsize=12, labelpad=10)
ax.set_zlabel('Stability / Output Level', fontsize=12, labelpad=10)

ax.set_title('Figure 2: Three-dimensional Visualization of Oscillatory Trajectories\n'
             'in the Normalized Labor-Capital-Stability Space\n'
             '(Color Gradient Indicates Damping Ratio | Surfaces Converge with Reskilling)',
             fontsize=13, pad=30)

# Legend
ax.legend(loc='upper left', fontsize=10)

# View angle for better visualization
ax.view_init(elev=25, azim=135)

plt.tight_layout()
plt.savefig('Figure_2_3D_Oscillatory_Trajectories.png', dpi=300, bbox_inches='tight')
plt.show()
