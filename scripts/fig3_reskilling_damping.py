import numpy as np
import matplotlib.pyplot as plt

years = np.arange(2015, 2046)
t = years - 2015
omega = 0.1158

def labor_share(t, R):
    """Damped harmonic oscillator (RLC analogy)"""
    amplitude = 0.038
    damping = 0.008 + R * 0.12   # higher R = stronger damping
    return 0.72 + amplitude * np.exp(-damping * t) * np.sin(omega * t + 0.3)

plt.figure(figsize=(10, 6))
plt.plot(years, labor_share(t, 0.00), 'k-', linewidth=2.5, label='Baseline (no additional reskilling)')
plt.plot(years, labor_share(t, 0.02), 'b-', linewidth=2.5, label='Moderate reskilling')
plt.plot(years, labor_share(t, 0.05), 'g-', linewidth=2.5, label='High reskilling')

plt.xlabel('Year')
plt.ylabel('Normalized Labor Share')
plt.title('Effect of Reskilling (Resistive Damping) on Labor-Share Oscillations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/fig3_reskilling_damping.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 3 saved.")
