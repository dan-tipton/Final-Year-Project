import numpy as np
import matplotlib.pyplot as plt

# Example TNG subhalo SFRs (M_sun/yr) - just illustrative values for demonstration
sfr = np.array([0.01, 0.1, 1, 5, 10, 50, 100])
# CCSN rate (yr^-1) = k * SFR, k ~ 0.007 SN/M_sun formed
k = 0.007
ccsn_rate = k * sfr

# Introduce some scatter to simulate variation between subhalos
np.random.seed(42)
ccsn_rate_scatter = ccsn_rate * (1 + 0.2 * (np.random.rand(len(sfr)) - 0.5))  # ±10% scatter

# Plot
plt.figure(figsize=(7,5))
plt.scatter(sfr, ccsn_rate_scatter, color='blue', label='TNG subhalos (mock)')
plt.plot(sfr, ccsn_rate, color='red', linestyle='--', label='Linear expectation (R=k*SFR)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Star Formation Rate [M$_\odot$/yr]')
plt.ylabel('Core-Collapse SN Rate [yr$^{-1}$]')
plt.title('Core-Collapse Supernova Rate vs Star Formation Rate')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.show()