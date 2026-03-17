import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

# -----------------------------
# 1. Load PLAsTiCC metadata
# -----------------------------
plasticc_file = "plasticc_train_metadata.csv"
df = pd.read_csv(plasticc_file)

# PLAsTiCC true_class IDs for Type II SNe (check your version)
# 42: SNII 62: SNIbc 
type_ii_classes = [42]

# Filter only Type II SNe
df_type2 = df[df['true_target'].isin(type_ii_classes)].copy()
df_type2['redshift'] = df_type2['true_z']

# -----------------------------
# 2. Bin by redshift
# -----------------------------
z_bins = np.arange(0, 3.1, 0.2)  # adjust as needed
df_type2['z_bin'] = pd.cut(df_type2['redshift'], bins=z_bins)

# -----------------------------
# 3. Compute volumetric rate density
# -----------------------------
volumes = []
for i in range(len(z_bins)-1):
    z_low = z_bins[i]
    z_high = z_bins[i+1]
    v = cosmo.comoving_volume(z_high) - cosmo.comoving_volume(z_low)
    volumes.append(v.value)  # Mpc^3

volumes = np.array(volumes)

# Count Type II SNe in each bin
counts = df_type2.groupby('z_bin').size()
counts_arr = np.array([counts.get(interval, 0) for interval in df_type2['z_bin'].cat.categories])

# Assume 1-year survey duration (adjust if needed)
rate_density_type2 = counts_arr / volumes  # events / Mpc^3 / yr

# -----------------------------
# 4. Plot Type II SN rate density
# -----------------------------
bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

plt.figure(figsize=(8,6))
plt.plot(bin_centers, rate_density_type2, marker='o', color='red', label='Type II SNe')
plt.yscale('log')
plt.xlabel('Redshift')
plt.ylabel('SN Type II Rate Density [events / Mpc^3 / yr]')
plt.title('PLAsTiCC Type II Supernova Rate Density vs Redshift')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()