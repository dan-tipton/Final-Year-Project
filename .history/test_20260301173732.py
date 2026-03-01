import matplotlib.pyplot as plt

# Data
snapshots = [2, 20, 40, 66, 98]
redshift = [11.980213315300292, 4.176834914726472, 1.4955121664955555, 0.524565820433923, 0.0095216669679447]
snr = [9.486045606326206e-09, 9.719088121678308e-06, 4.308394581349844e-06, 3.793799298528693e-07, 4.3146475576677437e-07]
snrd = [0.01528559976933945, 0.9935054509474583, 0.049334000331017626, 0.0009905232538958886, 0.00032707251187467153]

# Plotting
fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('SNR [yr$^{-1}$ M$_\odot^{-1}$]', color=color)
ax1.plot(redshift, snr, marker='o', color=color, label='SNR per mass')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.invert_xaxis()  # Redshift decreasing to the right

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('SNRD [yr$^{-1}$ M$_\odot^{-1}$ Gpc$^{-3}$]', color=color)
ax2.plot(redshift, snrd, marker='s', color=color, label='SNRD per volume')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')

fig.tight_layout()
plt.title('Core-Collapse Supernova Rate vs Redshift')
plt.show()