import illustris_python as il
import os
import matplotlib.pyplot as plt

# Path pattern for all group catalog files of snapshot 98
base = os.getcwd()
data_path = os.path.join(base, f"Data/GroupCat")

fields = ['SubhaloMass','SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath,98,fields=fields)

mass_msun = subhalos['SubhaloMass'] * 1e10 / 0.704
plt.plot(mass_msun,subhalos['SubhaloSFRinRad'],'.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Mass [$M_\odot$]')
plt.ylabel('Star Formation Rate [$M_\odot / yr$]')