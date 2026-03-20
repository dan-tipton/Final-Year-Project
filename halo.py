"""
predict the number of ccSNe they expect from each SF region based on the BPASS 
stellar population model predictions. With this info then can then compute the total 
number or ratio of various ccSNe type per galaxy and then investigate how this 
(in particular ratio I think) various with galaxy stellar mass for example. 
They may want to re-run their analysis at a different simulation redshift to 
investigate age and how rates vary as function of redshift and stellar mass. 

Rates Files (Data/TNG/Rates/supernova_type)
    - contain all valid halos in snapshot 
        - valid criteria:
            - SFR > 0 
            - Number of Halos > 0 
            - Star forming region - nH2 > 0.1
    - Data included
        - halo Id
        - sfr (groupcat in rad?)
        - sfrd Mo Yr-1 Mpc-3 (per halo volume)
        - snr Yr-1 (from BPASS)
        - snr_solar Yr-1 Mo-1 (from BPASS - may be wrong?)
            - maybe need to use initial mass instead?
        - snrd Yr-1 Mpc-3 (per halo volume)
        - mass Mo (of halo)
        - z (redshift contant for a file)
        - halo volume Mpc-3

Have rate file for all snapshot and all 4 sn types 
Need a total sn for a given snapshot to calculated ratio
Need to find the same halo in all 4 different files for different SN type
"""

import os 
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib.animation import FuncAnimation, PillowWriter

base = os.getcwd()

rate_base = os.path.join(base, 'Data/TNG/Rates')
sn_type = ["IIP", "II-Other", "Ib", "Ic"]
snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]
cols_suffix = ["sfr_", "sfrd_", "snr_", "snrs_", "snrd_"]
cols_suffix_2 = ["sfr_", "snr_", "snrs_"]
colors = ['#FF5733', '#33FF57', '#3357FF', "#FFD012"]

from functools import reduce
import pandas as pd

# merge function
def merge_keep_one(left, right):
    # keep only columns from right that are not already in left (except 'id')
    cols = ['id'] + [c for c in right.columns if c not in left.columns]
    
    return pd.merge(left, right[cols], on='id', how='inner')


# join data frames based on subhalo 
# produce one csv for each redshisft
def ratio_calc(save=None):
    all = {}
    for s in snapshots:
        rates = {}
        for sn in sn_type:
            rate_path = os.path.join(rate_base, sn, f"snapshot{s}_rates.csv")
            df = pd.read_csv(rate_path)
            df = df.rename(columns={
                'sfr': f'sfr_{sn}',
                'sfrd': f'sfrd_{sn}', 
                'snr': f'snr_{sn}', 
                'snr_solar': f'snrs_{sn}', 
                'snrd': f'snrd_{sn}'
            })
            rates[sn] = df

        merged_df = reduce(merge_keep_one, rates.values())

        for suffix in cols_suffix:
            cols = [col for col in merged_df.columns if col.startswith(suffix)]
            merged_df[f'{suffix}total'] = merged_df[cols].sum(axis=1)

        for sn in sn_type:
            for suffix in cols_suffix_2:
                curr = f"{suffix}{sn}"
                ratio = merged_df[curr] / merged_df[f"{suffix}total"]

                merged_df[f"{suffix}{sn}_ratio"] = ratio

        all[s] = merged_df
        if save != None:
            combined_base = os.path.join(base, 'Data/TNG/Combined')
            merged_df.to_csv(combined_base + f"/s{s}.csv")
    
    return all

def plotter(snaps, png_name='ratio.png', pcols=1):
    dfs = ratio_calc()
    if type(snaps) != int:
        num = len(snaps)
        prows = math.ceil(num / pcols)  
        fig, ax = plt.subplots(prows, pcols, figsize=(15, 5))
        ax = ax.flatten() 
        for i, s in enumerate(snaps):
            df = dfs[s]

            for idx, sn in enumerate(sn_type):
                curr = f"snr_{sn}_ratio"
                ratio = df[curr]
                mass = df["mass"]
                z = df["z"].iloc[0]

                ax[i].scatter(mass, ratio, label=f'{sn}', color=colors[idx], marker='.')
                ax[i].set_ylabel("ratio")
                ax[i].set_xlabel("mass")
                ax[i].set_xscale('log')
                ax[i].set_title(f"Redshift={z:.2f}")
        
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(sn_type))

    else:
        num = 1
        fig, ax = plt.subplots(figsize=(8, 7))
        df = dfs[snaps]
        for idx, sn in enumerate(sn_type):
            curr = f"snr_{sn}_ratio"
            ratio = df[curr]
            mass = df["mass"]
            z = df["z"].iloc[0]

            ax.scatter(mass, ratio, label=f'{sn}', color=colors[idx], marker='.')
            ax.set_ylabel("ratio")
            ax.set_xlabel("mass")
            ax.set_xscale('log')
            ax.set_title(f"Redshift={z:.2f}")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(sn_type))
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    fig.savefig(f"Data/Images/TNG/ratio/{png_name}", dpi=300)
    return fig

def animate_seperate(snaps, save_gif=False, gif_name="animation_seperate.gif"):
    dfs = ratio_calc()  # your function
    num = len(snaps)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax = ax.flatten()

    # Precompute scatter data for all snapshots
    scatter_data = []
    for s in snaps:
        df = dfs[s]
        snapshot_scatter = []
        for idx, sn in enumerate(sn_type):
            curr = f"snr_{sn}_ratio"
            ratio = df[curr]
            mass = df["mass"]
            z = df["z"].iloc[0]
            snapshot_scatter.append((mass, ratio, sn, colors[idx], z))
        scatter_data.append(snapshot_scatter)

    # Initialize plots
    for a in ax:
        a.clear()

    def update(frame):
        # Clear all axes
        for a in ax:
            a.clear()

        sdata = scatter_data[frame]
        for i, (mass, ratio, sn, color, z) in enumerate(sdata):
            ax[i].scatter(mass, ratio, label=sn, color=color, marker='.')
            ax[i].set_ylabel("ratio")
            ax[i].set_xlabel("mass")
            ax[i].set_xscale('log')
            ax[i].set_title(f"Redshift={z:.2f}")
        fig.suptitle(f"Snapshot {frame + 1}/{num}")

    ani = FuncAnimation(fig, update, frames=num, interval=1000, repeat=True)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sn_type))
    fig.tight_layout()  # leave space for legend
    ani.save(gif_name, writer=PillowWriter(fps=1))  # fps=1 → 1 frame per second
    return ani


def animate_plotter(snaps, save_gif=False, gif_name="animation.gif"):
    dfs = ratio_calc()  # your function returning a dict of DataFrames

    fig, ax = plt.subplots(figsize=(8, 6))

    # Precompute data for all snapshots
    scatter_data = []
    for s in snaps:
        df = dfs[s]
        snapshot = []
        z = df["z"].iloc[0]
        for idx, sn in enumerate(sn_type):
            curr = f"snr_{sn}_ratio"
            mass = df["mass"]
            ratio = df[curr]
            snapshot.append((mass, ratio, sn, colors[idx]))
        scatter_data.append((snapshot, z))

    def update(frame):
        ax.clear()
        snapshot, z = scatter_data[frame]
        for mass, ratio, sn, color in snapshot:
            ax.scatter(mass, ratio, label=sn, color=color, marker='.')
        ax.set_xscale('log')
        ax.set_xlabel("mass")
        ax.set_ylabel("ratio")
        ax.set_title(f"Redshift={z:.2f} (Snapshot {frame+1}/{len(snaps)})")
        ax.legend(loc='upper right')

    ani = FuncAnimation(fig, update, frames=len(snaps), interval=1000, repeat=True)

    if save_gif:
        ani.save(gif_name, writer=PillowWriter(fps=1))

    return ani

for s in snapshots:
    sfig = plotter(s, png_name=f"ratio_s{s}.png")


"""
snaps1= [2, 10, 20]
snaps2 = [26, 32, 40]
snaps3 = [50, 57, 66]
snaps4 = [80, 98]

fig1 = plotter(snaps1, png_name='s2_10_20.png')
fig2 = plotter(snaps2, png_name='s26_32_40.png')
fig3 = plotter(snaps3, png_name='s50_57_66.png')
fig4 = plotter(snaps4, png_name='s2_10_20.png')
"""

#new = np.array(snapshots)[::-1]
#animate_plotter(new, save_gif=True)
#animate_seperate(new, save_gif=True)

#plt.show()

