# region Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import time
import tkinter as tk
import glob

from Objects.BPASSAnalysis import BPASSAnalysis
from Objects.BPASSDataFormatter import BPASSDataFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import RLock
from scipy.stats import gaussian_kde

#region setup
tqdm.set_lock(RLock())

MAX_WORKERS = 4
colours = ['cyan', 'blue', 'orange', 'magenta', 'red', 'yellow', 'brown', 'forestgreen', 'purple', 'pink', 'grey', 'black']

bpass = BPASSDataFormatter()
allSupernovaArray, allIonizingArray, combinedSupernovaIon = bpass.getAllFormattedData()
bpassAnalysis = BPASSAnalysis(allSupernovaArray)

rates_folder = f"/Users/dan/Code/FYP/Data/TNG/Rates"

def count_lines_fast(path):
    with open(path, "rb") as f:
        count = 0
        BUFSIZE = 1024 * 1024
        while True:
            buf = f.read(BUFSIZE)
            if not buf:
                break
            count += buf.count(b"\n")
    return count

# region Build
def build_rates(snap):
    position = snapshots.index(snap) + 1
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{snap}/*"
    my_glob = glob.glob(input_path)

    subhalo_rows = []
    for file_name in my_glob:
        try:
            total_lines = count_lines_fast(file_name)
            # CSV has a single header row:
            total_rows = max(0, total_lines - 1)
        except Exception:
            total_rows = None

        # create a tqdm for the CSV inner loop
        if total_rows:
            pbar = tqdm(total=total_rows,
                        desc=f"Snap {snap} - {file_name.split('/')[-1]}",
                        position=position,
                        leave=False)
        else:
            pbar = tqdm(desc=f"Snap {snap} - {file_name.split('/')[-1]}",
                        position=position,
                        leave=False)

        # call subhaloData and pass the pbar so inner loop can update it
        subhalo_data = bpassAnalysis.subhaloData(file_name, pbar=pbar)
        pbar.close()

        subhalo_rows.append(dict(subhalo_data.items()))

    print(f"    {len(subhalo_rows)} subhalos with postive rates")
    subhalo_df = pd.DataFrame(subhalo_rows)
    subhalo_df.to_csv(f"/Users/dan/Code/FYP/Data/TNG/Rates/snapshot{snap}_rates.csv")

    return subhalo_df

# region Calc Density
def calculate_densities(snaps):

    redshifts = []
    snrd_box = []
    sfrd_box = []
    
    for idx, snap in enumerate(snaps):
        # read rate files
        rates_file = os.path.join(rates_folder, f"snapshot{snap}_rates.csv")
        subhalo_df = pd.read_csv(rates_file)

        if len(subhalo_df) <= 1:
            continue

        # collect all redshifts to plot with the total snr of snapshots
        redshift = subhalo_df['z'].iloc[0]
        redshifts.append(redshift)

        # Calculate the box size properly (75000 ckpc/h)
        h = 0.6774
        box_size_length = 75000 * 1e-6 / (h * (1+redshift))
        box_size = pow(box_size_length,3)

        # total snrd in the box 
        # found by summing all the snr (each subhalo)
        total_snr = sum(subhalo_df["snr"])
        # divide by the total box volume 
        total_snrd = total_snr / box_size
        snrd_box.append(total_snrd)

        # total star formation in the box 
        total_sfr = sum(subhalo_df["sfr"])
        total_sfrd = total_sfr / box_size
        sfrd_box.append(total_sfrd)

        print(f"{snap}: z: {redshift}, snr: {total_snr}, snrd: {total_snrd}")

    return redshifts, snrd_box, sfrd_box

# region averages
def weighted_average(snaps):
    
    avg_snrd_density = []
    avg_sfrd_density = []

    for idx, snap in enumerate(snaps):
        # read rate files
        rates_file = os.path.join(rates_folder, f"snapshot{snap}_rates.csv")
        subhalo_df = pd.read_csv(rates_file)

        if len(subhalo_df) <= 1:
            continue

        snrd_group = subhalo_df['snrd']
        sfrd_group = subhalo_df['sfrd']
        
        # compute densities for SNRd
        kde_snrd = gaussian_kde(snrd_group)
        weights_snrd = kde_snrd(snrd_group)
        
        kde_sfrd = gaussian_kde(sfrd_group)
        weights_sfrd = kde_sfrd(sfrd_group)
        
        avg_snrd_density.append(np.average(snrd_group, weights=weights_snrd))
        avg_sfrd_density.append(np.average(sfrd_group, weights=weights_sfrd))

    fig3, ax5 = plt.subplots()
    ax5.scatter(avg_sfrd_density, avg_snrd_density)
    ax5.set_title('Weighted Avergaes SFRd vs SNRd')
    ax5.set_xlabel('SFR Mo yr^-1 Gpc^-3')
    ax5.set_ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    #ax5.set_yscale('log')
    #ax5.set_xscale('log')

    plt.show()

    return fig3

# region Plot
def plot_rates(snaps):

    redshifts, snrd_box, sfrd_box = calculate_densities(snaps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('SFR Vs SNR [per year and solar mass]')
    ax1.set_xlabel('SFR Mo yr^-1')
    ax1.set_ylabel('SNR Supernova yr^-1 Mo^-1')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.set_title('SFR Vs SNR [per year, solar mass and volume]')
    ax2.set_xlabel('SFR Mo yr^-1 Gpc^-3')
    ax2.set_ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    fig2, ax3 = plt.subplots()
    sc1 = ax3.plot(redshifts, snrd_box, color='blue', label='SNRD', marker='x')
    ax3.set_title('Supernova Rate and Star Formation Rate across time')
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.set_yscale('log')

    # Create a second y-axis sharing the same x-axis
    ax4 = ax3.twinx()
    sc2 = ax4.plot(redshifts, sfrd_box, color='red', label='SFRD', marker='x')
    ax3.set_ylabel('SFR (Star Formation) Mo yr^-1 Gpc^-3')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.set_yscale('log')


    for idx, snap in enumerate(snaps):
        # read rate files
        rates_file = os.path.join(rates_folder, f"snapshot{snap}_rates.csv")
        subhalo_df = pd.read_csv(rates_file)

        if len(subhalo_df) <= 1:
            continue
        redshift = subhalo_df['z'].iloc[0]

        # plot snr vs sfr and snrd vs sfrd (halo level)
        ax1.scatter(subhalo_df['sfr'], subhalo_df["snr"], marker='.', color=colours[idx], label=f'z={redshift}')
        ax2.scatter(subhalo_df['sfrd'], subhalo_df["snrd"], marker='.', color=colours[idx], label=f'z={redshift}')

    ax1.legend()
    ax2.legend()
    fig2.legend()

    dfig = weighted_average(snaps)

    plt.show()

    return fig, fig2

snapshots = [2, 20, 40, 50, 57, 66, 80, 98]

build = True

if build == True:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(build_rates, snap) for snap in snapshots]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

dumb1, dumb2 = plot_rates(snapshots)