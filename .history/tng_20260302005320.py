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

tqdm.set_lock(RLock())

MAX_WORKERS = 4
colours = ['cyan', 'blue', 'orange', 'magenta', 'red', 'yellow', 'brown']

bpass = BPASSDataFormatter()
allSupernovaArray, allIonizingArray, combinedSupernovaIon = bpass.getAllFormattedData()
bpassAnalysis = BPASSAnalysis(allSupernovaArray)


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

def build_rates(snap):
    position = snapshots.index(snap) + 1
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Dev/Dev_Snapshot_{snap}.csv"
    my_glob = glob.glob(input_path)

    subhalo_rows = []
    for file_name in my_glob:
        try:
            total_lines = count_lines_fast(file_name)
            # if CSV has a single header row:
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
        # NOTE: this requires modifying subhaloData to accept an optional 'pbar' argument
        subhalo_data = bpassAnalysis.subhaloData(file_name, pbar=pbar)
        pbar.close()

        subhalo_rows.append(dict(subhalo_data.items()))

    print(f"    {len(subhalo_rows)} subhalos with postive rates")
    subhalo_df = pd.DataFrame(subhalo_rows)
    subhalo_df.to_csv(f"/Users/dan/Code/FYP/Data/TNG/Rates/Dev/snapshot{snap}_rates.csv")

    return subhalo_df


def plot_rates(snaps):

    redshifts = []
    all_snrd = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('SFR Vs SNR [per year and solar mass]')
    ax1.set_xlabel('SFR Mo yr^-1')
    ax1.set_ylabel('SNR Supernova yr^-1 Mo^-1')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.set_title('SFR Vs SNR [per year, solar mass and volume]')
    ax2.set_xlabel('SFR Mo yr^-1')
    ax2.set_ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    for idx, snap in enumerate(snaps):
        # read rate files
        snr_rates_path = f"/Users/dan/Code/FYP/Data/TNG/Rates/Dev/snapshot{snap}_rates.csv"
        subhalo_df = pd.read_csv(snr_rates_path)

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
        all_snrd.append(total_snrd)

        print(f"{snap}: z: {redshift}, snr: {total_snr}, snrd: {total_snrd}")

        ax1.scatter(subhalo_df['sfr'], subhalo_df["snr"], marker='.', color=colours[idx], label=f'z={redshift}')
        ax2.scatter(subhalo_df['sfr'], subhalo_df["snrd"], marker='.', color=colours[idx], label=f'z={redshift}')

    ax1.legend()
    ax2.legend()

    plt.show()

    return fig

snapshots = [2, 20, 40, 50, 57, 66, 98]

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(build_rates, snap) for snap in snapshots]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())

plot_rates(snapshots)