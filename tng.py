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
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.cosmology import z_at_value
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter

#region setup
tqdm.set_lock(RLock())

MAX_WORKERS = 4
colours = ['cyan', 'blue', 'orange', 'magenta', 'red', 'yellow', 'brown', 'limegreen', 'purple', 'pink', 'grey', 'black']

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

        #print(f"{snap}: z: {redshift}, snr: {total_snr}, snrd: {total_snrd}")

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

    #plt.show()

    return fig3

def sfrd_func(z, a, b, c, d):
    # takes madua and dickinson form but in Gpc^3
    sfrd = a * pow((1 + z), b)/(1 + pow((1 + z)/c, d)) * 1e9
    return sfrd

# region Plot
def plot_rates(snaps):

    redshifts, snrd_box, sfrd_box = calculate_densities(snaps)
    
    # sfr vs snr plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('SFR Vs SNR [per year and solar mass]')
    ax1.set_xlabel('SFR (Star Formation) Mo yr^-1')
    ax1.set_ylabel('SNR (Supernova) yr^-1 Mo^-1')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.set_title('SFR Vs SNR [per year, solar mass and volume]')
    ax2.set_xlabel('SFRD (Star Formation) Mo yr^-1 Gpc^-3')
    ax2.set_ylabel('SNRD (Supernova)  yr^-1 Mo^-1 Gpc^-3')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # vs redshift plots
    fig2, ax3 = plt.subplots(figsize=(8,5))
    sc1 = ax3.scatter(redshifts, snrd_box, color='Red', label='SNRD', marker='x')
    #ax3.set_title('Supernova Rate and Star Formation Rate across time')
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel(r'SNRd (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}\ Gpc^{-3}}$]')
    ax3.tick_params(axis='y', labelcolor='Red')
    ax3.set_yscale('log')

    # Create a second y-axis sharing the same x-axis
    ax4 = ax3.twinx()
    sc2 = ax4.scatter(redshifts, sfrd_box, color='Green', label='SFRD', marker='x')
    ax4.set_ylabel(r'SFRd (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Gpc^{-3}}$]')
    ax4.tick_params(axis='y', labelcolor='Green')
    ax4.set_yscale('log')

    # find equation 
    # order arrays to be ascedning 
    redshift_array = np.array(redshifts)[::-1]
    sfrd_array = np.array(sfrd_box)[::-1]
    snrd_array = np.array(snrd_box)[::-1]

    degree = 3
    snrd_coeffs = np.polyfit(redshift_array, snrd_array, degree)
    sfrd_coeffs = np.polyfit(redshift_array, sfrd_array, degree)
    print(sfrd_coeffs)
    
    # use poly func build in bpassAnalysis 
    redshift_linespace = np.linspace(redshift_array.min(), redshift_array.max(), 300)
    #snrd_poly = bpassAnalysis.polynomialCalc(snrd_coeffs, redshift_poly)
    #sfrd_poly = bpassAnalysis.polynomialCalc(sfrd_coeffs, redshift_linespace)

    #ax3.plot(redshift_linespace, snrd_poly, linestyle='--', marker=None, color='Red')
    #ax4.plot(redshift_linespace, sfrd_poly, linestyle='--', marker=None, color='Green')

    # madua and dickinson
    # their fomrula ius given in Mpc3 we have Gpc3
    csfrh = 0.015 * pow((1 + redshift_linespace), 2.7)/(1 + pow((1 + redshift_linespace)/2.9, 5.6)) * 1e9
    line1, = ax4.plot(redshift_linespace, csfrh, linestyle='--', color='orange', label="SFRd (Madau & Dickinson 2014)")

    # Initial guess use the value isn madau and dickinson 
    p0 = [0.015, 2.7, 2.9, 5.6]
    
    # scipy curve fit
    snrd_params, cov = curve_fit(sfrd_func, redshift_array, snrd_array, p0=p0)
    snrd_fit = sfrd_func(redshift_linespace, *snrd_params)
    line2, = ax3.plot(redshift_linespace, snrd_fit, linestyle='--', color='magenta', label="Estimate SNRD")
    print(snrd_params)

    sfrd_params, cov = curve_fit(sfrd_func, redshift_array, sfrd_array, p0=p0)
    sfrd_fit = sfrd_func(redshift_linespace, *sfrd_params)
    line3, = ax4.plot(redshift_linespace, sfrd_fit, linestyle='--', color='lime', label="Estimate SFRD")
    print(sfrd_params)

    lookback_time_grid = cosmo.lookback_time(redshift_array).value  # in Gyr
    redshift_to_age = interp1d(redshift_array, lookback_time_grid, bounds_error=False, fill_value="extrapolate")
    age_to_redshift = interp1d(lookback_time_grid, redshift_array, bounds_error=False, fill_value="extrapolate")

    ax5 = ax3.secondary_xaxis('top', functions=(redshift_to_age, age_to_redshift))
    ax5.set_xlabel("Cosmic Lookback [Gyr]")
    ax5.set_xscale('log')
    ax5.xaxis.set_major_formatter(ScalarFormatter()) 
    ax5.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])

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
    ax3.legend([sc1, sc2, line1, line2, line3], ['SNRd (TNG)', 'SFRd (TNG)', 'Madau & Dickinson 2014', 'SNRd (Function)', 'SFRd (Function)'],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False)
    plt.subplots_adjust(bottom=0.25)

    dfig = weighted_average(snaps)

    #plt.close(fig)
    #plt.close(dfig)
    plt.show()  # shows only fig2

    return fig, fig2

snapshots = [2, 10, 20, 32, 40, 50, 57, 66, 80, 98]

build = False

if build == True:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(build_rates, snap) for snap in snapshots]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

dumb1, dumb2 = plot_rates(snapshots)