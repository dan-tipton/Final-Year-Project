# region Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import time
import tkinter as tk
import glob
import numpy.ma as ma

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
from matplotlib.colors import LogNorm

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

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
    snrd_no_mass_box = []
    
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
        # gives in comving currenly
        h = 0.6774
        #box_size_length = 75000 * 1e-6 / (h * (1+redshift))
        box_size_length = 75000 * 1e-3 / h
        box_size = pow(box_size_length,3)

        # total snrd in the box 
        # found by summing all the snr (each subhalo)
        total_snr = sum(subhalo_df["snr"])
        # divide by the total box volume 
        total_snrd = total_snr / box_size
        snrd_box.append(total_snrd)

        # NEW SNRD Calculation (different units)
        new_rate = subhalo_df["snr"] * subhalo_df["mass"]
        mean_rate = np.mean(new_rate)
        median_rate = np.median(new_rate)
        max_rate = np.max(new_rate)
        #print(f"Mean: {mean_rate:2e}, Median: {median_rate:2e}, Max: {max_rate:2e}")
        total = np.sum(new_rate)
        top10 = np.sum(np.sort(new_rate)[-10:])
        rfract_max = max(new_rate) / total
        rfract_ten = top10 / total

        mfract_max = max(subhalo_df["mass"])/sum(subhalo_df["mass"])
        mfract_ten = np.sum(np.sort(subhalo_df["mass"])[-10:])/sum(subhalo_df["mass"])

        if mfract_ten > 0.5:
            if mfract_max > 0.5:
                print(f"Snapshot {snap}: Mass Dominated by massive: {mfract_max}")
            else:
                print(f"Snapshot {snap}: Mass Dominated by top 10: {mfract_ten}")

        if rfract_ten > 0.5:
            if rfract_max > 0.5:
                print(f"Snapshot {snap}: Rate Dominated by massive: {rfract_max}")
            else:
               print(f"Snapshot {snap}: Rate Dominated by top 10: {rfract_ten}")
        #print(" Fraction Max", max(new_rate) / total)
        #print(" Fraction Top 10", top10 / total)
        #print(" Length", len(new_rate))



        total_snr_times_mass = sum(subhalo_df["snr"] * subhalo_df["mass"])
        total_snrd_no_mass = total_snr_times_mass / box_size
        snrd_no_mass_box.append(total_snrd_no_mass)

        # total star formation in the box 
        total_sfr = sum(subhalo_df["sfr"])
        total_sfrd = total_sfr / box_size
        sfrd_box.append(total_sfrd)

        #print(f"{snap}: z: {redshift}, snr: {total_snr}, snrd: {total_snrd}")

    return redshifts, snrd_box, sfrd_box, snrd_no_mass_box

#region sfrd 
def calculated_sfrd():
    sfr_df = pd.read_csv(f"/Users/dan/Code/FYP/Data/TNG/total_sfr_per_redshift.csv")

    # Calculate the box size properly (75000 ckpc/h)
    h = 0.6774
    box_size_length = 75000 * 1e-3 / h
    box_size = pow(box_size_length,3)

    print("box size", box_size)

    sfrd_full = []

    for idx, row in sfr_df.iterrows():
        #if idx == 0:
        #    continue
        sfrd = row["sfr"]/box_size
        sfrd_full.append(sfrd)

    return sfrd_full
        

# region averages
def weighted_average(snaps):
    avg_snrd_density = []
    avg_sfrd_density = []

    av_snrd = []
    av_sfrd = []

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

        av_snrd.append(sum(snrd_group)/len(snrd_group))
        av_sfrd.append(sum(sfrd_group)/len(sfrd_group))

    fig3, ax6 = plt.subplots()
    #ax6.scatter(avg_sfrd_density, avg_snrd_density, marker='x', color='red')
    ax6.set_title('Averaged SFRD vs SNRD')
    ax6.set_xlabel(r'SFRD (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Gpc^{-3}}$]')
    ax6.set_ylabel(r'SNRD (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}\ Gpc^{-3}}$]')
    #ax5.set_yscale('log')
    #ax5.set_xscale('log')

    """
    # using weighted averages
    sfrd_linespace = np.linspace(np.array(avg_sfrd_density).min(), np.array(avg_sfrd_density).max(), 200)
    coeffs = np.polyfit(avg_sfrd_density, avg_snrd_density, 1) 
    m, c = coeffs
    snrd_line = m*sfrd_linespace + c
    ax6.plot(sfrd_linespace, snrd_line, linestyle='--', color='red')
    print(f"Gradient: {m},Intercept: {c}")
    """

    # standard deviation (sample std, N-1)
    std_dev = np.std(av_snrd, ddof=1)  # ddof=1 gives sample std
    # standard error of the mean
    std_error = std_dev / np.sqrt(len(av_snrd))

    ax6.errorbar(av_sfrd, av_snrd, yerr=std_error, color='blue', fmt='x', capsize=5, label="Average")
    sfrd_linespace = np.linspace(np.array(av_sfrd).min(), np.array(av_sfrd).max(), 200)
    coeffs = np.polyfit(av_sfrd, av_snrd, 1) 
    m, c = coeffs
    snrd_line = m*sfrd_linespace + c
    ax6.plot(sfrd_linespace, snrd_line, linestyle='--', color='blue', label="Line Of Best Fit")
    print(f"Gradient: {m},Intercept: {c}")
    ax6.legend()

    #plt.show()
    return fig3

def sfrd_func(z, a, b, c, d):
    # takes madua and dickinson form but in Gpc^3
    sfrd = a * pow((1 + z), b)/(1 + pow((1 + z)/c, d)) * 1e9
    return sfrd

# region Plot
def plot_rates(snaps):

    redshifts, snrd_box, sfrd_box, snrd_no_mass_box = calculate_densities(snaps)
    total_sfrds = calculated_sfrd()
    
    # sfr vs snr plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    fig.subplots_adjust(bottom=0.18, hspace=0.4, top=0.95)
    ax1.set_title('SFR vs SNR')
    ax1.set_xlabel(r'SFR (Star Formation) [$\mathrm{M_\odot\ yr^{-1}}$]')
    ax1.set_ylabel(r'SNRD (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}}$]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.set_title('SFRD vs SNRD ')
    ax2.set_xlabel(r'SFRd (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Gpc^{-3}}$]')
    ax2.set_ylabel(r'SNRD (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}\ Gpc^{-3}}$]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # vs redshift plots
    fig2, ax3 = plt.subplots(figsize=(8,7))
    fig3, ax3_1 = plt.subplots(figsize=(8,7))
    sc1 = ax3.scatter(redshifts, snrd_box, color='Red', label='SNRD', marker='x')
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel(r'SNRD (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}\ Mpc^{-3}}$]')
    ax3.tick_params(axis='y', labelcolor='Red')
    ax3.set_yscale('log')
    sc1.set_visible(False)

    # Create a second y-axis sharing the same x-axis
    ax4 = ax3.twinx()
    sc2 = ax4.scatter(redshifts, sfrd_box, color='Green', label='SFRD (Top 1000)', marker='.')
    sc21 = ax4.scatter(redshifts, total_sfrds, color='forestgreen', label='SFRD (All)', marker='.')
    ax4.set_ylabel(r'SFRd (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]')
    ax4.tick_params(axis='y', labelcolor='Green')
    ax4.set_yscale('log')
    sc2.set_visible(False)
    sc21.set_visible(False)

    # create a third y axis and move left (different supernova rate units)
    ax7 = ax3.twinx()  # creates a second y-axis (normally right)
    ax7.spines["left"].set_position(("axes", -0.15))  # shift to left
    ax7.spines["right"].set_visible(False)     
    ax7.yaxis.set_label_position('left')
    ax7.yaxis.tick_left()  
    ax7.set_ylabel(r'SNRD (Supernova) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]')
    ax7.yaxis.set_label_coords(-0.25, 0.5)
    ax7.set_yscale('log')
    sc11 = ax7.scatter(redshifts, snrd_no_mass_box, color='dodgerblue', label='SNRD (New Units)', marker='x')
    ax7.tick_params(axis='y', colors='dodgerblue')
    sc11.set_visible(False)

    # find equation 
    # order arrays to be ascedning 
    redshift_array = np.array(redshifts)[::-1]
    sfrd_array = np.array(sfrd_box)[::-1] 
    sfrd_total_array = np.array(total_sfrds)[::-1] 
    snrd_array = np.array(snrd_box)[::-1]
    snrd_no_mass_array = np.array(snrd_no_mass_box)[::-1]

    # scaling
    # calculate a scale factor for each redshift value to convert from top 1000 halos to all halos
    # SFRD values are compared between the two data sets 
    # this can be applied to the SNRD to get SNRD estimates for the whole box 
    scaling = sfrd_total_array/sfrd_array
    snrd_scaled_array = snrd_array * scaling 

    # madua and dickinson
    # their fomrula ius given in Mpc3 we have Gpc3
    # convert from comoving to physical by multiplying by (1+z)^3
    redshift_linespace = np.linspace(redshift_array.min(), redshift_array.max(), 300)
    csfrh = 0.015 * pow((1 + redshift_linespace), 2.7)/(1 + pow((1 + redshift_linespace)/2.9, 5.6)) #* 1e9 #* pow((1 + redshift_linespace),3)
    # quoted core collapse efficiency scaling for salpeter
    # we will want a 
    csnrh = csfrh * 0.0068
    line1, = ax4.plot(redshift_linespace, csfrh, linestyle='--', color='lime', label="SFRD (Madau & Dickinson 2014)")
    line11, = ax7.plot(redshift_linespace, csnrh, linestyle='--', color='aqua', label="SNRD (Madau & Dickinson 2014)")

    # Initial guess use the value isn madau and dickinson 
    p0 = [0.015, 2.7, 2.9, 5.6]
    
    # snrd
    snrd_params, cov = curve_fit(sfrd_func, redshift_array, snrd_array, p0=p0)
    snrd_fit = sfrd_func(redshift_linespace, *snrd_params)
    line2, = ax3.plot(redshift_linespace, snrd_fit, linestyle='--', color='crimson', label="Estimate SNRD")
    print("snrd params: ", snrd_params)
    line2.set_visible(False)

    # snrd scaled 
    snrd_params_1, cov = curve_fit(sfrd_func, redshift_array, snrd_scaled_array, p0=p0)
    snrd_fit_1 = sfrd_func(redshift_linespace, *snrd_params_1)
    line8, = ax3.plot(redshift_linespace, snrd_fit_1, linestyle='--', color='red', label="Estimate SNRD")
    print("snrd scaled params: ", snrd_params_1)

    # snrd - new units yr-1 Gpc-3
    snrd_new_params, cov = curve_fit(sfrd_func, redshift_array, snrd_no_mass_array, p0=p0)
    snrd_new_fit = sfrd_func(redshift_linespace, *snrd_new_params)
    line9, = ax7.plot(redshift_linespace, snrd_new_fit, linestyle='--', color='dodgerblue', label="Estimate SNRD")
    print("snrd new params: ", snrd_new_params)

    # sfrd - top 1000
    sfrd_params, cov = curve_fit(sfrd_func, redshift_array, sfrd_array, p0=p0)
    sfrd_fit = sfrd_func(redshift_linespace, *sfrd_params)
    line3, = ax4.plot(redshift_linespace, sfrd_fit, linestyle='--', color='lime', label="Estimate SFRD (Top 1000)")
    print("sfrd params", sfrd_params)
    line3.set_visible(False)

    # sfrd - all
    sfrd_all_params, cov = curve_fit(sfrd_func, redshift_array, sfrd_total_array, p0=p0)
    sfrd_all_fit = sfrd_func(redshift_linespace, *sfrd_all_params)
    line4, = ax4.plot(redshift_linespace, sfrd_all_fit, linestyle='--', color='green', label="Estimate SFRD (All)")
    print("sfrd all params", sfrd_all_params)

    print('HERE', max(csfrh), max(sfrd_all_fit))

    lookback_time_grid = cosmo.lookback_time(redshift_array).value  # in Gyr
    redshift_to_age = interp1d(redshift_array, lookback_time_grid, bounds_error=False, fill_value="extrapolate")
    age_to_redshift = interp1d(lookback_time_grid, redshift_array, bounds_error=False, fill_value="extrapolate")

    ax5 = ax3.secondary_xaxis('top', functions=(redshift_to_age, age_to_redshift))
    ax5.set_xlabel("Cosmic Lookback [Gyr]")
    ax5.set_xscale('log')
    ax5.xaxis.set_major_formatter(ScalarFormatter()) 
    ax5.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])

    all_snrd = []
    all_sfrd = []

    for idx, snap in enumerate(snaps):
        # read rate files
        rates_file = os.path.join(rates_folder, f"snapshot{snap}_rates.csv")
        subhalo_df = pd.read_csv(rates_file)

        if len(subhalo_df) <= 1:
            continue
        redshift = subhalo_df['z'].iloc[0]

        # plot snr vs sfr and snrd vs sfrd (halo level)
        ax1.scatter(subhalo_df['sfr'], subhalo_df["snr"], marker='.', color=colours[idx], label=f'z={round(redshift,3)}')
        ax2.scatter(subhalo_df['sfrd'], subhalo_df["snrd"], marker='.', color=colours[idx], label=f'z={round(redshift,3)}')

        all_sfrd.append(subhalo_df['sfrd'])
        all_snrd.append(subhalo_df['snrd'])

    # Densiy Scatter of all points
    final_sfrd = pd.concat(all_sfrd, ignore_index=True)
    final_snrd = pd.concat(all_snrd, ignore_index=True)
    x = np.array(final_sfrd.values)
    y = np.array(final_snrd.values)
    
    mask = (x > 0) & (y > 0)
    x_pos = x[mask]
    y_pos = y[mask]

    xbins = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), 100)
    ybins = np.logspace(np.log10(y_pos.min()), np.log10(y_pos.max()), 100)
    
    fig10, ax10 = plt.subplots(figsize=(8,5))
    counts, xedges, yedges, im = plt.hist2d(x_pos, y_pos, bins=[xbins, ybins], cmap='viridis', norm=LogNorm())
    counts_masked = ma.masked_invalid(counts)
    mesh = ax10.pcolormesh(xedges, yedges, counts_masked.T, cmap='viridis', norm=LogNorm())
    plt.colorbar(mesh, ax=ax10)
    ax10.set_yscale('log')
    ax10.set_xscale('log')

    # Create a single legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
            loc='lower center',
            ncol=4, frameon=False)

    ax3.legend(
        [sc1, sc11, sc2, sc21, line1, line11, line2, line3, line4, line8, line9], 
        #[sc1, sc21, line1, line2, line4], 
        ['SNRD (TNG)', 'SNRD (ALT)','SFRD (TNG - Top 100)', 'SFRD (TNG - All)', 'SFRD (Madau & Dickinson 2014)', 'SNRD (Madau & Dickinson 2014 (Salpeter))','SNRD (Function)', 'SFRD (Top 1000 - Function)', 'SFRD (All - Function)', 'SNRD (Scaled)', 'SNRD (ALT)'],
        #['SNRD (TNG)', 'SFRD (TNG - All)', 'Madau & Dickinson 2014', 'SNRD (Function)', 'SFRD (All - Function)'],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False)

    fig2.subplots_adjust(bottom=0.3, left=0.2)

    dfig = weighted_average(snaps)

    plt.close(dfig)
    plt.close(fig)
    #plt.close(fig2)
    plt.close(fig10)
    plt.show()

    return fig, fig2

# removed redshift 12 for now
snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]

build = False

if build == True:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(build_rates, snap) for snap in snapshots]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

dumb1, dumb2 = plot_rates(snapshots)