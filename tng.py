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
import astropy.units as u

from Objects.BPASSAnalysis import BPASSAnalysis
from Objects.BPASSDataFormatter import BPASSDataFormatter
from Objects.IMF import IMF
from Helpers.AICHelper import AICHelper
aic = AICHelper()

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import RLock
from scipy import stats
from scipy import integrate
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as ticker
from scipy.stats import linregress

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

import warnings

# Ignore only this type of user warning
warnings.filterwarnings("ignore", message=".*non-positive xlim.*")

#region setup
tqdm.set_lock(RLock())

MAX_WORKERS = 4
colours = ['cyan', 'blue', 'orange', 'magenta', 'red', 'yellow', 'brown', 'limegreen', 'purple', 'pink', 'grey', 'black']
colours = ['#800080',"#1a1a1a", "#3714ff", '#c0c0c0', "#1fb81f", '#40e0d0','#ffd700','#ffa500','#ff7f50',"#f53eff","#ff0000", '#87ceeb']
sn_colours = ['#FF5733', '#33FF57', '#3357FF', "#FFD012", "#B53DFF"]

bpass = BPASSDataFormatter()
allSupernovaArray, allIonizingArray, combinedSupernovaIon = bpass.getAllFormattedData()
bpassAnalysis = BPASSAnalysis(allSupernovaArray)
normIMF = IMF(1)
imf = IMF(normIMF.chabrier(0.9)/normIMF.salpeter(0.9))

rates_folder = f"/Users/dan/Code/FYP/Data/TNG/Rates_Err"

# count lines for progress bar
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
    # snType: 0: IIP, 1: II-Other, 2: Ib, 3: Ic, 4: Long-GRB, 5: Pair-Instab, 6: Low-mass
    #build_type = "IIP"
    #sn_type = 0
    build_types = {0:"IIP", 1:"II-Other", 2:"Ib", 3:"Ic"}
    for build, sn in build_types.items():
        #print(build, sn)
        position = snapshots.index(snap) + 1
        input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{snap}/*"
        my_glob = glob.glob(input_path)

        subhalo_rows = []
        for file_name in tqdm(my_glob, desc=f"Snapshot {snap} - {sn}", position=position, leave=False):
            subhalo_data = bpassAnalysis.subhaloData(file_name, build)#, pbar=pbar)
            subhalo_rows.append(dict(subhalo_data.items()))

        #print(f"    {len(subhalo_rows)} subhalos with postive rates")
        subhalo_df = pd.DataFrame(subhalo_rows)
        subhalo_df.to_csv(rates_folder + f"/{sn}/snapshot{snap}_rates.csv")

    return subhalo_df

# region Cosmic Densities
# calculate the sfrd and snrd for the top 1000 subhalos at each redshift
def calculate_densities(snaps, rates_folder_type):

    redshifts = []
    snrd_box = []
    sfrd_box = []
    snrd_mass = []
    snrd_no_mass = []
    snrd_box_errs = []
    
    for idx, snap in enumerate(snaps):
        # read rate files
        rates_file = os.path.join(rates_folder_type, f"snapshot{snap}_rates.csv")
        subhalo_df = pd.read_csv(rates_file)

        if len(subhalo_df) <= 1:
            continue

        # collect all redshifts to plot with the total snr of snapshots
        redshift = subhalo_df['z'].iloc[0]
        redshifts.append(redshift)

        # Calculate the box size properly (75000 ckpc/h)
        # gives in comving currenly
        h = 0.6774
        box_size_length = 75 / h
        box_size = pow(box_size_length,3)

        # total snrd in the box 
        # found by summing all the snr (each subhalo yr-1 Mpc-3)
        total_snr = sum(subhalo_df["snr"])
        total_snrd = total_snr / box_size
        snrd_box.append(total_snrd)

        # error in cosmic 
        squared_errs = pow(subhalo_df["snr_err"], 2)
        snrd_box_err = np.sqrt(sum(squared_errs))
        snrd_box_errs.append(snrd_box_err/box_size)

        #print(total_snrd, snrd_box_err/box_size)

        # total SNRD Calculations ( units yr-1 Mpc-3) - Not used 
        total_snr_new = sum(subhalo_df["snr_solar"] * subhalo_df["mass"])
        total_snrd_new = total_snr_new / box_size
        snrd_no_mass.append(total_snrd_new)

        # total SNRD Calculations (different units yr-1 Mo-1 Mpc-3)
        total_snr_mass = sum(subhalo_df["snr_solar"])
        total_snrd_mass = total_snr_mass / box_size
        snrd_mass.append(total_snrd_mass)

        # total star formation in the box 
        # note this only takes into account the top 1000 chosen
        total_sfr = sum(subhalo_df["sfr"])
        total_sfrd = total_sfr / box_size
        sfrd_box.append(total_sfrd)

    return redshifts, snrd_box, sfrd_box, snrd_no_mass, snrd_mass, snrd_box_errs

#region sfrd (all subhalos)
def calculated_sfrd():
    sfr_df = pd.read_csv(f"/Users/dan/Code/FYP/Data/TNG/total_sfr_per_redshift.csv")

    h = 0.6774
    box_size_length = 75000 * 1e-3 / h
    box_size = pow(box_size_length,3)

    sfrd_full = []
    redshifts = []
    for _, row in sfr_df.iterrows():
        sfrd = row["sfr"]/box_size
        sfrd_full.append(sfrd)
        redshifts.append(row['z'])

    return sfrd_full, redshifts     

# region averages
def average_rates(snaps):
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
    ax6.set_title('Averaged SFRD vs SNRD')
    ax6.set_xlabel(r'SFRD (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Gpc^{-3}}$]')
    ax6.set_ylabel(r'SNRD (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}\ Gpc^{-3}}$]')

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

    return fig3

# region MD2014 Formula
def sfrd_func(z, a, b, c, d):
    # takes madua and dickinson (Mpc3)
    sfrd = a * pow((1 + z), b)/(1 + pow((1 + z)/c, d)) 
    return sfrd

# curve fit values to MD2014
def curve_md14(redshifts, array, guess=1):
    # Initial guess use md14, mf17 or nv19
    if guess == 1:
        p0 = [0.015, 2.7, 2.9, 5.6]
    elif guess == 2:
        p0 = [0.01, 2.6, 3.2, 6.2]
    elif guess == 3:
        p0 = [0.01, 2.77, 2.9, 4.7]

    params, cov = curve_fit(sfrd_func, redshifts, array, p0=p0)
    #print('params', params)
    x_linespace = np.linspace(redshifts.min(), redshifts.max(), 300)
    md14_fit = sfrd_func(x_linespace, *params)

    return md14_fit, [params,cov]

# generate figure and axes
def plt_helper(size1, size2, xlabel, ylabel, logx=True, logy=True, legendspace=None):
    fig, ax = plt.subplots(figsize=(size1,size2))
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    if logx: 
        ax.set_xscale('log')

    if logy: 
        ax.set_yscale('log')

    if legendspace is not None:
        fig.subplots_adjust(bottom=legendspace)

    ax.tick_params(axis='both', labelsize=18)

    return fig, ax

# set labels for axes 
def plt_labels(fig, ax, col, gap=None):
    handles, labels = ax.get_legend_handles_labels()
    # tempoararily remove legend 
    fig.legend(handles, labels,loc='lower center',ncol=col, frameon=False)#, fontsize=22, markerscale=3)

    if gap != None:
        fig.tight_layout(rect=[0, gap, 1, 1])
        # tempoararily remove legend 
        #fig.tight_layout(rect=[0, 0, 1, 1])

    return fig, ax

# set labels for axes 
def plt_labels_multiple(fig, axs, col):
    handles = []
    labels = []
    for ax in axs:
        handle, label = ax.get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(label)
    fig.legend(handles, labels,loc='lower center',ncol=col, frameon=False)

    return fig, axs

# set up cosmic history plots
def plt_cosmo(redshifts, ylabel, ytwin=None, space=None):

    fig_cosmo, ax_cosmo1 = plt_helper(8, 7, 'Redshift (z)', ylabel, logx=False, legendspace=space)
    
    redshifts = [0] + redshifts
    lookback_time_grid = cosmo.lookback_time(redshifts).value  # in Gyr
    redshift_to_age = interp1d(redshifts, lookback_time_grid, bounds_error=False, fill_value="extrapolate")
    age_to_redshift = interp1d(lookback_time_grid, redshifts, bounds_error=False, fill_value="extrapolate")

    ax_cosmic = ax_cosmo1.secondary_xaxis('top', functions=(redshift_to_age, age_to_redshift))
    ax_cosmic.set_xlabel("Cosmic Lookback [Gyr]")
    ax_cosmic.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_cosmic.ticklabel_format(style='plain', axis='x')

    if ytwin is not None:
        ax_cosmo2 = ax_cosmo1.twinx()
        ax_cosmo2.set_ylabel(ytwin)
        ax_cosmo1.tick_params(axis='y', labelcolor='Red')
        ax_cosmo2.tick_params(axis='y', labelcolor='Blue')
        ax_cosmo2.set_yscale('log')
    else:
        ax_cosmo2 = None
    
    ax_cosmo1.set_yscale('log')
    #ax_cosmo1.set_xscale('log')

    ax_cosmo1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_cosmo1.ticklabel_format(style='plain', axis='x')

    return fig_cosmo, ax_cosmo1, ax_cosmo2

# average halo level densities
def average_rate_densities(snrd, sfrd):
    av_snrd = sum(snrd)/len(snrd)
    av_sfrd = sum(sfrd)/len(sfrd)

    return av_snrd, av_sfrd

# line fit
def line_fit(x, y):
    # fit line
    x_line = np.linspace(np.array(x).min(), np.array(x).max(), 200)
    m,c = np.polyfit(x, y, 1) 
    y_line = m * x_line + c

    return x_line, y_line, m, c

def log_line(x1,y1):
    x = np.asarray(x1)
    y = np.asarray(y1)
    mask = (x>0) & (y>0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    x_log = np.log10(x)
    y_log = np.log10(y)
    res = linregress(x_log, y_log)
    grad = res.slope
    #inter = np.exp(res.intercept)
    inter = 10**res.intercept
    return grad, inter, inter*x1**grad

# region plot halo level
def halo_level(snaps, rates_folder_type):
    
    # set up figures and axes
    fig_halo_rate, ax_hr = plt_helper(8,6, r'SFR (Star Formation Rate) [$\mathrm{M_\odot\ yr^{-1}}$]', r'SNR (Supernova Rate) [$\mathrm{yr^{-1}}$]', legendspace=0.2)
    fig_halo_density, ax_hrd = plt_helper(8,6, r'Volumetric SFR [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', r'Volumetric SNR [$\mathrm{yr^{-1}\ Mpc^{-3}}$]', legendspace=0.2)
    fig_hist, ax_hist = plt_helper(8,7, r'SFRD (Star Formation Rate Density) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', r'SNRD (Supernova Rate Density) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]')
    fig_dense, ax_dense = plt_helper(8, 7, r'SFRD (Star Formation Rate Density) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', r'SNRD (Supernova Rate Density) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]', legendspace=0.2)
    fig_res, ax_res = plt_helper(8, 7, r'SFRD (Star Formation Rate Density) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', r'SNRD (Supernova Rate Density) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]', legendspace=0.2, logx=False, logy=False)
    fig_av, ax_av = plt_helper(8, 7, r'SFRD (Star Formation Rate Density) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', r'SNRD (Supernova Rate Density) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]', legendspace=0.15, logx=False, logy=False)
    fig_mass, ax_mass = plt_helper(8, 6, r'Mass [$\mathrm{M_\odot}$]', r'SNR (Supernova Rate) [$\mathrm{yr^{-1}\ M_\odot^{-1}}$]', legendspace=0.1)

    all_snrd = []
    all_sfrd = []

    all_av_snrd = []
    all_av_sfrd = []

    all_snr_solar = []
    all_sfr = []
    all_snr = []
    all_mass = []

    for idx, snap in enumerate(snaps):
        # read rate files
        rates_file = os.path.join(rates_folder_type, f"snapshot{snap}_rates.csv")
        subhalo_df = pd.read_csv(rates_file)

        if len(subhalo_df) <= 1:
            continue
        redshift = subhalo_df['z'].iloc[0]

        # plot snr vs sfr and snrd vs sfrd (halo level)
        ax_hr.scatter(subhalo_df['sfr'], subhalo_df["snr"], marker='.', color=colours[idx], label=f'z={round(redshift,3)}')
        #kcc, kcc_inter, log_fit1 = log_line(subhalo_df['sfr'], subhalo_df["snr"])
        #ax_hr.plot(subhalo_df['sfr'], log_fit1, color=colours[idx], label=f'kcc: {kcc:.2}, a:{kcc_inter:.2}')
        
        # convert to Mpc3 from Gpc3 => times 1e-9
        sfrd = subhalo_df['sfrd']*1e-9
        snrd = subhalo_df["snrd"]*1e-9
        ax_hrd.scatter(sfrd, snrd, marker='.', color=colours[idx], label=f'z={round(redshift,3)}')

        # snr_solar should be calculated as sum(pixel_snr)/sum(mass)
        # these values corrosponds to bpass paper 
        halo_mass = subhalo_df['mass']
        snr_solar = subhalo_df["snr"]/halo_mass
        ax_mass.scatter(halo_mass, snr_solar, marker='.', color=colours[idx], label=f'z={round(redshift,3)}')
        #B_ms, A_ms, log_fit2 = log_line(halo_mass, snr_solar)  
        #ax_mass.plot(halo_mass, log_fit2, color=colours[idx], label=f'B: {B_ms:.2}, Log(A): {np.log10(A_ms):.1f}')

        # add to list to be used in 2d density histogram
        all_sfrd.append(sfrd)
        all_snrd.append(snrd)

        # add to average list 
        # average should be snrd in yr-1 Mpc-3 and sfrd in Mo yr-1 Mpc-3
        # this means an effective kcc scaling can be calculated in Mo-1 (should be linear)
        # snrd is now calulated in these unit in BPASS analysis as event rates are in yr-1
        av_snrd, av_sfrd = average_rate_densities(snrd, sfrd)
        all_av_snrd.append(av_snrd)
        all_av_sfrd.append(av_sfrd)

        all_mass.extend(halo_mass)
        all_snr_solar.extend(snr_solar)
        all_sfr.extend(subhalo_df['sfr'])
        all_snr.extend(subhalo_df['snr'])

    # Density Scatter of all points
    final_sfrd = pd.concat(all_sfrd, ignore_index=True)
    final_snrd = pd.concat(all_snrd, ignore_index=True)
    x = np.array(final_sfrd.values)
    y = np.array(final_snrd.values)
    
    # mask out zeros
    mask = (x > 0) & (y > 0)
    x_pos = x[mask]
    y_pos = y[mask]

    # log colour bins
    xbins = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), 150)
    ybins = np.logspace(np.log10(y_pos.min()), np.log10(y_pos.max()), 150)
    counts, xedges, yedges, im = ax_hist.hist2d(x_pos, y_pos, bins=[xbins, ybins], cmap='viridis', norm=LogNorm())
    cbar = fig_hist.colorbar(im, ax=ax_hist)
    cbar.set_label("Point Density Across All Redshifts")

    # Find which bin each point falls into
    x_indices = np.searchsorted(xedges, x_pos, side='right') - 1
    y_indices = np.searchsorted(yedges, y_pos, side='right') - 1
    x_indices = np.clip(x_indices, 0, counts.shape[0] - 1)
    y_indices = np.clip(y_indices, 0, counts.shape[1] - 1)

    # Get density and filter
    point_densities = counts[x_indices, y_indices]
    density_threshold = 5
    density_mask = point_densities >= density_threshold

    # filter and plot
    x_filtered = x_pos[density_mask]
    y_filtered = y_pos[density_mask]
    ax_dense.scatter(x_filtered, y_filtered, color='black', label=f'Scatter (Point Density > {density_threshold})', marker='.', s=0.3)

    # Fit a straight line in log-space
    #log_x = np.log10(x_filtered)
    #log_y = np.log10(y_filtered)
    #coeffs = np.polyfit(log_x, log_y, 1)
    #slope, intercept = coeffs
    #x_line = np.logspace(np.log10(x_filtered.min()), np.log10(x_filtered.max()), 200)
    #y_line = 10**(slope * np.log10(x_line) + intercept)

    slope, intercept = np.polyfit(x_filtered, y_filtered, 1)
    x_line = np.linspace(x_filtered.min(), x_filtered.max(), 200)
    y_line = slope * x_line + intercept

    y_pred = slope * x_filtered + intercept
    ss_res = np.sum((y_filtered - y_pred)**2)        # residual sum of squares
    ss_tot = np.sum((y_filtered - np.mean(y_filtered))**2)
    r2 = 1 - ss_res / ss_tot

    rmse = np.sqrt(np.mean((y_filtered - y_pred)**2))

    residuals = y_filtered - y_pred
    ax_res.scatter(x_filtered, residuals, marker='.')
    ax_res.axhline(0, color='black', linestyle='--')

    ax_dense.plot(x_line, y_line, color='red', linewidth=1.5, label=f'Slope={slope:.4f}, Intercept={intercept:.2f}, r2: {r2:.2}, rmse: {rmse:.2}')
    print("R²:", r2)
    print("RMSE:", rmse)
    print(f' Dense 1: Slope={slope:.4}, Intercept={intercept:.2}')

    kcc, inter, log_fit = log_line(x_filtered, y_filtered)
    #ax_dense.plot(x_filtered, log_fit, color='cyan', linewidth=1.5, label=f'Slope={kcc:.4}, Intercept={inter:.2}, mean={np.mean(log_fit/x_filtered):.2}')
    #print(f' Dense 2: Slope={kcc:.4}, Intercept={inter:.2}, mean={np.mean(log_fit/x_filtered)}')

    # average plots 
    x_line, y_line, av_slope, av_intercept = line_fit(all_av_sfrd, all_av_snrd)
    # standard deviation ddof=1 gives sample std
    snrd_dev = np.std(all_av_snrd, ddof=1)
    # standard error of the mean
    snrd_yerr = snrd_dev / np.sqrt(len(all_av_snrd))
    ax_av.errorbar(all_av_sfrd, all_av_snrd, yerr=snrd_yerr, color='black', fmt='x', capsize=5, label="Average")
    ax_av.plot(x_line, y_line, linestyle='--', color='blue', label=f'Slope={av_slope:.2e}, Intercept={av_intercept:.2e}')

    # labels
    plt_labels(fig_halo_rate, ax_hr, 4, 0.2)
    plt_labels(fig_halo_density, ax_hrd, 4, 0.2)
    plt_labels(fig_dense, ax_dense, 2)
    plt_labels(fig_av, ax_av, 2)
    plt_labels(fig_mass, ax_mass, 4, 0.2)

    B_ms, A_ms, log_fit3 = log_line(all_mass, all_snr_solar)
    ax_mass.plot(all_mass, log_fit3, color='black', label=f'B: {B_ms:.2}, Log(A):{np.log10(A_ms):.2f}')
    ax_mass.plot(halo_mass, (10**-6.5)*halo_mass**-0.58, color='black', label=f'Graur et al. 2015')

    kcc, kcc_inter, log_fit4 = log_line(all_sfr, all_snr)
    ax_hr.plot(all_sfr, log_fit4, color='black', label=f'kcc: {kcc:.2}, inter:{kcc_inter:.2}, test: {np.mean(log_fit4/all_sfr):.2}')

    ax_dense.set_yscale('linear')
    ax_dense.set_xscale('linear')

    return True

"""
def supernova_efficiency(_imf, sn_type=20):
    # Madau & Dickinson 2014
    # k_CC = integral(phi(m), m_min, m_max) / integral(m * phi(m), m_l, m_u)
    # bpass - max = 100, min = 1

    denominator, _ = integrate.quad(lambda m: m * _imf(m), 1, 100)
    if sn_type == 20:
        # up to 25 solar masses for type 2 
        numerator, _ = integrate.quad(_imf, 8, 25)
        #numerator, _ = integrate.quad(_imf, 8, 100)
    elif sn_type == 10:
        # 25 to 100 solar mass for type 1b/c
        numerator, _ = integrate.quad(_imf, 25, 100)
        #numerator, _ = integrate.quad(_imf, 8, 100)

    return numerator/denominator
"""

def supernova_efficiency(_imf, sn_type=20):
    # Madau & Dickinson 2014
    # k_CC = integral(phi(m), m_min, m_max) / integral(m * phi(m), m_l, m_u)
    # bpass - max = 100, min = 1

    # Smith 2014
    # Binaries: IIP 8.5-18.7, II-Other 18.7-37, Ib/c 37-100
    # IIP 8.5-13.7, II-Other 13.7-22, Ibc 22-100

    denominator, _ = integrate.quad(lambda m: m * _imf(m), 1, 100)
    if sn_type == 20:
        # up to 25 solar masses for type IIP
        #numerator, _ = integrate.quad(_imf, 8.5, 13.7)
        numerator, _ = integrate.quad(_imf, 8.5, 18.7)
    elif sn_type == 25:
        # up to 25 solar masses for type II-oither 
        #numerator, _ = integrate.quad(_imf, 13.7, 22)
        numerator, _ = integrate.quad(_imf, 18.7, 23.1)
    elif sn_type == 10:
        # 25 to 100 solar mass for type 1b/c
        #numerator, _ = integrate.quad(_imf, 22, 100)
        numerator, _ = integrate.quad(_imf, 37, 100)
    elif sn_type == 5:
        # approximation for star formation conversion 
        # 8 to 100 solar masses (include all regions)
        numerator, _ = integrate.quad(_imf, 8, 100)

    return numerator/denominator

def imf_scaling(_imf):
    integral, _ = integrate.quad(_imf, 0.1, 1)
    return integral

# region Plot Cosmic Level
def cosmic_level(snaps, kcc_type, rates_folder_type):
    redshifts, snrd, sfrd_1000, snrd_alt, snrd_mass, snrd_err = calculate_densities(snaps, rates_folder_type)
    sfrd_all, _ = calculated_sfrd()

    # order arrays to be ascedning 
    rev_redshifts = np.array(redshifts)[::-1]
    # sfrd
    rev_sfrd_1000 = np.array(sfrd_1000)[::-1] 
    rev_sfrd_all= np.array(sfrd_all)[::-1] 
    # snrd
    rev_snrd = np.array(snrd)[::-1]
    rev_snrd_err = np.array(snrd_err)[::-1]

    rev_snrd_alt = np.array(snrd_alt)[::-1]
    rev_snrd_mass = np.array(snrd_mass)[::-1]

    # scaling
    # calculate a scale factor for each redshift value to convert from top 1000 halos to all halos
    # SFRD values are compared between the two data sets 
    # this can be applied to the SNRD to get SNRD estimates for the whole box 
    sfrd_scaling = rev_sfrd_all/rev_sfrd_1000

    # apply scaling
    rev_snrd_1000_scaled = rev_snrd * sfrd_scaling 
    rev_snrd_1000_scaled_err = rev_snrd_err * sfrd_scaling 
    rev_snrd_alt_scaled = rev_snrd_alt * sfrd_scaling
    rev_snrd_mass_scaled = rev_snrd_mass * sfrd_scaling

    # madau and dickinson 2014
    # their fomrula ius given in Mpc3 we have Gpc3
    # convert from comoving to physical by multiplying by (1+z)^3
    redshift_linespace = np.linspace(rev_redshifts.min(), rev_redshifts.max(), 300)
    csfrh = 0.015 * pow((1 + redshift_linespace), 2.7)/(1 + pow((1 + redshift_linespace)/2.9, 5.6)) #* 1e9 #* pow((1 + redshift_linespace),3)
    mf17 = 0.01 * pow((1 + redshift_linespace), 2.6)/(1 + pow((1 + redshift_linespace)/3.2, 6.2))
    nv19 = 0.01 * pow((1 + redshift_linespace), 2.77)/(1 + pow((1 + redshift_linespace)/2.9, 4.7))  

    # snrd curve fits
    md14_snrd_scaled, _ = curve_md14(rev_redshifts, rev_snrd_1000_scaled)
    md14_snrd_alt_scaled, _ = curve_md14(rev_redshifts, rev_snrd_alt_scaled)
    md14_snrd_mass_scaled, _ = curve_md14(rev_redshifts, rev_snrd_mass_scaled)

    # sfrd 
    md14_sfrd_all, _ = curve_md14(rev_redshifts, rev_sfrd_all)

    # quoted core collapse efficiency scaling for salpeter
    # we will want a chabrier (need to caluclate it)
    # use IMFs defined previously 
    # formula is integral imf / integral mass * imf 
    # kcc given in Mo-1 

    kcc_chabrier = supernova_efficiency(imf.chabrier, kcc_type)
    #kcc_salpeter = supernova_efficiency(imf.salpeter, kcc_type)
    #kcc_chabrier_sys = supernova_efficiency(imf.chabrierSystem, kcc_type)

    #print(' CCSN', kcc_type, kcc_chabrier)
    csnrh_chabrier = csfrh * kcc_chabrier
    #csnrh_salpeter = csfrh * kcc_salpeter
    #csnrh_chabrier_sys = csfrh * kcc_chabrier_sys

    # trace sfrd using snrd 
    # use new units snrd in yr-1 Mpc-3 and divide by kcc (Mo-1) gets Mo yr-1 Mpc-3 (SFRD)
    csfrh_kcc_chabrier = md14_snrd_scaled / kcc_chabrier
    csfrh_kcc_chabrier_raw = rev_snrd_1000_scaled / kcc_chabrier
    csfrh_kcc_chabrier_raw_err = rev_snrd_1000_scaled_err / kcc_chabrier
    #print(' SFR', rev_snrd_1000_scaled[7])

    artificial = 4 * md14_snrd_scaled / kcc_chabrier

    chab_scale = imf_scaling(imf.chabrier)
    csfrh_bpass_scale = csfrh_kcc_chabrier / chab_scale

    
    # CSNRH plot
    fig_csnrh, ax_csnrh1, ax_csnrh2 = plt_cosmo(rev_redshifts, r'SNRD (Supernova) [$\mathrm{yr^{-1}\ M_\odot^{-1}\ Mpc^{-3}}$]', r'SNRD (Supernova) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]', space=0.2)
    # scatters
    sc_csnrh2 = ax_csnrh2.scatter(rev_redshifts, rev_snrd_1000_scaled, color='navy', label="TNG100", marker='.')
    #sc_csnrh4 = ax_csnrh2.scatter(rev_redshifts, rev_snrd_alt_scaled, color='Aqua', label='TNG100 - Alternate', marker='.')
    sc_csnrh4 = ax_csnrh1.scatter(rev_redshifts, rev_snrd_mass_scaled, color='firebrick', label='TNG100 - Mass Units', marker='.')
    # lines
    ls_csnrh2, = ax_csnrh2.plot(redshift_linespace, md14_snrd_scaled, linestyle='--', color='navy', label="TNG100")
    #ls_csnrh4, = ax_csnrh2.plot(redshift_linespace, md14_snrd_alt_scaled, linestyle='--', color='Aqua', label="TNG100 - Alternate")
    ls_csnrh4, = ax_csnrh1.plot(redshift_linespace, md14_snrd_mass_scaled, linestyle='--', color='firebrick', label="TNG100 - Mass units")
    ls_csnrh9, = ax_csnrh2.plot(redshift_linespace, csnrh_chabrier, linestyle='-', color='blue', label=f"MD14 (kcc={kcc_chabrier:.3})")
    #ls_csnrh9, = ax_csnrh2.plot(redshift_linespace, csnrh_chabrier_sys, linestyle=':', color='black', label="MD14 - Chabrier System")

    # CSFRH plot
    fig_csfrh, ax_csfrd, _ = plt_cosmo(rev_redshifts, r'SFRD (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', space=0.20)
    # scatters
    sc2 = ax_csfrd.scatter(rev_redshifts, rev_sfrd_all, color='lime', label='TNG100 - All', marker='.')
    sc3 = ax_csfrd.scatter(rev_redshifts, csfrh_kcc_chabrier_raw, color='gold', label="TNG100 - Kcc Chabrier", marker='.')
    # lines
    ls_csfrh2, = ax_csfrd.plot(redshift_linespace, md14_sfrd_all, linestyle='--', color='lime', label="TNG100 - All")
    ls_csfrh3, = ax_csfrd.plot(redshift_linespace, csfrh, linestyle='-', color='teal', label="Madau & Dickinson 2014")
    ls_csfrh4, = ax_csfrd.plot(redshift_linespace, csfrh_kcc_chabrier, linestyle='--', color='gold', label=f"TNG100 (kcc={kcc_chabrier:.3})")
    ls_csfrh5, = ax_csfrd.plot(redshift_linespace, artificial, linestyle='--', color='orange', label=f"Artifical (Gold * 4) (kcc={kcc_chabrier:.3})")
    ls_csfrh5, = ax_csfrd.plot(redshift_linespace, csfrh_bpass_scale, linestyle='--', color='red', label=f"Chabrier Scaling (0.1-1 Mo)")
    #ls_csfrh3, = ax_csfrd.plot(redshift_linespace, csfrh_kcc_chabrier_noscale, linestyle='--', color='gold', label="TNG100 - Kcc Chabrier no scale)")

    plt_labels_multiple(fig_csnrh, [ax_csnrh1, ax_csnrh2], 2)
    plt_labels(fig_csfrh, ax_csfrd, 2)

    #return md14_snrd_scaled, csfrh_kcc_chabrier, csnrh_chabrier, csfrh
    return rev_snrd_1000_scaled, csfrh_kcc_chabrier_raw, csnrh_chabrier, csfrh, rev_sfrd_all, rev_sfrd_1000, mf17, nv19, rev_snrd_1000_scaled_err, csfrh_kcc_chabrier_raw_err

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# region Build
snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]

build = False

if build == True:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(build_rates, snap) for snap in snapshots]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#region Graphs
_, redshifts = calculated_sfrd()
rev_redshifts = np.array(redshifts)[::-1]
redshift_linespace = np.linspace(rev_redshifts.min(), rev_redshifts.max(), 300)
fig_types1, ax_types1, _ = plt_cosmo(rev_redshifts, r'Volumetric SNR [$\mathrm{yr^{-1}\ Mpc^{-3}}$]')#, space=0.2)
fig_types2, ax_types2, _ = plt_cosmo(rev_redshifts, r'SFRD (Star Formation) [$\mathrm{yr^{-1}\ Mpc^{-3}}$]', space=0.2)

fig_total_sfr, ax_total_sfr, _ = plt_cosmo(rev_redshifts, r'SFRD (Star Formation) [$\mathrm{M_\odot\ yr^{-1}\ Mpc^{-3}}$]', space=0.3)
fig_total_snr, ax_total_snr, _ = plt_cosmo(rev_redshifts, r'SNRD (Supernova) [$\mathrm{10^{-4} yr^{-1}\ Mpc^{-3}}$]', space=0.2)

all_sn_types = ["IIP", "II-Other", "Ib", "Ic"]

total_sfr = []
total_snr_arr = []
total_snr_err_arr = []

test_dict = {}

for i, sn_type in enumerate(all_sn_types):
    rates_folder_type = rates_folder + f"/{sn_type}"
    print(sn_type)
    if sn_type in ["IIP"]:
        kcc_type = 20
    elif sn_type in ["II-Other"]:
        kcc_type = 25
    elif sn_type in ["Ib", "Ic"]:
        kcc_type = 10
    else:
        kcc_type = None

    # call functions to get rates
    halo_level(snapshots, rates_folder_type) # dont need halo at the moment 
    snrd, sfrh, snrd_md14_, sfrh_md14, sfrh_halos, sfrh_1000, sfrh_mf17, sfrh_nv19, snrd_err, sfrh_err  = cosmic_level(snapshots, kcc_type, rates_folder_type)

    # sum the sf and sn rates to get a total
    if len(total_sfr) == 0:
        total_sfr = sfrh
    else:
        total_sfr = total_sfr + sfrh

    """
    if len(total_snr) == 0:
        total_snr = snrd
        total_snr_err = snrd_err
    else:
        total_snr = total_snr + snrd
        total_snr_err = total_snr_err + snrd_err
    """

    total_snr_arr.append(snrd)
    total_snr_err_arr.append(snrd_err)

    print(' SNR:', ', '.join(f'{s * 10**4:.4f}' for s in snrd))
    #print(' Total SNR:', total_snr[7])

    plot_names = ['1', '2', '3', '4', 'halo_rates', 'halo_rate_density', 'halo_hist', 'halo_hist_reduced', 'halo_residuals', 'halo_average', 'halo_snr_solar', 'cosmic_snr', 'cosmic_sfr']
    for idx, fig_num in enumerate(plt.get_fignums()):
        if idx > 3:
            curr_fig = plt.figure(fig_num)
            plt.figure(fig_num).savefig(f"Data/Images/TNG/final/{sn_type}/{plot_names[idx]}.png", dpi=300)
            plt.close(curr_fig)

    ax_types1.plot(rev_redshifts, snrd, label=f'{sn_type}', color=sn_colours[i], zorder=10)
    ax_types1.errorbar(rev_redshifts, snrd, yerr=snrd_err, color='black', capsize=5, zorder=5)
    ax_types2.plot(rev_redshifts, sfrh, label=f'{sn_type}', color=sn_colours[i])
    test_dict[sn_type] = [sfrh, snrd]

total_snr = sum(total_snr_arr)
total_snr_err = np.sqrt(sum(total_snr_err_arr))
total_snr_err = sum(total_snr_err_arr)
#print('here', total_snr, total_snr_err)
ax_types1.plot(rev_redshifts, total_snr, label=f'Total', linestyle='--', color=sn_colours[4])
ax_types1.errorbar(rev_redshifts, total_snr, yerr=total_snr_err, color='black', capsize=5, zorder=5)
ax_types2.plot(rev_redshifts, total_sfr, label=f'Total', linestyle='--', color=sn_colours[4])
plt_labels(fig_types1, ax_types1, 3)
plt_labels(fig_types2, ax_types2, 3)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# kcc calculations 

# apply a generic kcc scaling the star formation predictions
# there is no supernova types depdenance so cannot dynamically change it 
kcc_IIP = supernova_efficiency(imf.chabrier, 20)
kcc_II_Other = supernova_efficiency(imf.chabrier, 25)
kcc_Ibc = supernova_efficiency(imf.chabrier, 10)

# averaged
kcc_avg = (kcc_IIP + kcc_II_Other + kcc_Ibc)/3

# with fraction scaling
kcc_1 = ((0.4388 * kcc_IIP) + (0.2263 * kcc_II_Other) + ((0.2268 + 0.1081)* kcc_Ibc))
kcc_1_lower = (((0.4388 + 0.044)* kcc_IIP) + ((0.2263 + 0.0199) * kcc_II_Other) + ((0.2268 + 0.0152 + 0.1081 + 0.041)* kcc_Ibc)) /(0.4388 + 0.044 + 0.2263 + 0.0199 + 0.2268 + 0.0152 + 0.1081 + 0.041)
kcc_1_upper = (((0.4388 - 0.044)* kcc_IIP) + ((0.2263 - 0.0199) * kcc_II_Other) + (((0.2268 - 0.0152) + (0.1081 - 0.041))* kcc_Ibc)) /((0.4388 - 0.044) + (0.2263 - 0.0199) + (0.2268 - 0.0152) + (0.1081 - 0.041))
# from halo fractions
#kcc_2 = ((0.469 * kcc_IIP) + (0.248 * kcc_II_Other) + ((0.184 + 0.099)*kcc_Ibc))

print('scaling avg;', kcc_avg)
print('scaling weighted;', kcc_1)
print('scaling upper;', kcc_1_upper)
print('scaling lower;', kcc_1_lower)
print('scaling error;', kcc_1_upper - kcc_1_lower)

# NOTE: upper and lower kcc limits are extrenely close to each other so are not used 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#region STAR Formation graph
# curve fit - guess using nv19 (closest to our values)
curve_sfr, _ = curve_md14(rev_redshifts, total_sfr, 3) # DONT THINK THIS IS RIGHT 
curve_all, _ = curve_md14(rev_redshifts, sfrh_halos, 3)
curve_100, _ = curve_md14(rev_redshifts, sfrh_1000, 3)

#print('total snr', *(total_snr*10**4))
print('total snr:', ', '.join(f'{s * 10**4:.4f}' for s in total_snr))
print('redshifts:', ', '.join(f'{r:.4f}' for r in rev_redshifts))
curve_snr, _ = curve_md14(rev_redshifts, total_snr, 3)
curve_sfr, params = curve_md14(rev_redshifts, total_snr/kcc_1, 3) # Apply kcc to snrd after averaging (accounts for fractions)
print('Params:',*params[0])
print('errors:',np.sqrt(np.diag(params[1])))

# plot lines
#ax_total_sfr.fill_between(redshift_linespace, curve_snr/kcc_IIP, curve_snr/kcc_Ibc, color='orange', alpha=0.1, label=f'[{kcc_Ibc:.4f} < kcc <  {kcc_IIP:.4f}]')
ax_total_sfr.plot(redshift_linespace, sfrh_md14, label=f'Madau & Dickinson (2014)', color='navy', ls='--')
ax_total_sfr.plot(redshift_linespace, sfrh_mf17, label=f'Madau & Fragos (2017)', color='purple', ls='--')
ax_total_sfr.plot(redshift_linespace, sfrh_nv19, label=f'Neijssel et. al (2019)', color='cyan', ls='--')
ax_total_sfr.plot(redshift_linespace, curve_sfr, label=f'Curve Fit - Current Study (Prediction From SNRD)', color='orange')
ax_total_sfr.plot(redshift_linespace, curve_all, label=f'Curve Fit - TNG100-1 (All Halos)', color='lime')
ax_total_sfr.plot(redshift_linespace, curve_100, label=f'Curve Fit - TNG100-1 (Top 1000 Halos)', color='forestgreen')

# scatter - no longer plotting total_sfr -> throws fractionbs out and makes total sfr too large
ax_total_sfr.scatter(rev_redshifts, total_snr/kcc_1, label=f'Current Study (Prediction From SNRD)', color='orange', marker='D', edgecolors='black')
ax_total_sfr.scatter(rev_redshifts, sfrh_halos, label=f'TNG100-1 (All Halos)', color='lime', marker='D', edgecolors='black')
ax_total_sfr.scatter(rev_redshifts, sfrh_1000, label=f'TNG100-1 (Top 1000 Halos)', color='forestgreen', marker='D', edgecolors='black')

plt_labels(fig_total_sfr, ax_total_sfr, 2, 0.17)
ax_total_sfr.set_yscale('linear')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#region Supernova rate graph

# prep predictions again (could move them out of func)
# remake md14 as we do not want to use snrd_md14_ from above as this will have kcc corrosponding to Ic supernova (high mass)
# this will be smaller than the averaged prediction from above 
snrd_md14 = 0.015 * pow((1 + redshift_linespace), 2.7)/(1 + pow((1 + redshift_linespace)/2.9, 5.6)) * kcc_1
snrd_mf17 = 0.01 * pow((1 + redshift_linespace), 2.6)/(1 + pow((1 + redshift_linespace)/3.2, 6.2)) * kcc_1
sfrd_nv19 = 0.01 * pow((1 + redshift_linespace), 2.77)/(1 + pow((1 + redshift_linespace)/2.9, 4.7))

# plot lines
ax_total_snr.plot(redshift_linespace, snrd_md14, label=f'Madau & Dickinson (2014) [kcc={kcc_1:.4f}]', color='navy', ls='--')
#ax_total_snr.plot(redshift_linespace, snrd_mf17, label=f'Madau & Fragos (2017)', color='purple', ls='--')
ax_total_snr.plot(redshift_linespace, sfrd_nv19 * kcc_avg, label=f'Neijssel et. al (2019) [kcc={kcc_avg:.4f}]', color='cyan', ls='--')
ax_total_snr.plot(redshift_linespace, sfrd_nv19 * kcc_1, label=f'Neijssel et. al (2019) [kcc={kcc_1:.4f}]', color='darkturquoise', ls='--')
ax_total_snr.fill_between(redshift_linespace, sfrd_nv19 * kcc_IIP, sfrd_nv19 * kcc_Ibc, color='royalblue', alpha=0.1, label=f'[{kcc_Ibc:.4f} < kcc <  {kcc_IIP:.4f}]')
ax_total_snr.plot(redshift_linespace, curve_snr, label=f'Current Study (Curve Fit)', color='orange', zorder=99)

# scatter
ax_total_snr.scatter(rev_redshifts, total_snr, label=f'Current Study', color='orange', marker='D', edgecolors='black', zorder=100)
ax_total_snr.errorbar(rev_redshifts, total_snr, yerr=total_snr_err, color='black',capsize=5)#, marker='D', zorder=90)
#ax_total_snr.scatter(rev_redshifts, sfrh_halos * kcc, label=f'TNG100-1 (All Halos)', color='lime', marker='D', edgecolors='black')

# plot data points
H0 = 67.74 # km/s/Mpc from the TNG project
h70 = H0/70 
h75 = H0/75
rho_b = 1.82e8 * h75 #Norberg et al -> luminosity density 
c1999 = ((0.48/100) /1e10 ) * rho_b # conversion to yr-1 Mpc-3 using  1 SN (100yr)−1 (1010LBo)-1 abd h75
c1999_err = ((0.23/100) /1e10 ) * rho_b
print(f"Hubble, {h70}, {h75}")
ax_total_snr.errorbar(0, 0.258e-4+0.447e-4, yerr=0.072e-4+0.139e-4, capsize=5, color='olive', label='Li et al. (2011)', fmt='o') #SN Mpc−3 yr−1 -> no conversion required - combimnation of II and Ibc rates
#ax_total_snr.errorbar(0, 1.5e-4, yerr=[[0.3e-4], [0.4e-4]], capsize=5, color='brown', label='Mattila et al. (2012)', fmt='o')
ax_total_snr.errorbar(0, c1999, yerr=[[c1999_err], [c1999_err]], capsize=5, color='gold', label='Cappellaro et al. (1999)', fmt='o') # converted using rate in SNu -> 1 SNu = 1 SN (100yr)−1 (1010LBo)-1
ax_total_snr.errorbar(0.0149, 0.7e-4 * h70**3, yerr=[[0.09e-4 * h70**3], [0.1e-4 * h70**3]], capsize=5, color='maroon', label='Pessi et al. (2025)', fmt='o') #yr−1Mpc−3h370 -> remove h depdendance (has fractiosn and indifical rates as well)
ax_total_snr.errorbar(0.072, 1.06e-4 * h70**3, yerr=[[0.19e-4 * h70**3], [0.19e-4 * h70**3]], xerr=[[0.009], [0.009]], capsize=5, color='hotpink', label='Taylor et al. (2014)', fmt='o') #(h/0.7)3 /(yr Mpc3)-> remove h70 dependance
ax_total_snr.errorbar(0.1, 0.688e-4 * h70**3, yerr=[[0.078e-4 * h70**3], [0.078e-4 * h70**3]], capsize=5, color='darkgrey', label='Ma, Xiaoran et al. (2025)', fmt='o') #yr−1Mpc−3h370 -> remove h depdendance
ax_total_snr.errorbar(0.26, 2.2e-4 * h75**3, yerr=[[0.7e-4 * h75**3], [0.8e-4 * h75**3]], capsize=5, color='black', label='Cappellaro et al. (2005)', fmt='o') # h3yr−1Mpc−3 -> h=h0/75
ax_total_snr.errorbar(0.29, 1.42e-4 * h70**3, yerr=[[0.3e-4 * h70**3], [0.3e-4 * h70**3]], capsize=5, color='pink', label='Bazin et al. (2009)', fmt='o') # yr-1 (h70-1 Mpc)-3 [Measured from 117 ccSNe]

# Strolger Et Al 2015 - all rates in  yr−1 Mpc−3 10−4h703
# Ma et al quote table 2 not table 4 
# GOODS + CANDLES + CLASH (table 4)
"""
ax_total_snr.errorbar(0.3, 2.13e-4 * h70**3, yerr=[[0.54e-4 * h70**3], [0.8e-4 * h70**3]], xerr=0.2, capsize=5, color='violet', label='Strolger et al. (2015)', fmt='o')
ax_total_snr.errorbar(0.7, 3.86e-4 * h70**3, yerr=[[0.72e-4 * h70**3], [0.96e-4 * h70**3]], xerr=0.2, capsize=5, color='violet', label='Strolger et al. (2015)', fmt='o')
ax_total_snr.errorbar(1.1, 3.07e-4 * h70**3, yerr=[[0.66e-4 * h70**3], [1.06e-4 * h70**3]], xerr=0.2, capsize=5, color='violet', label='Strolger et al. (2015)', fmt='o')
ax_total_snr.errorbar(1.5, 3.25e-4 * h70**3, yerr=[[1.32e-4 * h70**3], [2.03e-4 * h70**3]], xerr=0.2, capsize=5, color='violet', label='Strolger et al. (2015)', fmt='o')
ax_total_snr.errorbar(1.9, 3.16e-4 * h70**3, yerr=[[1.77e-4 * h70**3], [3.37e-4 * h70**3]], xerr=0.2, capsize=5, color='violet', label='Strolger et al. (2015)', fmt='o')
ax_total_snr.errorbar(2.3, 6.17e-4 * h70**3, yerr=[[3.52e-4 * h70**3], [6.67e-4 * h70**3]], xerr=0.2, capsize=5, color='violet', label='Strolger et al. (2015)', fmt='o')
"""
# CANDLES + CLASH (table 2) as used by Ma and Wang et al (2025)
ax_total_snr.errorbar(0.3, 1.97e-4 * h70**3, yerr=[[0.85e-4 * h70**3], [1.45e-4 * h70**3]], xerr=0.2, capsize=5, color='red', label='Strolger et al. (2015)', fmt='o')
ax_total_snr.errorbar(0.7, 2.68e-4 * h70**3, yerr=[[1.04e-4 * h70**3], [1.54e-4 * h70**3]], xerr=0.2, capsize=5, color='red', fmt='o')
ax_total_snr.errorbar(1.1, 1.70e-4 * h70**3, yerr=[[0.71e-4 * h70**3], [1.19e-4 * h70**3]], xerr=0.2, capsize=5, color='red', fmt='o')
ax_total_snr.errorbar(1.5, 3.25e-4 * h70**3, yerr=[[1.32e-4 * h70**3], [2.03e-4 * h70**3]], xerr=0.2, capsize=5, color='red', fmt='o')
ax_total_snr.errorbar(1.9, 3.16e-4 * h70**3, yerr=[[1.77e-4 * h70**3], [3.37e-4 * h70**3]], xerr=0.2, capsize=5, color='red', fmt='o')
ax_total_snr.errorbar(2.3, 6.17e-4 * h70**3, yerr=[[3.52e-4 * h70**3], [6.67e-4 * h70**3]], xerr=0.2, capsize=5, color='red', fmt='o')

"""
#Melinder et al. (2012)
ax_total_snr.errorbar(0.39, 3.29e-4, yerr=[[1.78e-4], [3.08e-4]], capsize=5, color='pink', label='Melinder et al. (2012)', fmt='o') # using statistical errors
ax_total_snr.errorbar(0.73, 6.40e-4, yerr=[[3.12e-4], [5.30e-4]], capsize=5, color='pink', label='Melinder et al. (2012)', fmt='o')

# Dahlen et al. Mpc−3 10−4 h70^3 -> remove h dependance
ax_total_snr.errorbar(0.39, 3e-4* h70**3, yerr=[[0.94e-4* h70**3], [1.28e-4* h70**3]], capsize=5, color='violet', label='Dahlen et al. (2012)', fmt='o')
ax_total_snr.errorbar(0.73, 7.39e-4* h70**3, yerr=[[1.52e-4* h70**3], [1.86e-4* h70**3]], capsize=5, color='violet', label='Dahlen et al. (2012)', fmt='o')
ax_total_snr.errorbar(1.11, 9.57e-4* h70**3, yerr=[[2.80e-4* h70**3], [3.76e-4* h70**3]], capsize=5, color='violet', label='Dahlen et al. (2012)', fmt='o')
"""

# set legend and axes
ax_total_snr.set_yscale('linear')
ax_total_snr.set_ylim(-1e-5, 12e-4)
#ax_total_snr.set_ylim(1e-5, 10e-4)
ax_total_snr.set_xlim(-0.2, 6)
ax_total_snr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val*1e4:g}'))
plt_labels(fig_total_snr, ax_total_snr, 3, 0.17)

fig_types1.savefig("Data/Images/TNG/final/cosmic/cosmic_type.png", dpi=300)
fig_types2.savefig("Data/Images/TNG/final/cosmic/cosmic_type(sfr).png", dpi=300)
fig_total_sfr.savefig(f"Data/Images/TNG/final/cosmic/cosmic_sfh.png", dpi=300)
fig_total_snr.savefig(f"Data/Images/TNG/final/cosmic/cosmic_snh.png", dpi=300)

plt.close(fig_total_sfr)
plt.close(fig_total_snr)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#region Fractions
def myfunc(x):
  return slope * x + intercept

fig_ratio, ax_ratio = plt_helper(8, 6, "Redshift (z)",  r'Supernova Fraction [%]', logx=False, logy=False, legendspace=0.2)
ratios = []
fracs = []
for idx, item in enumerate(test_dict.items()):
    name = item[0]
    csfrh = item[1][0]
    csnrh = item[1][1]

    #sf_ratio = sum(csfrh/total_sfr)/len(csfrh)
    #sn_ratio = sum(csnrh/total_snr)/len(csnrh)
    #print(f'{name}: \n  SN:{sn_ratio*100:.1f}\n  SF: {sf_ratio*100:.1f}')

    # calculate average across all redshifts
    sf_values = csfrh / total_sfr
    sn_values = csnrh / total_snr
    sf_ratio = np.mean(sf_values)
    sn_ratio = np.mean(sn_values)

    # calculate error in average across all redhisfts
    sf_err = np.std(sf_values, ddof=1) / np.sqrt(len(sf_values))
    sn_err = np.std(sn_values, ddof=1) / np.sqrt(len(sn_values))

    sn_err_1 = stats.sem(sn_values, axis=0, ddof=1)

    print(f'{name}:')
    print(f'  SN: {sn_ratio*100:.2f} ± {sn_err*100:.2f}, {sn_err_1*100:.2f}')
    print(f'  SN With Redshift: {sn_values*100}')
    print(f'  SF: {sf_ratio*100:.2f} ± {sf_err*100:.2f}')
    ratios.append(sn_ratio)
    fracs.append(np.round(sn_values*100,2))

    # plot across redshifts with error bars found from standard error of mean
    # THIS IS WRONG THERE IS NO ERROR DIVIDING BY EACH TYPE (current give 1 point for all)
    #y = csnrh/total_snr
    #std_dev = np.std(csnrh/total_snr, ddof=1)  # ddof=1 gives sample std
    #std_error = std_dev / np.sqrt(len(csnrh/total_snr))
    
    # find line of best fit
    y = csnrh/total_snr * 100
    x = np.linspace(rev_redshifts.min(), rev_redshifts.max(), 300)
    slope, intercept, r, p, std_err = stats.linregress(rev_redshifts, y)
    mymodel = list(map(myfunc, x))

    x_aic, y_aic = aic.apply_aic(rev_redshifts, y)
    ax_ratio.plot(x_aic, y_aic, color=sn_colours[idx], linewidth=1, zorder=30, label=f"{name} AIC")
    ax_ratio.plot(x, y_aic, color='black',  linewidth=2, zorder=20)

    #ax_ratio.plot(x, mymodel, color='black',  linewidth=2, zorder=20)
    #ax_ratio.plot(x, mymodel, color=sn_colours[idx], label=f"{name} lineregress", linewidth=1, zorder=30)
    #ax_ratio.errorbar(rev_redshifts, y, yerr=std_error, color='black', fmt='D', capsize=5, zorder=30)
    ax_ratio.scatter(rev_redshifts, y, label=name, color=sn_colours[idx], marker='D', edgecolors='black', zorder=40)

#plt_labels(fig_ratio, ax_ratio, 4, 0.07)
fig_ratio.tight_layout(rect=[0, 0, 1, 1])
fig_ratio.savefig("Data/Images/TNG/final/cosmic/cosmic_ratio.png", dpi=300)
print('total', sum(ratios))

#for group in zip(*fracs):
    #print(" & ".join(str(x) for x in group))