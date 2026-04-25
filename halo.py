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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, root_mean_squared_error
from scipy import stats
import scipy.optimize as sco
from pysr import PySRRegressor
from itertools import combinations

from Helpers.AICHelper import AICHelper
from Helpers.PolyHelper import PolyHelper

aic = AICHelper()
poly = PolyHelper()

base = os.getcwd()
rate_base = os.path.join(base, 'Data/TNG/Rates')
sn_type = ["IIP", "II-Other", "Ib", "Ic"]
snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]
cols_suffix = ["sfr_", "sfrd_", "snr_", "snrs_", "snrd_"]
cols_suffix_2 = ["sfr_", "snr_", "snrs_"]
colors = ['#FF5733', '#33FF57', '#3357FF', "#FFD012"]
colors1 = ["#C13D20", "#1EC93E", "#1D36A7", "#C7A20F"]
colors2 = ["#8C250E", "#128928", "#091C71", "#9B7C00"]
colors3 = ["#E2755C", "#8AF49E", "#8298FA", "#FDE588"]

def linear(x, slope, intercept):
    return slope * x + intercept

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

def myfunc(x, slope, intercept):
  return slope * x + intercept

# plot and calculate avergae
def redshift_bins(snaps, png_name='ratio.png', pcols=1):
    dfs = ratio_calc()

    # Ensure snaps is always a list for processing
    if isinstance(snaps, int):
        snaps = [snaps]

    num = len(snaps)
    prows = math.ceil(num / pcols)
    fig, axes = plt.subplots(prows, pcols, figsize=(8, 7*prows))
    fig_bin, axes_bin = plt.subplots(prows, pcols, figsize=(8, 7*prows))

    axes = axes.flatten() if num > 1 else [axes]
    axes_bin = axes_bin.flatten() if num > 1 else [axes_bin]

    ratio_data = {}

    for i, s in enumerate(snaps):
        df = dfs[s]
        z = df["z"].iloc[0]
        print(z)

        sub_data = {}
        sub_data['Redshift'] = round(z,2)
        for idx, sn in enumerate(sn_type):
            # convert ratio to fraction
            ratio = df[f"snr_{sn}_ratio"] * 100
            mass = df["mass"]
            axes[i].scatter(mass, ratio, label=sn, color=colors[idx], marker='.', zorder=1)

            # calcaulate an average ratio for each sn type at each redshift
            sub_data[sn] = round(np.mean(ratio), 2)
            # # standard error of the mean  -> (standard deviation (sample std, N-1))
            std_dev = np.std(ratio, ddof=1)  # ddof=1 gives sample std
            std_error = std_dev / np.sqrt(len(ratio))
            sub_data[f'{sn}_err'] = round(std_error, 2)

            # set up bins
            n_bins = 50
            m_min = df["mass"].min()
            m_max = df["mass"].max()
            bins = np.logspace(np.log10(m_min), np.log10(m_max), n_bins)
            df["mass_bin"] = pd.cut(df["mass"], bins=bins)

            # average binned values
            binned_avg = df.groupby("mass_bin")[f"snr_{sn}_ratio"].mean() * 100
            bin_centers = [interval.mid for interval in binned_avg.index]
            avg_ratio = binned_avg.values

            # error in bins 
            std_dev_bin = np.std(avg_ratio, ddof=1)  # ddof=1 gives sample std
            std_error_bin = std_dev_bin / np.sqrt(len(avg_ratio))

            # plot
            axes_bin[i].errorbar(bin_centers, avg_ratio, yerr=std_error_bin, color='black', fmt='D', capsize=5, zorder=10)
            axes_bin[i].scatter(bin_centers, avg_ratio, label=sn, color=colors[idx], marker='D', edgecolors='black', zorder=20)

            axes[i].errorbar(bin_centers, avg_ratio, yerr=std_error_bin, color='black', fmt='D', capsize=5, zorder=10)
            axes[i].scatter(bin_centers, avg_ratio, label=sn, color=colors[idx], marker='D', edgecolors='black', zorder=20)

            # linear regress
            z_log = np.log10(bin_centers).reshape(-1, 1)  # or np.log(x) for natural log
            slope, intercept, r, p, std_err = stats.linregress(z_log.ravel(), avg_ratio)
            y_pred = myfunc(z_log, slope, intercept)
            #regress = linear(z_log, slope, intercept)
            #print(f"    {slope:.2f} $\pm$ {std_err:.2f} & ")
            
            axes[i].plot(bin_centers, y_pred, color='black', linewidth=2, zorder=30)  
            axes[i].plot(bin_centers, y_pred, color=colors[idx], linewidth=1, zorder=40)  
            axes_bin[i].plot(bin_centers, y_pred, color='black', linewidth=2, zorder=30) 
            axes_bin[i].plot(bin_centers, y_pred, color=colors[idx], linewidth=1, zorder=40) 

        ratio_data[s] = sub_data
        axes[i].set_xlabel(r'Subhalo Mass [$\mathrm{M_\odot}$]', fontsize=18)
        axes[i].set_ylabel("Supernova Fraction [%]", fontsize=18)
        axes[i].set_xscale('log')
        axes[i].set_title(f"Redshift={z:.2f}", fontsize=18)
        axes[i].tick_params(axis='both', labelsize=18)

        axes_bin[i].set_xlabel(r'Subhalo Mass [$\mathrm{M_\odot}$]', fontsize=18)
        axes_bin[i].set_ylabel("Supernova Fraction [%]", fontsize=18)
        axes_bin[i].set_xscale('log')
        axes_bin[i].set_title(f"Redshift={z:.2f}", fontsize=18)
        axes_bin[i].tick_params(axis='both', labelsize=18)
    
    # legends
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc="lower center", ncol=len(sn_type))
    #fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(f"Data/Images/TNG/ratio/mass/all/{png_name}", dpi=300)
    plt.close(fig)

    handles, labels = axes_bin[0].get_legend_handles_labels()
    #fig_bin.legend(handles, labels, loc="lower center", ncol=len(sn_type))
    #fig_bin.tight_layout(rect=[0, 0.07, 1, 1])
    fig_bin.tight_layout(rect=[0, 0, 1, 1])
    fig_bin.savefig(f"Data/Images/TNG/ratio/mass/reduced/b{png_name}", dpi=300)
    plt.close(fig_bin)

    
    return fig, ratio_data

def animate_plotter(snaps, save_gif=False, gif_name="animation.gif"):
    dfs = ratio_calc()  # your function returning a dict of DataFrames

    fig, ax = plt.subplots(figsize=(8, 6))

    # Precompute data for all snapshots
    scatter_data = []
    binnage_data = []
    for s in snaps:
        df = dfs[s]
        snapshot = []
        snapshot1 = []
        z = df["z"].iloc[0]
        for idx, sn in enumerate(sn_type):
            curr = f"snr_{sn}_ratio"
            mass = df["mass"]
            ratio = df[curr]
            snapshot.append((mass, ratio, sn, colors[idx]))

            # binnage
            n_bins = 50
            m_min = df["mass"].min()
            m_max = df["mass"].max()
            bins = np.logspace(np.log10(m_min), np.log10(m_max), n_bins)
            df["mass_bin"] = pd.cut(df["mass"], bins=bins)
            binned_avg = df.groupby("mass_bin")[f"snr_{sn}_ratio"].mean()
            bin_centers = [interval.mid for interval in binned_avg.index]
            avg_ratio = binned_avg.values

            # Take log of x
            x_log = np.log10(bin_centers).reshape(-1, 1)  # or np.log(x) for natural log
            model = LinearRegression()
            model.fit(x_log, avg_ratio)
            y_pred = model.predict(x_log)

            snapshot1.append((bin_centers, y_pred, sn, colors[idx]))

        binnage_data.append((snapshot1, z))
        scatter_data.append((snapshot, z))

    def update(frame):
        ax.clear()
        snapshot, z = scatter_data[frame]
        snapshot1, z = binnage_data[frame]
        for mass, ratio, sn, color in snapshot:
            ax.scatter(mass, ratio, label=sn, color=color, marker='.')
        
        for binc, av, sn, color in snapshot1:
            ax.plot(binc, av, label=sn, color=color)
        
        ax.set_xscale('log')
        #ax.set_xlabel(r'Subhalo Mass [$\mathrm{M_\odot}$]')
        ax.set_xlabel('Subhalo Mass [$\mathrm{M_\odot}$]')
        ax.set_ylabel("Supernova Fraction [%]")
        ax.set_title(f"Redshift={z:.2f} (Snapshot {frame+1}/{len(snaps)})")
        ax.legend(loc='upper right')

    ani = FuncAnimation(fig, update, frames=len(snaps), interval=1000, repeat=True)

    if save_gif:
        ani.save(gif_name, writer=PillowWriter(fps=2))

    return ani

def run_redshift():
    all_rows = []
    for s in snapshots:
        sfig, data = redshift_bins(s, png_name=f"s{s}_ratio.png")
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'Snapshot'
        df.reset_index(inplace=True)
        all_rows.append(df)

    all_data = pd.concat(all_rows, ignore_index=True)
    all_data.to_csv('Final/Data/all_snapshots_ratio.csv', index=False)

    # Convert to LaTeX
    latex_table = all_data.to_latex(index=False, float_format="%.2f")
    with open("Final/Data/all_snapshots_ratio.tex", "w") as f:
        f.write(latex_table)

    return all_data


model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/", "pow"],
    #unary_operators=["exp", "log"],
    model_selection="best",
    parsimony=5,
    maxsize=10,
)
    
def run_cosmic():
    df = pd.read_csv('Final/Data/all_snapshots_ratio.csv')
    with open("models.txt", "w") as f:
        f.write("Models\n")

    with open("complexity8.txt", "w") as f:
        f.write("Complexity 8\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, sn in enumerate(sn_type):
        print(sn)
        z = df["Redshift"]
        ratio = df[f'{sn}']
        err = df[f'{sn}_err']

        #x = np.array(z).reshape(-1, 1)
        x = np.array(z)
        y = np.array(ratio)
        e = np.array(err)
        x_lin = np.linspace(min(x),max(x),300)
        
        """
        model.fit(x, y)
        best = model.get_best()
        best_func = model.get_best()["lambda_format"]
        y_pred = best_func(x_lin)

        with open("models.txt", "a") as f:
            f.write(str(model))
            f.write(str(best))

        with open("best_models.txt", "a") as f:
            f.write(f"{best["equation"]}\n")
        
        complex_3 = model.equations_.iloc[1]["lambda_format"]
        complex_5 = model.equations_.iloc[2]["lambda_format"]
        complex_8= model.equations_[model.equations_["complexity"] == 8]["lambda_format"]
        """

        #slope, intercept, r, p, std_err = stats.linregress(z, ratio)
        #regress = linear(z, slope, intercept)

        x_aic, y_aic, coeffs, errs = aic.apply_aic(x, y, e)
        print(f'    Coeffs: {coeffs}')
        print(f'    Errors: {errs}')

        ax.errorbar(z, ratio, yerr=err, color='black', fmt='D', capsize=5, zorder=30)
        ax.scatter(z, ratio, label=sn, color=colors[idx], marker='D', edgecolors='black', zorder=40)
        
        #ax.plot(z, regress, color='black', linewidth=2, zorder=10) 
        #ax.plot(z, regress, color=colors[idx], linewidth=1, zorder=20, label=f"{sn} lineregress") 

        ax.plot(x_aic, y_aic, color=colors[idx], linewidth=1, zorder=20, label=f"{sn} aic") 
        ax.plot(x_aic, y_aic, color='black', linewidth=2, zorder=10) 

        # calculate average across all redshifts
        sn_ratio = np.mean(ratio)
        # calculate error in average across all redhisfts
        sn_err = np.std(ratio, ddof=1) / np.sqrt(len(ratio))
        print(f'  Halo: {sn}: {sn_ratio:.2f} ± {sn_err:.2f}')
        
    ax.set_xlabel("Redshift (z)", fontsize=18)
    ax.set_ylabel("Supernova Fraction [%]", fontsize=18)
    #ax.set_ylim(0,100)
    ax.tick_params(axis='both', labelsize=18)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sn_type), fontsize=18)
    fig.tight_layout(rect=[0, 0.3, 1, 1])
    fig.savefig(f"Data/Images/TNG/ratio/mass/cosmic_halo.png", dpi=300)
    return 0

'''
def apply_aic(x, y, errs, additional_checks=False):
    # Maximum order of magnitude is 4 (quartic)
    x_plot = np.linspace(min(x), max(x), 300)

    # set up first aic 
    # linear 
    p0 = np.polyfit(x, y, 1)
    initial = poly.polynomialCalc(p0, x)
    rss = aich.rss(y, initial)
    max_aic = aich.aic(2, len(y), rss)
    prev_aic = aich.aic(2, len(y), rss)
    prev_plot = poly.polynomialCalc(p0, x_plot)

    selected_plot = prev_plot
    aic_values = []
    aic_values.append(max_aic)

    final_order = 1
    for order in range(2,5):
        # initial guess with polyfit
        coeffs = np.polyfit(x, y, order)
        
        # determine polynomial values
        poly_values = poly.polynomialCalc(coeffs, x)
        poly_plot = poly.polynomialCalc(coeffs, x_plot)

        # apply AIC to determine best order
        rss = aich.rss(y, poly_values)
        aic = aich.aic(order + 1, len(y), rss)
        prob = aich.probability(prev_aic, aic)
        aic_values.append(aic)
        
        #print(f"    Current Order {order}\n       Simple:  {max_aic} \n       Complex:  {aic} \n       Prob:  {prob}")

        if prob > 0.95:
            # reject complex model 
            final_order = order - 1
            pass
            #return x_plot, prev_plot
        else:
            # accept complex model, move on 
            final_order = order
            prev_aic = aic
            selected_plot = poly_plot

    if additional_checks == True:
        # check more combinations than just against the linear fit
        for i, j in combinations(range(len(aic_values)), 2):
            max_a = aic_values[i]
            min_a = aic_values[j]
            prob = aich.probability(max_a, min_a)
            #print(f'Order:{i+1}, {max_a}, Order:{j+1}, {min_a}, Prob:{prob}')  # apply your formula here

            if prob < 0.95:
                # check against higher order
                prob = aich.probability(aic_values[j], aic_values[j+1])
                if prob < 0.95:
                    # accept as new again
                    coeffs = np.polyfit(x, y, j+2)
                    selected_plot = poly.polynomialCalc(coeffs, x_plot)
                    final_order = j+2
                    #print(f'Accepted order - {j+2}')
                else:
                    # accept current j as the new order 
                    coeffs = np.polyfit(x, y, j+1)
                    selected_plot = poly.polynomialCalc(coeffs, x_plot)
                    final_order = j+1
                    #print(f'Accepted order - {j+1}')
                break
                

            # check higher orders against original
            if (j-i > 1) and (prob < 0.95):
                # order difference is greater than 1, and probability is good
                # check the higher order against one lower to make sure it is better
                prob = aich.probability(aic_values[i+1], min_a)
                if prob < 0.95:
                    # accept this as the new order 
                    coeffs = np.polyfit(x, y, j+1)
                    selected_plot = poly.polynomialCalc(coeffs, x_plot)
                    final_order = j+1
                    #print(f'Accepted order - {j+1}')
                    break

    print(f'Final Order: {final_order}')
    return x_plot, selected_plot

    """
    probs = []
    for idx, aic in enumerate(aic_values):
        prob = aich.probability(min_aic, aic)
        probs.append(prob)
        #print(f"Order{idx+2}\n   min: {min_aic}\n   max:  {aic} \n  prob:  {prob}")
        print(f"    Current Order {idx+1}\n       Simple:  {min_aic} \n       Complex:  {aic} \n       Prob:  {prob}")

    best_index = np.argmin(probs)
    best_order = orders[best_index] + 1
    if min(probs) < 0.95:
        # probabily less than 0.95 so accept
        print(f'    Chosen order: {best_order}')
        p0 = np.polyfit(x, y, best_order)
        model = lambda x, *params : poly.polynomialFunc(best_order, x, np.array(params))
        coeffs, matrix = sco.curve_fit(model, x, y, p0, errs, absolute_sigma=True, nan_policy='omit', method='trf')
        y_plot = poly.polynomialCalc(coeffs, x_plot)

    else:
        # revert to initial 
        # probabily greater than 0.95 so reject
        print(f'Chosen order: 1')
        y_plot = initial_plot

    return x_plot, y_plot
    """
    
    """
        # make choice
        if prob > 0.95:
            # reject more complex model (previous model is better)
            # return the previous order's coeffs
            coeffs = prevCoeffs
            errCoeffs = prevErrCoeffs
            poly_values = poly.polynomialCalc(coeffs, x)
            poly_plot = poly.polynomialCalc(coeffs, xPlot)
            selectedOrder = selectedOrder - 1
            continue
            #return coeffs, errCoeffs, selectedOrder, poly_plot, xPlot
        # else accept more complex model (current model is better)
        # ie do nothing and continue to next order
        prevAIC = aic
        prevCoeffs = coeffs
        prevErrCoeffs = errCoeffs
    """

    return coeffs, errCoeffs, selectedOrder, poly_plot, xPlot
'''

#run_redshift()
run_cosmic()
#animate_plotter(snapshots, True)


"""
snaps1= [2, 10, 20]
snaps2 = [26, 32, 40]
snaps3 = [50, 57, 66]
snaps4 = [80, 98]

fig1 = plotter(snaps1, png_name='s2_10_20.png')
fig2 = plotter(snaps2, png_name='s26_32_40.png')
fig3 = plotter(snaps3, png_name='s50_57_66.png')
fig4 = plotter(snaps4, png_name='s2_10_20.png')

#new = np.array(snapshots)[::-1]
#animate_plotter(new, save_gif=True)
#animate_seperate(new, save_gif=True)

#plt.show()
"""

