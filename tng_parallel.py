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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import RLock
from scipy import integrate
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm

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

bpass = BPASSDataFormatter()
allSupernovaArray, allIonizingArray, combinedSupernovaIon = bpass.getAllFormattedData()
bpassAnalysis = BPASSAnalysis(allSupernovaArray)
normIMF = IMF(1)
imf = IMF(normIMF.chabrier(0.9)/normIMF.salpeter(0.9))

rates_folder = f"/Users/dan/Code/FYP/Data/TNG/Rates"


import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def build_rates(snap):
    """
    Build rates for a single snapshot folder.
    bpass_obj: instance of bpassAnalysis class
    """
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{snap}/*"
    my_glob = glob.glob(input_path)

    all_rows = []

    max_workers = min(32, os.cpu_count() * 2)  # adjust if needed

    def process_file(file_name):
        """Process one snapshot CSV file"""
        df = pd.read_csv(file_name)
        result_rows = []

        for idx, row in df.iterrows():
            # Extract row-specific parameters
            inputMtl = row['mtl']
            inputAge = row['age_bin']

            # Call generateSupernovaRate (reads coefficient CSV internally)
            sn_rate_data = bpassAnalysis.generateSupernovaRate(0, "_chab100", "bin", inputMtl, inputAge, 0)
            result_rows.append(dict(sn_rate_data.items()))
        
        all_results_df = pd.DataFrame(result_rows)
        all_results_df = all_results_df.reindex(columns=['x', 'y', 'Redshift', 'Halo_ID', 'Halo_SFR', 'Halo_Volume', 'Number_of_Subhalos', 'Mass', 'Age_Myr', 'Age_Log(yrs)', 'Z', 'ccSNRate', 'IMF', 'ccSNe', 'Mean', 'Std', 'FWHM'])
         
        # POST BPASS UPDATE 
        # ccSNRate now gives an event rate in [yr-1]
        pixel_snr = all_results_df["ccSNRate"]
        pixel_snr_solar = pixel_snr / all_results_df["Mass"]

        # sum pixel level to get to halo level rates
        subhalo_snr = sum(pixel_snr)
        subhalo_snr_solar = sum(pixel_snr_solar)

        # halo level densities
        halo_volume = all_results_df["Halo_Volume"].iloc[0]
        subhalo_sfr = all_results_df['Halo_SFR'].iloc[0]
        subhalo_snr_density = subhalo_snr / halo_volume
        subhalo_sfr_density = subhalo_sfr / halo_volume

        # other details 
        mass = sum(all_results_df["Mass"])
        z = all_results_df["Redshift"].iloc[0]
        subhalo_id = all_results_df['Halo_ID'].iloc[0]

        subhalo_dataframe = {
            'id': subhalo_id,
            'sfr': subhalo_sfr,
            'sfrd': subhalo_sfr_density,
            'snr': subhalo_snr,
            'snr_solar': subhalo_snr_solar,
            'snrd': subhalo_snr_density,
            'mass':mass,
            'z': z, 
            'halo_volume': halo_volume,
        }
        return subhalo_dataframe

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in my_glob}

        for f in tqdm(as_completed(futures),
                      total=len(futures),
                      desc=f"Snapshot {snap}",
                      leave=False):
            df_file = f.result()              # already processed with extra pandas logic
            all_rows.extend(df_file.to_dict('records'))

    # Convert to DataFrame and save
    df_snapshot = pd.DataFrame.from_records(all_rows)
    df_snapshot.to_csv(f"snapshot{snap}_rates.csv", index=False)
    return df_snapshot

def main():
    snapshots = snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]  # replace with your actual snapshot list
    results = []

    for snap in snapshots:
        df = build_rates(snap)
        results.append(df)

    return results

if __name__ == "__main__":
    all_results = main()