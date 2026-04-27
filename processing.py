import polars as pl
import numpy as np
import pandas as pd
import os
import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
from functools import partial
import cProfile
import pstats
from scipy import stats
import time

from Objects.TNGFormatter import TNG, RateData
tng = TNG()

sn_type = ["IIP", "II-Other", "Ib", "Ic"]

import polars as pl
import numpy as np
from dataclasses import dataclass

@dataclass
class CoeffSet:
    coeffs: np.ndarray      # [C0, C1, C2, C3, C4]
    errors: np.ndarray      # [Err_C0, Err_C1, Err_C2, Err_C3, Err_C4]
    std: float

def load_coeff_file(path) -> dict[str, CoeffSet]:
    all_store = {}
    for path in path.glob("*.csv"):
        df = pl.read_csv(path)
        store = {}
        for row in df.iter_rows(named=True):
            store[row["SN"]] = CoeffSet(
                coeffs=np.array([row["C0"], row["C1"], row["C2"], row["C3"], row["C4"]]),
                errors=np.array([row["Err_C0"], row["Err_C1"], row["Err_C2"], row["Err_C3"], row["Err_C4"]]),
                std=row["Std"],
            )
        all_store[str(path).replace('/Users/dan/Code/FYP/Data/coeff_data/imf_chab100/','')] = store
    return all_store

def process_file(coeff_store, data_path,sn):
    # coeff_store injected via Pool initializer
    df = pl.read_csv(data_path).with_row_index("_idx")
    rates = []
    masses = []
    stds = []

    temp_rows = []
    for row in df.iter_rows(named=True):
        mtl = row['mtl']
        age = row['age_bin']
        coeff_set_name = f"coeffs-bin-imf_chab100.Age{str(round(age,1)).replace('.','_')}yrs.csv"
        all_coeff_set = coeff_store[coeff_set_name]
        coeff_set = all_coeff_set[sn]
        rate_data = tng.randomiseRate(coeff_set.coeffs, coeff_set.std, mtl)
        row['RandomRate'] = rate_data.rate
        temp_rows.append(row)
        rates.append(rate_data.rate)

    df = pl.DataFrame(temp_rows)
    print(max(rates))
    print(sum(rates))
    df.write_csv("output.csv")

    '''
        rates.append(rate_data.rate)
        masses.append(row['mass_solar'])
        stds.append(rate_data.std)
        volume = row['halo_volume']
        sfr = row['halo_SFR']
        redshift = row['redshift']
        halo_id = row['halo_id']
    '''

    #subhalo_data = SubhaloData(rates, masses, stds, volume, sfr, redshift)
    mass = sum(masses)
    pixel_snr_solar = np.array(rates) / np.array(masses)
    snr = sum(rates)
    snr_solar = sum(pixel_snr_solar)

    # halo level densities
    snr_density = snr / volume
    sfr_density = sfr / volume

    # errors - square of the sum of standard deviations squared
    squared_stds = pow(np.array(stds), 2)
    error = np.sqrt(sum(squared_stds))
    squared_stds_solar = pow(np.array(stds)/np.array(masses), 2)
    error_solar = np.sqrt(sum(squared_stds_solar))
    error_vol = error/volume
    subhalo_data = {'id': halo_id,
        'sfr': sfr,
        'sfrd': sfr_density,
        'snr': snr,
        'snr_solar': snr_solar,
        'snrd': snr_density,
        'mass':mass,
        'z': redshift, 
        'halo_volume': volume,
        'snr_err': error,
        'snr_solar_err': error_solar,
        'snrd_err': error_vol
    }
    
    return subhalo_data

my_coeff_store = load_coeff_file(Path("/Users/dan/Code/FYP/Data/coeff_data/imf_chab100"))
rates_folder = f"/Users/dan/Code/FYP/Data/TNG/Rates_V3"
snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]
snapshots = [26]
#snapshots = [2, 10]

# Main loop — snapshots run sequentially, files within each run in parallel
print('start')
start_time = time.time()

split_times = {}
for s in snapshots:
    split_start = time.time()
    position = snapshots.index(s) + 1
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{s}/*"
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_26/Subhalo62862_Redshift2.896.csv"
    my_glob = glob.glob(input_path)
    
    test = []
    all_subhalo_data = {}
    for sn in ["IIP", "II-other", "Ib", "Ic"]:
        for file_name in tqdm(my_glob, desc=f"Snapshot {s} - {sn}", position=position, leave=False):
            all_subhalo_data = process_file(my_coeff_store, file_name, sn)
            test.append(all_subhalo_data)

        subhalo_df = pd.DataFrame(test)
        subhalo_df.to_csv(rates_folder + f"/{sn}/snapshot{s}_rates_demo.csv")
    
    split_times[s] = (time.time() - split_start)

end_time = time.time()
elapsed = end_time-start_time
print(f' Finished in {elapsed:.2f}s/{elapsed/60:.2f}min/{elapsed/(60*60):.2f}hr')
print(f'Snapshot Split:')
for key in split_times.keys():
    split = split_times[key]
    print(f'    {key}: {split:.2f}s/{split/60:.2f}min/{split/(60*60):.2f}hr')
