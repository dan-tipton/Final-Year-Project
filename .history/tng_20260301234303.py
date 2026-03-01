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

MAX_WORKERS = 4

bpass = BPASSDataFormatter()
allSupernovaArray, allIonizingArray, combinedSupernovaIon = bpass.getAllFormattedData()
bpassAnalysis = BPASSAnalysis(allSupernovaArray)

def build_rates(snap):
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{snap}/*"
    my_glob = glob.glob(input_path)

    subhalo_rows = []
    for file_name in my_glob:
        subhalo_data = bpassAnalysis.subhaloData(file_name)
        subhalo_rows.append(dict(subhalo_data.items()))

    print(f"    {len(subhalo_rows)} subhalos with postive rates")
    subhalo_df = pd.DataFrame(subhalo_rows)
    subhalo_df.to_csv(f"/Users/dan/Code/FYP/Data/TNG/Rates/snap{snap}_rates.csv")

    return subhalo_df

snapshots = [2, 20, 40, 66, 98]

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(build_rates, snap) for snap in snapshots]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())