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

def build_rates(snapNum):
    input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{snapNum}/*"
    my_glob = glob.glob(input_path)

    subhalo_rows = []
    for file_name in my_glob:
        subhalo_data = bpassAnalysis.subhaloData(file_name)
        subhalo_rows.append(dict(subhalo_data.items()))

    print(f"    {len(subhalo_rows)} subhalos with postive rates")
    subhalo_df = pd.DataFrame(subhalo_rows)
    subhalo_df.to_csv(f"/Users/dan/Code/FYP/Data/TNG/Rates/snap{snapNum}_rates.csv")

    return


results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_subhalo, sid) for sid in subhalo_ids]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())