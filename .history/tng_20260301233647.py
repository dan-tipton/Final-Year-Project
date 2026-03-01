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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_WORKERS = 4

def build_rates():
    return


results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_subhalo, sid) for sid in subhalo_ids]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())