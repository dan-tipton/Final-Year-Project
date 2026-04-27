import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

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


set_0 = f"/Users/dan/Code/FYP/Data/TNG/Rates"
set_1 = f"/Users/dan/Code/FYP/Data/TNG/Rates_V1"
set_2 = f"/Users/dan/Code/FYP/Data/TNG/Rates_V2"
set_3 = f"/Users/dan/Code/FYP/Data/TNG/Rates_V3"
colours = ['#FF5733', '#33FF57', '#3357FF', "#FFD012", "#B53DFF", "#B53DFF"]

snapshots = [2, 10, 20, 26, 32, 40, 50, 57, 66, 80, 98]
sets = [set_0, set_1, set_2, set_3]

h = 0.6774
box_size_length = 75 / h
box_size = pow(box_size_length,3)

fig, ax = plt.subplots(figsize=(8,7))
ax1 = ax.twinx()

scaled_snrd = []
for i, s in enumerate(sets):
    snrd_box = []
    sfrd_box = []
    for idx, sp in enumerate(snapshots):
        # read rate files
        rates_file = s + '/' + 'IIP' + f"/snapshot{sp}_rates.csv"
        subhalo_df = pd.read_csv(rates_file)

        total_snr = sum(subhalo_df["snr"])
        total_snrd = total_snr / box_size
        snrd_box.append(total_snrd)

        total_sfr = sum(subhalo_df["sfr"])
        total_sfrd = total_sfr / box_size
        sfrd_box.append(total_sfrd)
    
    print(f"{s.replace('/Users/dan/Code/FYP/Data/TNG/','')}")
    sfrd_all, redshifts = calculated_sfrd()
    rev_redshifts = np.array(redshifts)[::-1] 
    formatted_redshift = [f"{x:.4g}" for x in rev_redshifts]
    print(f"    Redshifts: {formatted_redshift}")

    rev_snrd = np.array(snrd_box)[::-1]
    formatted_raw = [f"{x:.4g}" for x in rev_snrd]
    print(f"    RAW: {formatted_raw}")
    rev_sfrd = np.array(sfrd_box)[::-1] 
    rev_sfrd_all= np.array(sfrd_all)[::-1] 
    sfrd_scaling = rev_snrd/rev_sfrd_all

    snrd_scaled = rev_snrd * sfrd_scaling 
    #scaled_snrd.append(snrd_scaled)

    formatted_scaling = [f"{x:.4g}" for x in sfrd_scaling]
    formatted_scaled = [f"{x:.4g}" for x in snrd_scaled]
    print(f"    Scaling: {formatted_scaling}")
    print(f"    SCALED: {formatted_scaled}")

    if i == 0 or i == 1:
        ax.plot(rev_redshifts, snrd_scaled, marker='D', color=colours[i], label=s.replace('/Users/dan/Code/FYP/Data/TNG/',''))
    else:
        ax1.plot(rev_redshifts, snrd_scaled, color=colours[i], marker='D', label=s.replace('/Users/dan/Code/FYP/Data/TNG/',''))

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax1.get_legend_handles_labels()

ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
ax.set_ylabel('0 and 1')
ax1.set_ylabel('2 and 3')

fig.show()
plt.show()


# FOR IIP
# data at redshifts 4, 2.9, 2.1 are way too high 
# corrosponds to snapshots 20, 26, 32
# V2 and V3 near identical differnce between formatters is not the issue?