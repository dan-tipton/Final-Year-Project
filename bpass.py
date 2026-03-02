"""

Master python script controlling all functionality related to BPASS
Objects relating to bpass data are imported, some object may not run when ran alone and require this master script
to run correctly

----------- Mode Functionality Selection -----------
    -1: Automatic Plot Generation for all Age, Photon Rate, Metallicity
    0: Single Plot Generation based on user input    
    1: Comparison Plot Generation based on user input
    2: Coefficient Generation for all IMF, Metallicity, Star Type combinations 
    3: SN Rate selection using normal statistics 
    4: generate supernova rate for subhalo data 
    5: plot snr data from TNG forums
"""

# MODE SELECTION
mode = 4

# region Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import time
import tkinter as tk
import glob

from Helpers.PandasHelper import PandasHelper
from Objects.BPASSPlotter import BPASSPlotter
from Objects.BPASSAnalysis import BPASSAnalysis
from Objects.PopUp import PopUp, SecondaryPopUp
from Objects.BPASSDataFormatter import BPASSDataFormatter
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde


# region Set-Up
start = time.time()
# initialise data formatter and get formatted data
bpass = BPASSDataFormatter()
allSupernovaArray, allIonizingArray, combinedSupernovaIon = bpass.getAllFormattedData()
bpassAnalysis = BPASSAnalysis(allSupernovaArray)

if mode == -1:
    # region Automtaic Plotting
    # initalise two instanes to stop the plots overwriting each other
    bpassPlotter = BPASSPlotter(allSupernovaArray, combinedSupernovaIon)
    bpassPlotter2 = BPASSPlotter(allSupernovaArray, combinedSupernovaIon)


    # loop through each imf and generate all plots
    for imf in bpass.imfNames:
        i = 0
        j = 0
        print(f"Automatic BPASS Plotting for imf: {imf}")

        # single vs binary loop
        for sinbin in bpass.starType:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()

            # loop through each mtl listed above
            for mtl in bpass.metallicities:

                # Age plots
                agePlot = bpassPlotter.agePlots(imf, mtl, sinbin, False)
                agePlot.title(f"Age/SN Rate: {sinbin}-imf{imf}.z{mtl}")
                ageSavePath = os.path.join(bpass.bpassImagePath, f"imf{imf}/Age", f"imf{imf}_{sinbin}_z{mtl}_age.png")
                agePlot.savefig(ageSavePath)
                agePlot.close()
                plt.close()

                # Photon rate plots
                ionPlot = bpassPlotter.ionPlots(imf, mtl, sinbin, False)
                ionPlot.title(f"Photon Rate/SN Rate: {sinbin}-imf{imf}.z{mtl}")
                ionSavePath = os.path.join(bpass.bpassImagePath, f"imf{imf}/Ion", f"imf{imf}_{sinbin}_z{mtl}_ion.png")
                ionPlot.savefig(ionSavePath)
                ionPlot.close()
                plt.close()

                i += 1

            # mtl plots - dont require a mtl input obvs!
            mtlPlot = bpassPlotter2.metalPlots(imf, sinbin, norm=False, dashed=False, ax1=ax1, ax2=ax2)
            mtlPOLYSavePath = os.path.join(bpass.bpassImagePath, f"imf{imf}/Mtl", "POLY",f"imf{imf}_{sinbin}_mtl.png")
            mtlRAWSavePath = os.path.join(bpass.bpassImagePath, f"imf{imf}/Mtl", "RAW",f"imf{imf}_{sinbin}_mtl.png")
            fig1.savefig(mtlPOLYSavePath)
            fig2.savefig(mtlRAWSavePath)
            plt.close(fig1)
            plt.close(fig2) 
            mtlPlot.close()
            j += 1
            
            print(f"    {i} Age and Ion plots saved to {bpass.bpassImagePath}/{imf}")
        print(f"    {j} Metallicity plots saved to {bpass.bpassImagePath}/{imf}/Mtl")


elif mode == 0:
    # region Single Plotting
    myPlotter = BPASSPlotter(allSupernovaArray, combinedSupernovaIon)
    root = tk.Tk()
    root.withdraw()
    myPopUp = PopUp(root)
    myPopUp.first_popup()
    root.mainloop()

    graph, imf, sinbin, mtl, norm = myPopUp.graph, myPopUp.imf, myPopUp.sinbin, myPopUp.mtl, myPopUp.norm
    
    userPlot = plt
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    if graph == 0:
        graphName = "Age/SN Rate"
        userPlot = myPlotter.agePlots(imf, mtl, sinbin, False)
    elif graph == 1:
        graphName = "Metallicity/SN Rate"
        userPlot = myPlotter.metalPlots(imf, sinbin, norm, False, ax1=ax1, ax2=ax2)
    elif graph == 2:
        graphName = "Photon Rate/SN Rate"
        userPlot = myPlotter.ionPlots(imf, mtl, sinbin, False)

    print(f"Current plot being show: \n {graphName} \n  IMF {imf} with {sinbin} stars and z={mtl}")

    userPlot.title(f"{graphName}: {sinbin}-imf{imf}.z{mtl}")
    
    # Close any remaining Tk windows before showing figures.
    try:
        # destroy all child windows first
        for w in root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        # process pending events and then destroy the root
        try:
            root.update()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
    except Exception:
        pass

    userPlot.show()

elif mode == 1:
    # region Comparison plotting
    myPlotter = BPASSPlotter(allSupernovaArray, combinedSupernovaIon)
    root = tk.Tk()
    root.withdraw()
    myPopUp = PopUp(root)
    myPopUp.first_popup()
    root.mainloop()
    #root.quit()

    graph, imf, sinbin, mtl, norm = myPopUp.graph, myPopUp.imf, myPopUp.sinbin, myPopUp.mtl, myPopUp.norm
            
    userPlot = plt
    # Create only the figures/axes needed. Metallicity plots (graph == 1)
    # require two axes (ax1, ax2). For other graph types we only need one.
    fig1 = ax1 = fig2 = ax2 = None
    if graph == 1:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

    graphName = 'default'

    singleBinary = 'single' if sinbin == 'sin' else 'binary'
    if graph == 0:
        graphName = "Age/SN Rate"
        userPlot = myPlotter.agePlots(imf, mtl, sinbin, False)
    elif graph == 1:
        graphName = "Metallicity/SN Rate"
        # metalPlots expects ax1 and ax2 to be valid axes objects
        if ax1 is None or ax2 is None:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        userPlot = myPlotter.metalPlots(imf, sinbin, norm, False, ax1, ax2)
    elif graph == 2:
        graphName = "Photon Rate/SN Rate"
        userPlot = myPlotter.ionPlots(imf, mtl, sinbin, False)

    mySecondaryPopUp = SecondaryPopUp(root, graph, imf, sinbin, mtl, norm)
    mySecondaryPopUp.first_popup()
    root.mainloop()
    #root.quit()

    graph, imf2, sinbin2, mtl2 = mySecondaryPopUp.graph, mySecondaryPopUp.imf, mySecondaryPopUp.sinbin, mySecondaryPopUp.mtl

    singleBinary2 = 'single' if sinbin2 == 'sin' else 'binary'
    if graph == 0:
        graphName = "Age/SN Rate"
        userPlot = myPlotter.agePlots(imf2, mtl2, sinbin2, True)
    elif graph == 1:
        graphName = "Metallicity/SN Rate"
        # ensure axes exist for the secondary metal plot
        if ax1 is None or ax2 is None:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        userPlot = myPlotter.metalPlots(imf2, sinbin2, norm, True, ax1, ax2)
    elif graph == 2:
        graphName = "Photon Rate/SN Rate"
        userPlot = myPlotter.ionPlots(imf2, mtl2, sinbin2, True)

    print(f"Current plot being show: \n {graphName} \n  IMF {imf} with {singleBinary} stars and z={mtl} \n      vs.\n   IMF {imf2} with {singleBinary2} stars and z={mtl2}")
    if graph == 1:
        # For metal plots, userPlot is a pyplot-like object returned from metalPlots
        userPlot.title(f"{graphName}: {sinbin}-imf{imf} (-) vs {sinbin2}-imf{imf2} (--)")
        if ax1 is not None:
            ax1.set_title(f"Poly Fit {sinbin}-imf{imf} vs {sinbin2}-imf{imf2}")
    else:
        userPlot.title(f"{graphName}: {sinbin}-imf{imf}.z{mtl} (-) vs {sinbin2}-imf{imf2}.z{mtl2} (--)")
    
    # Close any remaining Tk windows before showing figures.
    try:
        # destroy all child windows first
        for w in root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        # process pending events and then destroy the root
        try:
            root.update()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
    except Exception:
        pass

    # show the plot 
    userPlot.show()

# region Coefficient Generation
elif mode == 2:
    for imf in bpass.imfNames:
        print(f"BPASS Data: imf{imf}, binary")
        bpassAnalysis.generateAllCoeffs(imf, 'bin', plot=False)
    #ceoffdf = myDataExtractor.generateCoeffs('_chab100', 'bin', 7.1, plot=True)

    #myDataExtractor.plotAllPolynomials('_chab100', 'bin')

elif mode == 2.5:
    bpassAnalysis.generateCoeffs('_chab100', 'bin', 7.8, True)
    #bpassAnalysis.generateCoeffs('170_100', 'bin', 7.4, True)

#region SN Rate selection 
elif mode == 3: 
    """
    given an set metalliclity and age get an SN rate that has some randomness applied 
    randomness is based on the standard deviation of the bpass data around their polynomial firs 
    normal distribution is used where std deviation is the the sigma and the peak is the SN rate at 
    the given metallicity and age
    due to lack of correlation between ages, the age selection is within the given bins

    # AGE: 6 - 10 in steps of 0.1
    # METALLICITY: 0 - 0.04
    # snType - 0: IIP, 1: II-Other, 2: Ib, 3: Ic, 4: Long-GRB, 5: Pair-Instab, 6: Low-mass
    """

    dummyPath = "/Users/dan/Code/FYP/Data/TNG/Snapshot_Demo/DummyRun3.csv"
    #dummyPath = "/Users/dan/Code/FYP/Data/TNG/Input/Subhalo41538_Redshift0.01.csv"
    dumb_df = pd.read_csv(dummyPath)

    result_rows = []
    for idx, row in dumb_df.iterrows():
        inputMtl = row['mtl']
        inputAge = row['age_bin']

        #inputMtl = 0.02
        #inputAge = 7.4
        sn_rate_data = bpassAnalysis.generateSupernovaRate(1, "_chab100", "bin", inputMtl, inputAge, 1)
        sn_rate_data['x'] = row['x']
        sn_rate_data['y'] = row['y']
        sn_rate_data['Redshift'] = row['redshift']
        sn_rate_data['Mass'] = row['mass_solar']
        sn_rate_data['Age_Myr'] = row['age_Myr']
        sn_rate_data['Halo_ID'] = row['halo_id']

        result_rows.append(dict(sn_rate_data.items()))

    all_results_df = pd.DataFrame(result_rows)
    all_results_df = all_results_df.reindex(columns=['x', 'y', 'Redshift', 'Mass', 'Age_Myr', 'Age_Log(yrs)', 'Z', 'ccSNRate', 'IMF', 'ccSNe', 'Mean', 'Std', 'FWHM'])
    # remove any regions with Supernova Rate (ccSNRate) of zero
    all_results_df = all_results_df[all_results_df["ccSNRate"] != 0]

    all_results_df.to_csv("/Users/dan/Code/FYP/Data/TNG/Output/out.csv")
    
    total_supernova_rate = sum(all_results_df["ccSNRate"])
    print(f'total supernova rate for subhalo: {total_supernova_rate}')

#region SN Rate selection Using function
elif mode == 4: 
    """
    Same as above just build around multiple csv inputs
    """
    
    build = False
    snapNums = [2, 20, 40, 66, 98]
    #snapNums = [40]
    colours = ['cyan', 'blue', 'orange', 'magenta', 'red', 'yellow', 'brown']

    snap_snr_per_mass = []
    snap_snr_per_volume = []
    redshifts = []
    all_snrd = []

    # plot 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('SFR Vs SNR [per year and solar mass]')
    ax1.set_xlabel('SFR Mo yr^-1')
    ax1.set_ylabel('SNR Supernova yr^-1 Mo^-1')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.set_title('SFR Vs SNR [per year, solar mass and volume]')
    ax2.set_xlabel('SFR Mo yr^-1')
    ax2.set_ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    ax2.set_xscale('log')
    ax2.set_yscale('log')


    for idx, snapNum in enumerate(snapNums): 
        if build == True:
            print(f"---- SNAPSHOT {snapNum} DATA ----")
            input_path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_{snapNum}/*"
            my_glob = glob.glob(input_path)

            print(len(my_glob), "subhalos to analyse")

            subhalo_rows = []
            for file_name in my_glob:
                subhalo_data = bpassAnalysis.subhaloData(file_name)
                subhalo_rows.append(dict(subhalo_data.items()))

            print(f"    {len(subhalo_rows)} subhalos with postive rates")
            subhalo_df = pd.DataFrame(subhalo_rows)
            subhalo_df.to_csv(f"/Users/dan/Code/FYP/Data/TNG/Rates/snap{snapNum}_rates.csv")

        # read rate files
        snr_rates_path = f"/Users/dan/Code/FYP/Data/TNG/Rates/snap{snapNum}_rates.csv"
        subhalo_df = pd.read_csv(snr_rates_path)

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
        all_snrd.append(total_snrd)

        print(f"{snapNum}: z: {redshift}, snr: {total_snr}, snrd: {total_snrd}")

        ax1.scatter(subhalo_df['sfr'], subhalo_df["snr"], marker='.', color=colours[idx], label=f'z={redshift}')
        ax2.scatter(subhalo_df['sfr'], subhalo_df["snrd"], marker='.', color=colours[idx], label=f'z={redshift}')

        """
        total_snr_per_mass = sum(snr_rates_df['snr_per_mass'])
        total_snr_per_volume = sum(snr_rates_df['snr_per_volume'])

        # add to arrays of total snr for snapshot 
        snap_snr_per_mass.append(total_snr_per_mass)
        snap_snr_per_volume.append(total_snr_per_volume)

        # plot 
        ax1.scatter(snr_rates_df['sfr'], snr_rates_df['snr_per_mass'], marker='.', color=colours[idx], label=f'z={redshift}')
        ax2.scatter(snr_rates_df['sfr'], snr_rates_df['snr_per_volume'], marker='.', color=colours[idx], label=f'z={redshift}')

        ax4.scatter(snr_rates_df['z'], snr_rates_df['snr_per_volume'], marker='.', color=colours[idx], label=f'z={redshift}')
        """

    fig2, ax3 = plt.subplots()
    #ax3.scatter(redshifts, snap_snr_per_volume)
    ax3.scatter(redshifts, all_snrd)
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    ax3.set_title(f'Supernova Rate over redshift')
    ax3.set_yscale('log')
    #ax3.set_xscale('log')
    ax1.legend()
    ax2.legend()

    plt.show()

    """
    dummyPath = f"/Users/dan/Code/FYP/Data/TNG/Old/Snap98_trial.csv"
    dumb_df = pd.read_csv(dummyPath)

    dummyPath1 = f"/Users/dan/Code/FYP/Data/TNG/Old/Snap2_trial.csv"
    dumb_df1 = pd.read_csv(dummyPath1)
    
    dummyPath2 = f"/Users/dan/Code/FYP/Data/TNG/Old/Snap40_trial.csv"
    dumb_df2 = pd.read_csv(dummyPath2)

    total_snr_0 = sum(dumb_df['snr_volume'])
    total_snr_12 = sum(dumb_df1['snr_volume'])
    total_snr_1_5 = sum(dumb_df2['snr_volume'])
    print(total_snr_0)
    print(total_snr_12)
    print(total_snr_1_5)

    x = [dumb_df['z'].iloc[0], dumb_df2['z'].iloc[0], dumb_df1['z'].iloc[0]]
    y = [total_snr_0, total_snr_1_5, total_snr_12]

    plt.scatter(x,y)
    plt.xlabel('Redshift ')
    plt.ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    plt.title(f'Supernova Rate over redshift')
    plt.show()

    plt.scatter(dumb_df['sfr'], dumb_df['snr_new'], marker='.', label=f'z={round(dumb_df['z'].iloc[0],3)}')
    plt.scatter(dumb_df1['sfr'], dumb_df1['snr_new'], marker='.', color='red', label=f'z={round(dumb_df1['z'].iloc[0],3)}')
    plt.scatter(dumb_df2['sfr'], dumb_df2['snr_new'], marker='.', color='purple', label=f'z={round(dumb_df2['z'].iloc[0],3)}')
    plt.xlabel('SFR Mo yr^-1')
    plt.ylabel('SNR Supernova yr^-1 Mo^-1')
    plt.title(f'Star Formation Rate vs Supernova Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.scatter(dumb_df['sfr'], dumb_df['snr_volume'], marker='.', label=f'z={round(dumb_df['z'].iloc[0],3)}')
    plt.scatter(dumb_df1['sfr'], dumb_df1['snr_volume'], marker='.', color='red', label=f'z={round(dumb_df1['z'].iloc[0],3)}')
    plt.scatter(dumb_df2['sfr'], dumb_df2['snr_volume'], marker='.', color='purple', label=f'z={round(dumb_df2['z'].iloc[0],3)}')
    plt.xlabel('SFR Mo yr^-1')
    plt.ylabel('SNR Supernova yr^-1 Mo^-1 Gpc^-3')
    plt.title(f'Star Formation Rate vs Supernova Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    """


elif mode == 5:
    
    snapNums = [2, 40, 98]
    colours = ['cyan', 'blue', 'magenta', 'pink', 'red', 'orange', 'yellow', 'green']
    sums = []

    h = 0.6774

    for idx, snapNum in enumerate(snapNums): 
        snr_rates_path = f"/Users/dan/Code/FYP/Data/TNG/Rates/snap{snapNum}_rates.csv"
        snr_rates_df = pd.read_csv(snr_rates_path)
        z = snr_rates_df['z'].iloc[0]

        sfr_per_volume = snr_rates_df['sfr']/snr_rates_df['halo_volume']
        av_sfr = sum(sfr_per_volume)/len(sfr_per_volume)

        box_size_length = 75000 * 1e-6 / (h * (1+z))
        box_size = pow(box_size_length,3)

        sfrd = snr_rates_df['sfr'] * (len(snr_rates_df['sfr'])/box_size)
        av_sfrd = sum(sfrd)/len(sfrd)
        
        plt.scatter(z, av_sfrd, marker = 'x', color ='red')

    # madau & dickinsion 2014
    zs = []
    for zi in np.arange(0, 8, 0.2):
        zs.append(zi)
        sfrd_theory = 0.015 * pow((1 + zi),2.7)/(1 + pow(((1 + zi)/2.9), 5.6)) * 1e9
        print(sfrd_theory) 
        plt.scatter(zi, sfrd_theory, color ='blue', marker='.')

    plt.show()


elif mode == 7:
    # plot tng rates given by Dylan in the QA padlet 

    path = f"/Users/dan/Code/FYP/Data/TNG/Snapshot_Demo/tngrates.csv"
    df = pd.read_csv(path, delimiter=' ', header=None)

    plt.scatter(df[0], df[2], marker='.')
    plt.show()

else: 
    pass

end = time.time()
print(f"\nElapsed time: {end - start:.2f} seconds")