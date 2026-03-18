"""
Extracts BPASS supernova data using previously generated arrays 
Generates data in age specifc form and returns csv files
Generates polynomial coefficents for each supernova type at given age, imf and single/binary
    - Saves to csv files
From these coefficients the polynomial fits can be plotted
    - Saved to imf specific folder in Data/Images/PolyFits
The plotAllPolynomials function can be used to plot all polynomials for a given imf and single/binary
    - The aim of this was to yield an overall relation between fits however there was no clear trend
Generates a random Supernova rate for a given:
    - SN Type, Age, Metallicity, IMF, single or binary
    - This is completed using normal distributions 
    - a plot is shown for visualisation

Core code duplicated from BPASSPlotter.py
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import scipy.optimize as sco
import scipy.stats as scs
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", message="Data has no positive values, and therefore cannot be log-scaled.")


from Helpers.PlotHelper import PlotHelper
from Helpers.PandasHelper import PandasHelper
from Helpers.AICHelper import AICHelper
from Helpers.StatsHelper import StatsHelper


class BPASSData():
    def __init__(self, imf:str, sinbin:str, age:float, plot:plt):
        self.imf = imf
        self.sinbin = sinbin
        self.age = age
        self.plot = plot

# region BPASS Analysis
class BPASSAnalysis():

    def __init__(self, supernovaArray):
        self.supernovaArray = supernovaArray
        self.basePath = os.getcwd()
        self.agePath = os.path.join(self.basePath, "Data/age_data")
        self.coeffPath = os.path.join(self.basePath, "Data/coeff_data")
        self.imagePath = os.path.join(self.basePath, "Data/Images/BPASS")
        
    def getDiagonals(self, arr):
        # get diagnoal entries of a 2d array 
        # used for the covariance matrix to get errors on coeffs
        if arr.ndim == 1:
            yield arr
        else:
            yield from self.getDiagonals(arr.diagonal())
            yield from self.getDiagonals(np.flip(arr, 0).diagonal())
    

    def generateAgeSpecific(self, imf, sinbin):
        # imf and Data is class with attributes
        # .imfname - (name)
        # .data - (Array of Mtl and Data)
        # --- Mtl and Data is class with attributes 
        # --- .mtl (metallicity)
        # --- .sinbin (single or binary)
        # --- .data (the dataframe as taken from bpass .dat file)

        # Dictionary to hold age-specific DataFrames
        ageDataframes = {}

        for imf_entry in self.supernovaArray:
            if imf_entry.imfname == imf:
                for mtlData in imf_entry.data:
                    if mtlData.sinbin == sinbin:
                        mtl = mtlData.mtl
                        df = mtlData.data
                        #print('mtl', mtl)
                        for _, row in df.iterrows():
                            age = row['Log(age/yrs)']
                            bin_width = row['Length']
                            #print(f' age: {age}, bin width: {bin_width}')
                            if age not in ageDataframes:
                                ageDataframes[age] = []

                            ageDataframes[age].append({
                                'mtl': mtl,
                                'Ia': row['Ia']/ bin_width,
                                'Err_Ia': row['Err_Ia']/ bin_width,
                                'IIP': row['IIP']/ bin_width,
                                'Err_IIP': row['Err_IIP']/ bin_width,
                                'II-other': row['II-other'] / bin_width,
                                'Err_II-other': row['Err_II-other']/ bin_width,
                                'Ib': row['Ib']/ bin_width,
                                'Err_Ib': row['Err_Ib']/ bin_width,
                                'Ic': row['Ic']/ bin_width,
                                'Err_Ic': row['Err_Ic']/ bin_width,
                                'Long-GRB': row['Long-GRB']/ bin_width,
                                'Err_Long-GRB': row['Err_Long-GRB']/ bin_width,
                                'Pair-Instab': row['Pair-Instab']/ bin_width,
                                'Err_Pair-Instab': row['Err_Pair-Instab']/ bin_width,
                                'Low-mass': row['Low-mass']/ bin_width,
                                'Err_Low-mass': row['Err_Low-mass']/ bin_width,
                            })

        # Convert lists to DataFrames and save as CSV
        for age, rows in ageDataframes.items():
            age_df = pd.DataFrame(rows)
            # Move rows to correct order (zem4 to top, then zem5 to top)
            age_df = pd.concat([age_df.iloc[[11],:], age_df.drop(11, axis=0)], axis=0)
            age_df = pd.concat([age_df.iloc[[12],:], age_df.drop(12, axis=0)], axis=0)
            filename = os.path.join(self.agePath + f"/imf{imf}", f"supernova-{sinbin}-imf{imf}.Age{str(round(age,2)).replace('.','_')}yrs.csv")
            age_df.to_csv(filename, index=False)

        print(f"    Saved {len(ageDataframes)} CSV files to '{self.agePath}'")

        return ageDataframes
    
    def generateCoeffs(self, imf, sinbin, age, plot:bool):
        # for a give imf, single or binary and age determine the polynomial coeffs for each supernova type
        # read in the age specific data frame (this data is produced in the function above)
        fileName = f"supernova-{sinbin}-imf{imf}.Age{str(round(age,2)).replace('.','_')}yrs.csv"
        df = pd.read_csv(os.path.join(self.agePath + f"/imf{imf}", fileName), sep=',')
        # Cut off the metallicity and Type Ia columns as we don't need them
        headers = df.columns.values[2:]

        # Array of all metallicities in BPASS as numbers (used for plotting x axis)
        allMtls = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.008, 0.010, 0.020, 0.030, 0.040]
        # keep plot colours conistsent as there is a plt.plot and plt.errorbar
        colours = ['blue', 'red', 'orange', 'pink', 'green', 'purple', 'brown', 'black', 'cyan', 'lime', 'gold', 'navy']

        i = 0
        coeffDataframe = []
        myPlot, ax = plt.subplots(figsize=(8,7))
        for header in headers:
            # Skip error columns
            if 'Err_' not in header:
                i = i + 1
                snRates = np.array(df[header])
                errSnRates = np.array(df['Err_'+header])

                # determine the weights used for polyfit error
                allWeights = []
                for _, row in df.iterrows():
                    if row[header] == 0:
                        allWeights.append(0)
                    else:
                        allWeights.append(1/row[header])

                """
                # Mask to exclude zero or negative rates
                mask = snRates > 0
                # Apply mask to all arrays
                maskedRates = snRates[mask]
                maskedMtls = np.array(allMtls)[mask]
                maskedWeights = np.array(allWeights)[mask]
                maskedErrs = errSnRates[mask]
                """

                mask = snRates > 0                     
                maskedRates = snRates[mask]
                maskedMetals = np.array(allMtls)[mask]
                maskedErrs = errSnRates[mask]
                maskedErrs[maskedErrs == 0] = 1e-20

                # Apply polynomial fit and use AIC statistics to determine order of polynomial
                # Maximum order of magnitude is 4 (quartic)
                prevAIC = 0
                prevCoeffs = [0, 0, 0, 0, 0]
                prevErrCoeffs = [0, 0, 0, 0, 0]
                polyPlot = []
                for order in range(1,5):

                    """
                    Numpy Poly Fit

                    # must have at least order + 1 data points to fit the polynomial
                    orderLimit = order + 1
                    if len(maskedMtls) > orderLimit and len(maskedWeights) > orderLimit and len(maskedRates) > orderLimit:
                        coeffs, matrix = np.polyfit(maskedMtls, maskedRates, order, cov=True, w=maskedWeights)
                        errCoeffs = np.sqrt(list(self.getDiagonals(matrix))[0])
                    else: 
                        coeffs = [0, 0, 0, 0, 0]
                        errCoeffs = [0, 0, 0, 0, 0]

                    #for orders of polynomial less than 4 fill coeffs and errCoeffs with zeros (up to 5 terms)
                    while len(coeffs) < 5:
                        coeffs = np.append(0, coeffs)
                        errCoeffs = np.append(0, errCoeffs)
                    """

                    orderLimit = order + 1
                    if len(maskedRates) > orderLimit:
                        #slope, intercept, r, p, std_err = scs.linregress(np.array(allMtls), np.array(snRates))
                        #p0 = [slope, intercept]
                        p0 = np.polyfit(maskedMetals, maskedRates, order).tolist()
                        while len(p0) <= order:
                            p0.append(0)

                        # Scipy Curve fit 
                        model = lambda x, *params : self.polynomialFunc(order, x, np.array(params))
                        coeffs, matrix = sco.curve_fit(model, maskedMetals, maskedRates, p0, maskedErrs, absolute_sigma=True, nan_policy='omit', method='trf')
                        errCoeffs = np.sqrt(list(self.getDiagonals(matrix))[0])
                    else: 
                        coeffs = [0, 0, 0, 0, 0]
                        errCoeffs = [0, 0, 0, 0, 0]            

                    while len(coeffs) < 5:
                        coeffs = np.append(0, coeffs)
                        errCoeffs = np.append(0, errCoeffs)

                    # determine polynomial values
                    polyList = self.polynomialCalc(coeffs, maskedMetals)
                    xPlot = np.arange(1e-5, 0.040, 0.0001)
                    polyPlot = self.polynomialCalc(coeffs, xPlot)

                    # apply AIC to determine best order
                    aicHelper = AICHelper()
                    rss = aicHelper.rss(maskedRates, polyList)
                    aic = aicHelper.aic(order + 1, len(polyPlot), rss)
                    prob = aicHelper.probability(prevAIC, aic)

                    #print(f"AIC {header} Prev {prevAIC}, Curr {aic}, Probability: {prob}")
                    if prob > 0.95:
                        # reject more complex model (previous model is better)
                        # return the previous order's coeffs
                        coeffs = prevCoeffs
                        errCoeffs = prevErrCoeffs
                        polyList = self.polynomialCalc(coeffs, maskedMetals)
                        polyPlot = self.polynomialCalc(coeffs, xPlot)
                        break
                    # else accept more complex model (current model is better)
                    # ie do nothing and continue to next order
                    prevAIC = aic
                    prevCoeffs = coeffs
                    prevErrCoeffs = errCoeffs

                # Print coefficents is showing plot for comparisons
                if plot:
                    print(f"    {header}")
                    print(f"        Coeffs: {' '.join(str(coeff) for coeff in coeffs)}")

                numDataPoints = len(maskedRates)
                freeParams = np.count_nonzero(np.array(coeffs))
                # Find std deviation of points from fit if valid fir
                if numDataPoints > 0 and freeParams > 0:
                    # residula = actual data - poly fit data
                    res = np.array(maskedRates) - np.array(polyList)
                    std = np.sqrt(np.sum(pow(res,2)) / (numDataPoints - freeParams))
                else:
                    std = 0

                # add data to dataframe which can be saved as a csv
                coeffDataframe.append({
                    'SN': header,
                    'C0': coeffs[0],
                    'C1': coeffs[1],
                    'C2': coeffs[2],
                    'C3': coeffs[3],
                    'C4': coeffs[4],
                    'Err_C0': errCoeffs[0],
                    'Err_C1': errCoeffs[1],
                    'Err_C2': errCoeffs[2],
                    'Err_C3': errCoeffs[3],
                    'Err_C4': errCoeffs[4],
                    'Std': std
                })

                # plotting
                ax.plot(xPlot, polyPlot, label=header, color=colours[i])
                #ax.scatter(allMtls, snRates, color=colours[i])#, yerr=errSnRates, fmt='o', label="_nolegend_", color=colours[i])
                ax.errorbar(allMtls, snRates, yerr=errSnRates, fmt='o', label="_nolegend_", color=colours[i])
                #ax.set_yscale('log')
                myPlot.subplots_adjust(bottom=0.2, right=0.95)
                ax.set_ylabel(r'Event Rate [$\mathrm{yr^{-1}}$]')
                ax.set_xlabel('Metallicity (Z)')
                ax.set_title(f'BPASS Age Bin: {age:.2} log(yrs)') 
        
        handles, labels = ax.get_legend_handles_labels()
        myPlot.legend(handles, labels,loc='lower center',ncol=3, frameon=False)

        # If plot is true open and show plots
        if plot: 
            myPlot.show()
            plt.show()
        
        
        # Save the plot figure
        full_path = os.path.join(self.imagePath + f"/imf{imf}/Coeff", f"plot-{sinbin}-imf{imf}.Age{str(round(age,2)).replace('.','_')}yrs.png")
        myPlot.savefig(full_path, dpi=300)  
        plt.close(myPlot)

        # save to csv 
        coeffdf = pd.DataFrame(coeffDataframe)
        filename = f"coeffs-{sinbin}-imf{imf}.Age{str(round(age,2)).replace('.','_')}yrs.csv"
        filepath = os.path.join(self.coeffPath + f"/imf{imf}", filename)
        coeffdf.to_csv(filepath, index=False)

        #print(f"Saved {filename} CSV file to '{self.coeffPath}'")
        return coeffdf
    
    def generateAllCoeffs(self, imf, sinbin, plot:bool):
        # generate coeffs for all ages for given imf and sinbin
        ageDataframes = self.generateAgeSpecific(imf, sinbin)
        for idx, age in enumerate(ageDataframes.keys()):
            self.generateCoeffs(imf, sinbin, age, plot)
        print(f"    Saved {idx + 1} coefficient CSV files to '{self.coeffPath}'")

    
    def polynomialCalc(self, coeffs, xVals):
    # generate polynomial values for given coeff
    # length of coeffs determines order of polynomial
    # use xVals as array of x values to generate poly values

        polyList = []
        for x in xVals:
            poly = 0
            for idx, coeff in enumerate(coeffs):
                polyTerm = coeff * x ** (len(coeffs) - 1 - idx)
                poly += polyTerm
            polyList.append(poly)
        return polyList
    
    def polynomialFunc(self, order, x, coeffs):
    # generate polynomical function based on order
        poly = 0
        for idx in range(order, 1, -1):
            polyTerm = coeffs[idx] * x ** (order - idx)
            poly += polyTerm

        return poly
    
    def plotAllPolynomials(self, imf, sinbin):
        ageDataframes = self.generateAgeSpecific(imf, sinbin)
        for idx, age in enumerate(ageDataframes.keys()):
            fileName = f"coeffs-{sinbin}-imf{imf}.Age{str(round(age,2)).replace('.','_')}yrs.csv"
            df = pd.read_csv(os.path.join(self.coeffPath, fileName), sep=',')

            df_IIP = df.iloc[0]
            coeffs = [df_IIP['C0'], df_IIP['C1'], df_IIP['C2'], df_IIP['C3'], df_IIP['C4']]
            xPlot = np.arange(1e-5, 0.040, 0.0001)
            polyList = self.polynomialCalc(coeffs, xPlot)

            plt.plot(xPlot, polyList, label = age)
            
        plt.title(f"All poly plots for {sinbin}-imf{imf}")
        #plt.yscale('log')
        plt.ylabel('Log SN Rate')
        plt.xlabel('Z')
        plt.legend()
        plt.show()

        return 0

    def generateSupernovaRate(self, snType, imf, sinbin, z, age, plot):
        """
        ACTUALLY NUMBER NOT RATE!!!!!!
        generate random SN Rate using BPASS data
        
        snType: 0: IIP, 1: II-Other, 2: Ib, 3: Ic, 4: Long-GRB, 5: Pair-Instab, 6: Low-mass
        age: in log(yrs)
        plot: 1 plot the graph all other values do not
        """
        err = 0
        normDataframe = {
            'IMF': f'imf{imf}',
            'Age_Log(yrs)': age,
            'Z': z,
            'ccSNe': None,
            'Mean': 0, 
            'Std': 0,
            'FWHM': 0,
            'ccSNRate': 0,
        }

        stats = StatsHelper(1000)

        # Read in coefficients from data 
        fileName = f"coeffs-{sinbin}-imf{imf}.Age{str(round(age,1)).replace('.','_')}yrs.csv"

        try:
            df = pd.read_csv(os.path.join(self.coeffPath, f"imf{imf}" ,fileName), sep=',')
        except FileNotFoundError:
            err = 1 
            print(f'FAILED: File Not Found, attempted file name - {fileName}')
            return normDataframe

        # Select a type of supernova and corrosponding data
        sn_df = df.iloc[snType]
        coeffs = [sn_df['C0'], sn_df['C1'], sn_df['C2'], sn_df['C3'], sn_df['C4']]
        # std and snRate used to generate noramal dist
        std = sn_df['Std']
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std
        snRate = self.polynomialFunc(len(coeffs)-1, z, coeffs)
        normDataframe['ccSNe'] = sn_df.iloc[0]
        if snRate < 0: 
            #invalid SN rate - normally due to polynomial dropping off at the selected metallicity 
            err = 1
            #print(f'FAILED: Negative SN rate - INPUTS( Z: {z}, Age: {age}, Type: {sn_df.iloc[0]})')
        else:
            # generate normal dist and randomly select a rate
            normal = stats.normalDist(snRate, std)
            randomRate = random.choice(normal)

            # check that the normal distribution has positive values
            if np.array(normal).max() < 0:
                err = 1 
                print(f'FAILED: Normal Distribution contains no positive values - INPUTS( Z: {z}, Age: {age}, Type: {sn_df.iloc[0]})')
                plot = 1
            else:
                # ensure a positive rate is chosen (some distribtions may slightly spread into the negative)
                while randomRate < 0:
                    randomRate = random.choice(normal)

        # Generate a theoretical plot for visulisation
        if plot == 1:
            theoretical_x = np.linspace(min(normal), max(normal), 1000)
            theoretical_y = norm.pdf(theoretical_x, loc=snRate, scale=std)
            yMax = max(theoretical_y)
            sigmaHeight = yMax * 0.6

            # histogram plots real random data
            plt.hist(normal, bins=40, density=True, alpha=0.5, color='firebrick')
            plt.plot(theoretical_x, theoretical_y, color='red', linewidth=2)
            plt.axvline(randomRate, color='red', linestyle='dashed', linewidth=2, label=f'Random value: {randomRate:.2f}')

            # mean line
            plt.arrow(snRate, 0, 0, yMax, color='black', 
                    head_width=10, head_length=0.00012, length_includes_head=True, shape='full')
            plt.arrow(snRate, yMax, 0, -yMax, color='black', 
                    head_width=10, head_length=0.00012, length_includes_head=True, shape='full')
            plt.text(snRate, yMax + 0.0001, f'μ: {snRate:.2f}', ha='center')
            
            # sigma line
            plt.arrow(snRate, sigmaHeight, std, 0, color='black', shape='full', 
                    head_width=0.00005, head_length=15, length_includes_head=True, width = 0.00001)
            plt.arrow(snRate + std,  sigmaHeight, -std, 0, color='black', shape='full', 
                    head_width=0.00005, head_length=15, length_includes_head=True, width = 0.00001)
            plt.text(snRate + std / 2,  sigmaHeight + 0.0001, f'σ: {std:.2f}', ha='center')

            # FWHM line
            plt.axvline(snRate - fwhm/2, color='black', linestyle='dashed', linewidth=1, alpha=0.5)
            plt.axvline(snRate + fwhm/2, color='black', linestyle='dashed', linewidth=1, alpha=0.5)
            plt.arrow(snRate - fwhm/2, 0.0041, fwhm, 0, color='black', shape='full', 
                    head_width=0.00005, head_length=15, length_includes_head=True, width = 0.00001)
            plt.arrow(snRate + fwhm/2, 0.0041, -fwhm, 0, color='black', shape='full', 
                    head_width=0.00005, head_length=15, length_includes_head=True, width = 0.00001)
            plt.text(snRate - 30, 0.0041 + 0.0001, f'FWHM: {fwhm:.2f}', ha='center')
        
            plt.title(f"Normally Distributed Supernova Rates:\n{str(sn_df.iloc[0])} imf{imf} {'binary' if str(sinbin) == 'bin' else 'single'} z:{str(z)} Age:{str(age)}yrs")
            
            plt.xlabel("Supernova Rate")
            plt.ylabel("Density")
            #plt.ylim(0, 0.0050)
            plt.legend()
            plt.show()

        if err == 1:
            return normDataframe
        else:
            # add data to a dictionary
            normDataframe = {
                'IMF': f'imf{imf}',
                'Age_Log(yrs)': age,
                'Z': z,
                'ccSNe': sn_df.iloc[0],
                'Mean': float(snRate), 
                'Std': float(std),
                'FWHM': float(fwhm),
                'ccSNRate': randomRate,
            }
            #print(f'SUCCESS: INPUTS( Z: {z}, Age: {age}, Type: {sn_df.iloc[0]})')
        return normDataframe
    
    def subhaloData(self, path, pbar=None):
        df = pd.read_csv(path)
        result_rows = []
        i = 0
        for idx, row in df.iterrows():
            inputMtl = row['mtl']
            inputAge = row['age_bin']

            # slowing down code for no reason by reassigning or is it minimal?
            # snType: 0: IIP, 1: II-Other, 2: Ib, 3: Ic, 4: Long-GRB, 5: Pair-Instab, 6: Low-mass
            sn_rate_data = self.generateSupernovaRate(0, "_chab100", "bin", inputMtl, inputAge, 0)
            sn_rate_data['x'] = row['x']
            sn_rate_data['y'] = row['y']
            sn_rate_data['Redshift'] = row['redshift']
            sn_rate_data['Mass'] = row['mass_solar']
            sn_rate_data['Age_Myr'] = row['age_Myr']
            sn_rate_data['Halo_ID'] = row['halo_id']
            sn_rate_data['Halo_SFR'] = row['halo_SFR']
            sn_rate_data['Halo_Volume'] = row['halo_volume']
            sn_rate_data['Number_of_Subhalos'] = row['number_of_subhalos']


            result_rows.append(dict(sn_rate_data.items()))
            i += 1
            if pbar is not None:
                pbar.update(1)

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
        
        """
        #supernova_rate_per_solar_mass_per_volume = supernova_rate_per_solar_mass / all_results_df["Halo_Volume"]
        #supernova_rate_per_solar_mass_per_volume = supernova_rate_per_solar_mass * (len(supernova_rate)/box_size)
        #supernova_rate_per_solar_mass_per_volume = supernova_rate_per_solar_mass * (all_results_df['Number_of_Subhalos']/box_size)

        # star formation rate density
        # using same method as above (CHECK)
        # need answer in yr-1 Gpc-3 
        sfrd = sum(all_results_df['Halo_SFR']) * (all_results_df['Number_of_Subhalos'].iloc[0]/box_size)
        print('SFRD', sfrd)

        # calculate total rates for the subhalo
        #total_supernova_number = sum(all_results_df["ccSNRate"])
        #total_supernova_rate_per_solar_mass = sum(supernova_rate_per_solar_mass)
        #total_supernova_rate_per_solar_mass_per_volume = sum(supernova_rate_per_solar_mass_per_volume)
        
        id = all_results_df['Halo_ID'].iloc[0]
        sfr = all_results_df['Halo_SFR'].iloc[0]
        snr = total_supernova_number
        mass = sum(all_results_df["Mass"])
        z = all_results_df['Redshift'].iloc[0]
        snr_new = total_supernova_rate_per_solar_mass
        snr_volume = total_supernova_rate_per_solar_mass_per_volume

        # remove any regions with Supernova Rate (ccSNRate) of zero
        all_results_df = all_results_df[all_results_df["ccSNRate"] != 0]

        #print(f'total supernova rate for subhalo {sn_rate_data['Halo_ID']}: {total_supernova_number}, {total_supernova_rate_per_solar_mass}')

        subhalo_dataframe = {
            'id': id,
            'sfr': sfr,
            'snr': snr,
            'snr_per_mass': snr_new,
            'snr_per_volume': snr_volume,
            'mass':mass,
            'z': z, 
            #'halo_volume': sum(all_results_df["Halo_Volume"]),
            'halo_volume': halo_volume,
            'sfrd': sfrd
        }
        return subhalo_dataframe
        """
