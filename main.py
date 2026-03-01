
# Enter values and plot data
# region Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from functools import partial
import os
import pandas as pd
import glob

from Objects.AstroData import AstroData
from Objects.StarMass import StarMass
from Objects.IMF import IMF
from Objects.Analysis import Analysis

from Helpers.StatsHelper import StatsHelper
from Helpers.FormatHelper import SNeRatio, ImfAndSNe

# region Set-Up
# initialise class instances
myStarMass = StarMass(10000)
normIMF = IMF(1)
myIMF = IMF(normIMF.chabrier(0.9)/normIMF.salpeter(0.9))
myStats = StatsHelper(10)
myAnalysis = Analysis(8, 25, 80, 0.7)

# intialise dictionary using keys 
keys = ['salpeter', 'kroupa', 'chabrier', 'chabrierSystem', 'char01', 'char1', 'char10', 'char100', 'charAlpha135', 'charAlpha335']
imfDict = {key: [] for key in keys}

# region Integrals
# use partial to give parameters without specifying mass
functions = [
    myIMF.salpeter, myIMF.kroupa, myIMF.chabrier, myIMF.chabrierSystem, 
    partial(myIMF.characteristic, mc=0.1, alpha=-2.35), partial(myIMF.characteristic, mc=1, alpha=-2.35), 
    partial(myIMF.characteristic, mc=10, alpha=-2.35), partial(myIMF.characteristic, mc=100, alpha=-2.35), 
    partial(myIMF.characteristic,mc=1, alpha=-1.35), partial(myIMF.characteristic, mc=1, alpha=-3.35)
    ]

# inetgrate over mass range
results = {}
norms = {}
for idx, func in enumerate(functions):
    integral, error = quad(func, myStarMass.minMass, myStarMass.maxMass)
    results[keys[idx]] = integral

    # for a given mass of a star forming region get normalisation constant
    # normalisation constant such that the integral over the mass range is equal to mass of star forming region
    norms[keys[idx]] = myStarMass.massRegion / integral


# region IMF
# solar mass list and normalisation constant
massList = myStarMass.generateListSolarMasses()
#normA = myIMF.chabrier(0.9, 1)/myIMF.salpeter(0.9)

for idx, func in enumerate(functions):
    for mass in massList:
        imfDict[keys[idx]].append(norms[keys[idx]] * func(mass))

"""
# Old code for getting plot data 

for mass in massList:
    # different 
    imfDict['salpeter'].append(norms['salpeter'] * myIMF.salpeter(mass))
    imfDict['kroupa'].append(norms['kroupa'] * myIMF.kroupa(mass))
    imfDict['chabrier'].append(norms['chabrier'] * myIMF.chabrier(mass))
    imfDict['chabrierSystem'].append(norms['chabrierSystem'] * myIMF.chabrierSystem(mass))
    
    # constant alpha (𝜉) with variable masses
    imfDict['char01'].append(norms['char01'] * myIMF.characteristic(mass, 0.1, -2.35))
    imfDict['char1'].append(norms['char1'] * myIMF.characteristic(mass, 1, -2.35))
    imfDict['char10'].append(norms['char10'] * myIMF.characteristic(mass, 10, -2.35))
    imfDict['char100'].append(norms['char100'] * myIMF.characteristic(mass, 100, -2.35))

    # M☉ = 1 and alpha (𝜉) changes
    imfDict['charAlpha135'].append(norms['charAlpha135'] * myIMF.characteristic(mass, 1, -1.35))
    imfDict['charAlpha335'].append(norms['charAlpha335'] * myIMF.characteristic(mass, 1, -3.35))
"""

"""
#metalicity data against probability

normalData = []
for metalicity in myStats.metalicityList:
    normalData.append(myStats.GenerateNormal(0.1, 0.6, metalicity))

zCoordValues = myStats.GetMetalicity(normalData, 100)
ccSNList = []
for zCoord in zCoordValues:
    type2 = 0
    type1bc = 0
    z = zCoord.x
    if z < 0.7:
        type2 = lower 
    elif z > 0.7:
        type1bc = lower

    type1bc = type1bc + upper
    ccSNList.append(type2/type1bc)

print(ccSNList)
"""

# region Z Data
# generate some data
def getZData(input):
    match input:
        case 1: 
            return list(myStats.normalDist(0.7, 0.1))
        case 2:
            return list(myStats.poissonDist(0.7))
        case 3:
            return list(myStats.logNormalDist(0.7, 0.1))
        case _:
            return list()
    
zData = getZData(1)
    
# loop through each metallicity and return supernova ratio
mtlDataframe = {key: [] for key in keys}
for idx, imf in enumerate(functions):
    pop1, pop2 = myAnalysis.stellarPopulation(imf)
    ratioData = myAnalysis.supernovaeRatio(pop1, pop2, zData)
    mtlDataframe[keys[idx]].append(ImfAndSNe(keys[idx], SNeRatio(zData, ratioData)))

# generate plots of metallicity against supernova ratio
for key, snDataset in mtlDataframe.items():
    for dataset in snDataset:
        # sort data by z so plot line connects correctly
        combined = sorted(zip(dataset.ccSNe.z, dataset.ccSNe.ratio), key=lambda pair: pair[0])
        x_sorted, y_sorted = zip(*combined)

        # plot data
        plt.plot(x_sorted, y_sorted, '-o')
        plt.xlabel("Metallicity (Z)")
        plt.ylabel("Core-Collapse Supernovae Ratio (Type II / Type Ibc)") 
        plt.title(f"ccSNe Ratio vs Z for IMF: {dataset.imf}")

        # Save the plot figure
        full_path = os.path.join(os.getcwd() + "/Data/Images/IMFs", f"{dataset.imf}-ccSNeRatio_Metallicity.png")
        plt.savefig(full_path)
        plt.close()  
    #plt.show()



# region Plotting
"""
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(massList, salpeterList, label='Salpeter')
ax1.plot(massList, kroupaList, label='Kroupa')
ax1.plot(massList, chabrierList, label='Chabrier')
ax1.plot(massList, chabrierSystemList, label='Chabrier System')
ax1.plot(massList, charList01, label='m*c = 0.1M☉', marker=None, linestyle='--')
ax1.plot(massList, charList1, label='m*c = 1M☉', marker=None, linestyle='--')
ax1.plot(massList, charList10, label='m*c = 10M☉', marker=None, linestyle='--')
ax1.plot(massList, charList100, label='m*c = 100M☉',marker=None, linestyle='--')
ax1.plot(massList, charListAlpha135, label='Alpha = -1.35', marker=None, linestyle='dotted')
ax1.plot(massList, charListAlpha335, label='Alpha = -3.35', marker=None, linestyle='dotted')

ax1.set_title('IMF')
ax1.set_xlabel('Log (Mass [M☉])')
ax1.set_ylabel('Log (ξ(m))')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(10**-10,10**4)

ax2.plot(myStats.metalicityList, normalData)
ax2.set_title('Metalicity')
ax2.set_xlabel('Fe/H [M☉]')
ax2.set_ylabel('Probability')

ax1.legend()
plt.show()

"""

# region Massive Stars
df = pd.DataFrame(imfDict[keys[3]])
df['mass'] = massList

# filter dataframe for masses above 8 solar masses
df_supernovaLimit = df[df['mass'] > 8]
dm = df_supernovaLimit["mass"].diff().iloc[1]   # constant bin width
massive_stars_norm = sum(df_supernovaLimit[0] * dm)
print('number of massive stars per 1e6', massive_stars_norm)

massive_star_per_solarmass_formed = massive_stars_norm / 1e6
print('number of massive stars per solar mass formed', massive_star_per_solarmass_formed)

input_path = "/Users/dan/Code/FYP/Data/TNG/Snapshot_98/*"
my_glob = glob.glob(input_path)

subhalo_rows = []
i = 0

# loop through halos
for file_name in my_glob:
    df_sub = pd.read_csv(file_name)

    # get the number of massive stars in each popultaion in each halo 
    massive_stars_sub_region = massive_star_per_solarmass_formed * df_sub['mass_solar']
    total_massive_stars_sub = sum(massive_stars_sub_region)
    total_mass_in_sub = sum(df_sub['mass_solar'])
    print(f'Total Mass: {total_mass_in_sub} (solar mass), with {total_massive_stars_sub} massive stars')
    i += 1
    if i > 10:
        break

    # for each region find if there are any massive stars, 
    # if so find their metallicity and age
    # from that use the BPASS data to find a supernova rate 

    # wont all regions have some massive stars ]
    # no different to what was done previously?????


# region Plot
for key, values in imfDict.items():
    label = key
    linestyle = 'solid'
    if 'char' in key:
        label = f'm*c = {key}M☉'
        linestyle = '--'
    if 'Alpha' in key:
        label = f'Alpha = -{key}'
        linestyle = 'dotted'
    plt.plot(massList, values, label=key, marker=None, linestyle=linestyle)

plt.xlabel("Log (Mass [M☉])")
plt.ylabel("Log (ξ(m)Δm)")
plt.title(f"Initial Mass Functions (IMF), normalised to {myStarMass.massRegion:.1e} solar masses")
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-2,10**9)
plt.legend()
#plt.show()

