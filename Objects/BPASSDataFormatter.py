"""
New addition to take the following functions out of the master file bpass.py

getSupernovaData( input imf )
    - read the dat file into a pandas dataframe
    - extracts data and formats using custom classes
getIonizingData( input imf )
    - read the dat file into a pandas dataframe
    - extracts data and formats using custom classes

getAllFormattedData()
    - calls both functions above and returns both dataframes

Functions are similar and could be cleaned in the future
"""

# region Imports
import pandas as pd
import os
from Helpers.PandasHelper import PandasHelper


# region Format Classes
class ImfAndData():
    def __init__(self, imfname, data):
        self.imfname = imfname
        self.data = data

class MtlAndData():
    def __init__(self, mtl, sinbin, data):
        self.mtl = mtl
        self.data = data
        self.sinbin = sinbin

class ImfAndMtlAndData():
    def __init__(self, imf, mtl, sinbin, data):
        self.imf = imf
        self.mtl = mtl
        self.data = data
        self.sinbin = sinbin

#region Data Formatter
class BPASSDataFormatter():
    def __init__(self):
        self.basePath = os.getcwd()
        self.dataPath = os.path.join(self.basePath, "Data/bpass_v2.2.1")
        self.bpassImagePath = os.path.join(self.basePath, "Data/Images/BPASS")

        self.baseFolder = "imf"
        self.baseSupernova = "supernova-"
        self.baseIonizing = 'ionizing-'
        self.endFile = ".dat"

        self.starType = ["sin","bin"]
        self.imfNames = ['_chab100','_chab300','100_100','100_300','135_100','135_300','135all_100','170_100','170_300']
        self.metallicities = ["001","002","003","004","006","008","010","014","020","030","040","em4","em5"] 

    #region Get Supernova Data
    def getSupernovaData(self, imf):
        myArray = []
        # path to the folder containing chosen imf
        imfPath = os.path.join(self.dataPath, self.baseFolder + imf)

        # sin vs bin
        for sinbin in self.starType:

            # loop through mtl list
            for mtl in self.metallicities:
                # file name
                fileName = self.baseSupernova + sinbin + "-imf" + imf + '.z' + mtl + self.endFile

                # full path to the file
                supernovaPath = os.path.join(imfPath, fileName)

                # intialise class with path 
                myPandas = PandasHelper(supernovaPath)
                rawData = myPandas.readBPASS()

                df = rawData.rename(columns=
                                {
                                    0: 'Log(age/yrs)',
                                    1: 'Ia',
                                    2: 'IIP',
                                    3: 'II-other',
                                    4: 'Ib', 
                                    5: 'Ic', 
                                    6: 'Long-GRB',
                                    7: 'Pair-Instab',
                                    8: 'Low-mass', 
                                    9: 'Err_Ia',
                                    10: 'Err_IIP',
                                    11: 'Err_II-other',
                                    12: 'Err_Ib', 
                                    13: 'Err_Ic', 
                                    14: 'Err_Long-GRB',
                                    15: 'Err_Pair-Instab',
                                    16: 'Err_Low-mass', 
                                    17: 'Length' 
                                }) 
                
                myArray.append(MtlAndData(mtl, sinbin, df))
        return myArray

    #region Get Ionizing Data
    def getIonizingData(self, imf):
        myArray = []
        # path to the folder containing chosen imf
        imfPath = os.path.join(self.dataPath, self.baseFolder + imf)

        # sin vs bin
        for sinbin in self.starType:

            # loop through mtl list
            for mtl in self.metallicities:
                # file name
                fileName = self.baseIonizing + sinbin + "-imf" + imf + '.z' + mtl + self.endFile

                # full path to the file
                ionizingPathPath = os.path.join(imfPath, fileName)

                # intialise class with path 
                myPandas = PandasHelper(ionizingPathPath)
                rawData = myPandas.readBPASS()

                df = rawData.rename(columns=
                                {
                                    0: 'Log(age/yrs)',
                                    1: 'Log(N Ion/s)',
                                    2: 'Log(L_HAlpha)',
                                    3: 'Log(L_FUV)',
                                    4: 'Log(L_NUV)', 
                                }) 
                
                myArray.append(MtlAndData(mtl, sinbin, df))
        return myArray

    # region return formatted data
    def getAllFormattedData(self):
        # Array of dataframes with corrosponding imf names
        allSupernovaArray = [] 
        allIonizingArray = []
        combinedSupernovaIon = []

        # loop throught all the imfs and get data 
        for imf in self.imfNames:
            supernovaData = self.getSupernovaData(imf)
            ionizingData = self.getIonizingData(imf)
            
            for i in range(len(supernovaData)):
                tempSnDf = supernovaData[i].data
                tempIonDf = ionizingData[i].data
                mtl = supernovaData[i].mtl
                sinbin = supernovaData[i].sinbin

                combinedSupernovaIon.append(ImfAndMtlAndData(imf, mtl, sinbin, pd.merge(tempSnDf,tempIonDf, on='Log(age/yrs)')))


            allSupernovaArray.append(ImfAndData(imf, supernovaData))
            allIonizingArray.append(ImfAndData(imf, ionizingData))

        return allSupernovaArray, allIonizingArray, combinedSupernovaIon
