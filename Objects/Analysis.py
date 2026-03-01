
"""
Script to determine the stellar population for a given IMF within given mass ranges 
Outputs two stellar populations 
Using the number of stars in each population calculate the ratio of type 2 to type 1bc supernova
"""

# imports
import numpy as np 
from scipy.integrate import quad

# region Analysis
class Analysis():

    def __init__(self, limit1, limit2, limit3, zThreshold):
        # limit1: minimum stellar mass limit 
        # limit2: maximum stellar mass for population 1 and minimum for population 2
        # limit3: maximum stella mass for population 2
        # zThreshold: metallicity threshold
        self.limit1 = limit1
        self.limit2 = limit2 
        self.limit3 = limit3
        self.zThreshold = zThreshold

    def stellarPopulation(self, IMF):
        # inetgrate over given limits
        # returns the number of stars in each mass limit
        integral1, err1 = quad(IMF, self.limit1, self.limit2)
        integral2, err2 = quad(IMF, self.limit2, self.limit3)
        return integral1, integral2
    
    """
    # WIP
    def stellarPopulation2(self, massList, IMFList):
        index1a = int([m for m in massList if m >= self.limit1][0])
        index1b = int([m for m in massList if m >= self.limit2][0]) + 1

        index2a = int([m for m in massList if m >= self.limit2][0])
        index2b = int([m for m in massList if m >= self.limit3][0]) + 1

        integral1 = np.trapezoid(IMFList[index1a:index1b], x=massList[index1a:index1b])
        integral2 = np.trapezoid(IMFList[index2a:index2b], x=massList[index2a:index2b])
        return integral1, integral2
    """

    def supernovaeRatio(self, population1, population2, zList):
        # for population 1 (lower mass limit)
        # if metallicity less than threshold then type2 
        # if metallicity greater threshold then type1bc 
        # for population 2 always type1bc as they are defined by the mass limit

        # calculate a ratio for each z in the list
        supernovaData = []
        ratioData = []
        for z in zList:
            nType2 = 0
            nType1bc = 0 

            # if metallicity is high then there will be no type 2 in our population
            if z <= self.zThreshold: 
                nType2 = population1
            elif z > self.zThreshold: 
                nType1bc = population2

            # based on some critea we are yet to define
            dummyParam = 1
            dummyLimit = 1
            if dummyParam == dummyLimit: 
                nType1bc = population2

            supernovaData.append([z, nType2/nType1bc])
            ratioData.append(nType2/nType1bc)   

        return ratioData
