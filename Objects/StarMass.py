
# Generate an Array of star masses 
# Solar Mass = 1.989 * 10 ** 30 kg.

# Imports
import numpy as np
from .AstroData import AstroData

# region Star Mass
class StarMass: 
    def __init__(self, nStars):
        self.nStars = nStars
        self.minMass = 0.01
        self.maxMass = 150
        self.massRegion = 1e6 # 1 million solar masses

    def generateListSolarMasses(self):
        length = int(self.nStars - round(self.nStars/2))
        list1 = np.linspace(self.minMass, 1, num=round(self.nStars/2)).tolist()
        list2 = np.linspace(1, self.maxMass, num=length).tolist()
        starList = list1 + list2
        return starList
    
    def generateList(self):
        solarMass = AstroData.solarMass
        length = int(self.nStars - round(self.nStars/2))
        list1 = np.linspace(self.minMass * solarMass, 1 * solarMass, num=round(self.nStars/2)).tolist()
        list2 = np.linspace(1 * solarMass, self.maxMass * solarMass, num=length).tolist()
        starList = list1 + list2
        return starList
