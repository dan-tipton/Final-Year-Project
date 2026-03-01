
"""
Helper class to generate various statistical distributions
"""
# imports
import numpy as np 

# region Stats Helper
class StatsHelper:
    def __init__(self, mySize):
        # mySize represnts the shape of the returned array
        self.mySize = mySize

    def normalDist(self, mean, std):
        normal = np.random.normal(mean, std, self.mySize)
        return normal.tolist()
    
    def poissonDist(self, lam):
        poisson = np.random.poisson(lam, self.mySize)
        return poisson.tolist()
    
    def logNormalDist(self, mean, std):
        logNormal = np.random.lognormal(mean, std, self.mySize)
        return logNormal.tolist()