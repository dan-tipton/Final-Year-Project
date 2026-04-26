
import numpy as np
import random as r 
from Helpers.StatsHelper import StatsHelper
from Helpers.PolyHelper import PolyHelper
from scipy import stats as scipy_stats
from dataclasses import dataclass
stats = StatsHelper(1000)
poly = PolyHelper()


@dataclass
class RateData():
    Z:float
    mean:float
    std:float
    fwhm:float
    rate:float

class TNG():
    def __init__(self):
        pass
    
    def randomiseRate(self, coeffs, std, mtl):
        # std and snRate used to generate noramal dist
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std
        snRate = np.polyval(coeffs, mtl)
        if snRate < 0: 
            #invalid SN rate - normally due to polynomial dropping off at the selected metallicity 
            normal = None
            randomRate = 0
        else:
            # generate normal dist and randomly select a rate
            normal = stats.normalDist(snRate, std)
            randomRate = r.choice(normal)

            # check that the normal distribution has positive values
            if np.array(normal).max() < 0:
                print(f'FAILED: Normal Distribution contains no positive values')
                return 0
            else:
                # ensure a positive rate is chosen (some distribtions may slightly spread into the negative)
                while randomRate < 0:
                    randomRate = r.choice(normal)

        return RateData(mtl, snRate, std, fwhm, randomRate)
