"""
Helper to determine AIC Statistics for model selection. 
Once AIC values have been determined, generate a probability based on two AIC reults 
"""

# imports
import numpy as np

# region AIC Helper
class AICHelper():

    def __init__(self):
        pass
    
    def rss(self, actual, predicted):
        # Calculate the Residual Sum of Squares (rss)
        # actual: Actual data points from bpass
        # predicted: Predicted data points from the polynomial

        residuals = actual - predicted
        return np.sum(residuals ** 2)

    def aic(self, k, n, rss):
        # Calculate the Akaike Information Criterion (AIC)
        # k: Number of free parameters in the model
        # n: Number of data points
        # rss: Residual sum of squares of the model
        
        if rss == 0:
            epsilon = 1e-10
            return 2 * k + n * np.log(rss + epsilon)
        return 2 * k + n * np.log(rss)
    
    def probability (self, aic_max, aic_min):
        # Calculate the probability of model 1 over model 2 (should it be chosen)
        # aic_max: value for model 1 (simpler model)
        # aic_min: value for model 2 (complex model
        if aic_max == 0:
            return 0

        exponent = (aic_min - aic_max) / 2
        # large numbers as exponents return infinite 
        if exponent > 709:
            return 1e10
        return np.exp(exponent)
