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
    
    def probability (self, aic1, aic2):
        # Calculate the probability of model 1 over model 2 (should it be chosen)
        # aic1: value for model 1 (generally the lower order polynomial)
        # aic2: value for model 2
        if aic1 == 0:
            return 0

        exponent = (aic2 - aic1) / 2
        # large numbers as exponents return infinite 
        if exponent > 709:
            return 1e10
        return np.exp(exponent)
