"""
Helper to determine AIC Statistics for model selection. 
Once AIC values have been determined, generate a probability based on two AIC reults 
"""

# imports
import numpy as np
from Helpers.PolyHelper import PolyHelper
from itertools import combinations
import scipy.optimize as sco
poly = PolyHelper()

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
    
    def apply_aic(self, x, y, errs, additional_checks=False):
        # Maximum order of magnitude is 4 (quartic)

        if len(x) == 0 and len(errs) == 0:
            return 0, 0, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

        # set up first aic as linear 
        if len(y) > 2:
            w = 1 / errs
            w = w / w.max()  # scale to [0, 1] — relative weights preserved

            try:
                prev_coeffs, cov = np.polyfit(x, y, 1, w=w, cov=True)
                prev_coeff_err = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                prev_coeffs = np.polyfit(x, y, 1)
                prev_coeff_err = [0, 0, 0, 0, 0]
            x_plot = np.linspace(min(x), max(x), 300)
        else:
            return 0, 0, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

        initial = poly.polynomialCalc(prev_coeffs, x)
        rss = self.rss(y, initial)
        max_aic = self.aic(2, len(y), rss)
        prev_aic = self.aic(2, len(y), rss)
        prev_plot = poly.polynomialCalc(prev_coeffs, x_plot)
        
        # other variables
        selected_plot = prev_plot
        aic_values = []
        aic_values.append(max_aic)
        final_order = 1

        for order in range(2,5):
            if len(y) > order + 1:
                # initial guess with polyfit
                try:
                    coeffs, cov = np.polyfit(x, y, order, w=1/errs, cov=True)
                    coeff_errs = np.sqrt(np.diag(cov))
                except np.linalg.LinAlgError:
                    coeffs = np.polyfit(x, y, order)
                    coeff_errs = [0, 0, 0, 0, 0]
                
                # determine polynomial values
                poly_values = poly.polynomialCalc(coeffs, x)
                poly_plot = poly.polynomialCalc(coeffs, x_plot)

                # apply AIC to determine best order
                rss = self.rss(y, poly_values)
                aic = self.aic(order + 1, len(y), rss)
                prob = self.probability(prev_aic, aic)
                aic_values.append(aic)

                if prob > 0.95:
                    # reject complex model 
                    final_order = order - 1
                    coeffs = prev_coeffs
                    coeff_errs = prev_coeff_err
                    #pass
                    #return x_plot, prev_plot
                else:
                    # accept complex model, move on 
                    final_order = order
                    prev_aic = aic
                    selected_plot = poly_plot
                    prev_coeffs = coeffs
                    prev_coeff_err = coeff_errs
            else:
                return  x_plot, selected_plot, prev_coeffs, prev_coeff_err
            if len(y) > order + 1:
                if additional_checks == True:
                    # check more combinations than just against the linear fit
                    for i, j in combinations(range(len(aic_values)), 2):
                        max_a = aic_values[i]
                        min_a = aic_values[j]
                        prob = self.probability(max_a, min_a)
                        #print(f'Order:{i+1}, {max_a}, Order:{j+1}, {min_a}, Prob:{prob}')  # apply your formula here

                        if j+1 > len(aic_values):
                            if prob < 0.95:
                                # check against higher order
                                prob = self.probability(aic_values[j], aic_values[j+1])
                                if prob < 0.95:
                                    # accept as new again
                                    coeffs = np.polyfit(x, y, j+2)
                                    selected_plot = poly.polynomialCalc(coeffs, x_plot)
                                    final_order = j+2
                                    #print(f'Accepted order - {j+2}')
                                else:
                                    # accept current j as the new order 
                                    coeffs = np.polyfit(x, y, j+1)
                                    selected_plot = poly.polynomialCalc(coeffs, x_plot)
                                    final_order = j+1
                                    #print(f'Accepted order - {j+1}')
                                break
                        else:
                            break
                            

                        # check higher orders against original
                        if (j-i > 1) and (prob < 0.95):
                            # order difference is greater than 1, and probability is good
                            # check the higher order against one lower to make sure it is better
                            prob = self.probability(aic_values[i+1], min_a)
                            if prob < 0.95:
                                # accept this as the new order 
                                coeffs = np.polyfit(x, y, j+1)
                                selected_plot = poly.polynomialCalc(coeffs, x_plot)
                                final_order = j+1
                                #print(f'Accepted order - {j+1}')
                                break

        #print(f'Final Order: {final_order}')
        return x_plot, selected_plot, coeffs, coeff_errs
    
    def apply_aic_simple(self, x, y):
        # Maximum order of magnitude is 4 (quartic)

        # set up first aic as linear 
        if len(y) > 2:
            prev_coeffs = np.polyfit(x, y, 1)
            x_plot = np.linspace(min(x), max(x), 300)
        else:
            return 0, 0, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

        initial = poly.polynomialCalc(prev_coeffs, x)
        rss = self.rss(y, initial)
        max_aic = self.aic(2, len(y), rss)
        prev_aic = self.aic(2, len(y), rss)
        prev_plot = poly.polynomialCalc(prev_coeffs, x_plot)
        
        # other variables
        selected_plot = prev_plot
        aic_values = []
        aic_values.append(max_aic)

        for order in range(2,5):
            if len(y) > order + 1:
                # initial guess with polyfit
                coeffs = np.polyfit(x, y, order)
                
                # determine polynomial values
                poly_values = poly.polynomialCalc(coeffs, x)
                poly_plot = poly.polynomialCalc(coeffs, x_plot)

                # apply AIC to determine best order
                rss = self.rss(y, poly_values)
                aic = self.aic(order + 1, len(y), rss)
                prob = self.probability(prev_aic, aic)
                aic_values.append(aic)

                if prob > 0.95:
                    # reject complex model 
                    coeffs = prev_coeffs

                else:
                    # accept complex model, move on 
                    prev_aic = aic
                    selected_plot = poly_plot
                    prev_coeffs = coeffs
            else:
                return  x_plot, selected_plot, prev_coeffs
            
        return x_plot, selected_plot
