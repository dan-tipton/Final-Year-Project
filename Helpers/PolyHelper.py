import numpy as np 

class PolyHelper():
    def __init__(self):
        pass
    
    def getDiagonals(self, arr):
        # get diagnoal entries of a 2d array 
        # used for the covariance matrix to get errors on coeffs
        if arr.ndim == 1:
            yield arr
        else:
            yield from self.getDiagonals(arr.diagonal())
            yield from self.getDiagonals(np.flip(arr, 0).diagonal())

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