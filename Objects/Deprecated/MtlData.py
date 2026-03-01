
# Formula to generate arrays
import numpy as np
import random
from Helpers.FormatHelper import CoordinateFormatter

class MtlData:

    def __init__(self, min, max, length):
        self.length = length
        interval = (max-min)/length
        self.metalicityList = np.arange(min, max, interval)

    def generateNormal(self, sigma, mu, x):
        # mean = mu, standard deviation = sigma
        #data = np.random.normal(mu, sigma, size=(1, self.length))
        #return data.tolist()
    
        partOne = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = -(pow((x - mu),2))/(2 * pow(sigma,2))
        partTwo = np.exp(exponent)

        Gaussian = partOne * partTwo
        return Gaussian

    def generateSqaure():
        return 0
    
    def getMetalicity(self, prob, randomChoices):
        # prob is probability ie gaussian data list
        mtlSelection = []
        for _ in range(randomChoices):
            probChoiceId = random.randint(0, len(prob)-1)
            #mtlSelection.append([float(self.metalicityList[probChoiceId]), float(prob[probChoiceId])])
            mtlSelection.append(CoordinateFormatter(float(self.metalicityList[probChoiceId]), float(prob[probChoiceId])))

        return mtlSelection
