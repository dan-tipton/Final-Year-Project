
# Formula to generate arrays

# Imports
import numpy as np

# region IMF
class IMF:
    def __init__(self, A):
        self.A = A
        self.k0 = 1
        self.k1 = self.k0 * pow(0.08, -0.3 + 1.3)
        self.k2 = self.k1 * pow(0.5, -1.3 + 2.3)
        self.k3 = self.k2 * pow(1, -2.3 + 2.3)

    def salpeter(self, m):
        self.name = "Salpeter"
        # mass in solar masses
        salpeterConstant = 0.03
        powerLaw = pow(m, -2.35)
        salpeter = salpeterConstant * powerLaw
        return salpeter
    
    def kroupa(self, m):
        self.name = "Kroupa"
        #E(m)dm is proportional to m ^ - alpha
        # m in solar masses
        alpha = 2.3
        kroupaConstant = 1
        
        if 1 <= m <= 150:
            kroupaConstant = self.k3
        elif 0.5 <= m < 1:
            kroupaConstant = self.k2
        elif 0.08 <= m < 0.5:
            alpha = 1.3
            kroupaConstant = self.k1
        elif m < 0.08:
            alpha = 0.3
            kroupaConstant = self.k0

        kroupa = kroupaConstant * pow(m, -alpha)
        return kroupa
    
    def chabrier(self, m):
        self.name = "Chabrier"
        # m in solar masses
        part1 = 0.158/(m * np.log(10))
        exponent = -(pow(np.log10(m) - np.log10(0.08),2))/(2 * pow(0.69,2))
        part2 = np.exp(exponent)
        chabrier = 1/self.A * part1 * part2
        if m > 1:
            chabrier = self.salpeter(m)

        return chabrier
    
    def chabrierSystem(self, m):
        self.name = "Chabrier-System"
        # m in solar masses
        part1 = 0.086/(m * np.log(10))
        exponent = -(pow(np.log10(m) - np.log10(0.22),2))/(2 * pow(0.57,2))
        part2 = np.exp(exponent)
        chabrier = 1/self.A * part1 * part2
        if m  > 1:
            chabrier = self.salpeter(m)

        return chabrier
    
    def characteristic(self, m, mc, alpha):
        self.name = "Characteristic"
        part1 = pow(m, alpha)
        exponent = -mc/m
        part2 = np.exp(exponent)
        Chararcteristic = part1 * part2
        
        return Chararcteristic
