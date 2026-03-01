
"""
Various universal classes to help with formatting data
These classes are accessiable by all sections
"""

class CustomTuple: 
    def __init__(self, a, b):
        self.a = a 
        self.b = b

class CoordinateFormatter: 
    def __init__(self, x, y):
        self.x = x 
        self.y = y

class SNeRatio:
    def __init__(self, z, ratio):
        self.z = z
        self.ratio = ratio

class ImfAndSNe:
    def __init__(self, imf:str, ccSNe):
        self.imf = imf
        self.ccSNe = ccSNe
    
    def __iter__(self):
        return iter(self.imf + self.ccSNe)
