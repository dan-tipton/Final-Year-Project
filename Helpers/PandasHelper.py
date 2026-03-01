
"""
Basic helper to read the BPASS files as they have strange column seperation

"""
# imports
import pandas as pd

# region Pandas Helper
class PandasHelper:
    def __init__(self, path):
        self.path = path

    def readBPASS(self):
        df = pd.read_csv(self.path, sep='\s+', header=None)
        return df