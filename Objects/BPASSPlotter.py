"""
Metal Plots 
--- Sums the age bins for a given imf and sinbin
--- metalicity is on the x
--- Plots two graphs 
--- One graph is poly fit and other graph is the raw data 


Age Plots 
--- Plots one graph for given metallicity imf and sinbin
--- graph of age on the x and supernova rate on y
--- One graph is poly fit and other graph is the raw data 
"""

from Helpers.PlotHelper import PlotHelper
import matplotlib.pyplot as plt
import numpy as np

#region BPASS Plotter
class BPASSPlotter():

    def __init__(self, supernovaArray, combinedSupernovaIonArray):
        self.supernovaArray = supernovaArray
        self.combinedSupernovaIonArray = combinedSupernovaIonArray
        
    def rearrange_list(self, lst):
        if len(lst) < 2:
            return lst  # Not enough elements to rearrange

        second_to_last = lst[-2]
        last = lst[-1]
        rest = lst[:-2]

        return [last, second_to_last] + rest


    def metalPlots(self, imf, sinbin, norm: bool , dashed: bool, ax1, ax2):
        returnPlot = plt
        singleBinary = 'single' if sinbin == 'sin' else 'binary'
        for term in self.supernovaArray:
            if term.imfname == imf:
                xArray = []
                typeIIArray = []
                typeIbArray = []
                typeIcArray = []
                typeIIOtherArray = []
                typeLongGRBArray = []
                typePairArray = []
                typeLowArray = []

                for data in term.data:
                    currdf = 0
                    if data.sinbin == sinbin:

                        currdf = data.data

                        xArray.append(data.mtl)
                        typeIIArray.append(currdf['IIP'].sum())
                        typeIbArray.append(currdf['Ib'].sum())
                        typeIcArray.append(currdf['Ic'].sum())
                        typeIIOtherArray.append(currdf['II-other'].sum())
                        typeLongGRBArray.append(currdf['Long-GRB'].sum())
                        typePairArray.append(currdf['Pair-Instab'].sum())
                        typeLowArray.append(currdf['Low-mass'].sum())


                ### plot here
                #    print(len(typeIbArray))
                myPlot = PlotHelper("-", "Z", "SN Rate")
                polyPlot = PlotHelper("-", "Z", "Poly SN Rate")
                if dashed: 
                    myPlot = PlotHelper("--", "Z", "SN Rate")
                    polyPlot = PlotHelper("--", "Z", "Poly SN Rate")
                xArray = self.rearrange_list(xArray)
                typeIIArray = self.rearrange_list(typeIIArray)
                typeIbArray = self.rearrange_list(typeIbArray)
                typeIcArray = self.rearrange_list(typeIcArray)
                typeIIOtherArray = self.rearrange_list(typeIIOtherArray)
                typeLongGRBArray = self.rearrange_list(typeLongGRBArray)
                typePairArray = self.rearrange_list(typePairArray)
                typeLowArray = self.rearrange_list(typeLowArray)

                if norm:
                    typeIbArray = (np.array(typeIbArray)/np.array(typeIIArray)).tolist()
                    typeIcArray =  (np.array(typeIcArray)/np.array(typeIIArray)).tolist()
                    typeIIOtherArray =  (np.array(typeIIOtherArray)/np.array(typeIIArray)).tolist()
                    typeLongGRBArray =  (np.array(typeLongGRBArray)/np.array(typeIIArray)).tolist()
                    typePairArray =  (np.array(typePairArray)/np.array(typeIIArray)).tolist()
                    typeLowArray =  (np.array(typeLowArray)/np.array(typeIIArray)).tolist()

                    typeIIArray =  (np.array(typeIIArray)/np.array(typeIIArray)).tolist()

                xArrayNum = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.008, 0.010, 0.020, 0.030, 0.040]
                coeffTypeII = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typeIIArray, dtype=float), 3)
                coeffTypeIb = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typeIbArray, dtype=float), 3)
                coeffTypeIc = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typeIcArray, dtype=float), 3)
                coeffTypeIIOtherArray = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typeIIOtherArray, dtype=float), 3)
                coeffLongGRBArray = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typeLongGRBArray, dtype=float), 3)
                coeffPairArray = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typePairArray, dtype=float), 3)
                coeffLowArray = np.polyfit(np.array(xArrayNum, dtype=float), np.array(typeLowArray, dtype=float), 3)

                """
                def poly(coefficients, x):
                    return (coefficients[0] * x ** 6  + 
                            coefficients[1] * x ** 5 + 
                            coefficients[2] * x ** 4 + 
                            coefficients[3] * x ** 3 + 
                            coefficients[4] * x ** 2 + 
                            coefficients[5] * x + 
                            coefficients[6] 
                            )
                """

                def poly(coefficients, x):
                    return (coefficients[0] * x ** 3  + 
                            coefficients[1] * x ** 2 + 
                            coefficients[2] * x + 
                            coefficients[3]
                            )
                
                yPolyTypeII = []
                yPolyTypeIb = []
                yPolyTypeIc = []
                yPolyTypeIIOther = []
                yPolyTypeLongGRB = []
                yPolyTypePairArray = []
                yPolyTypeLowArray = []
                for x in xArrayNum: 
                    yPolyTypeII.append(poly(coeffTypeII, x))
                    yPolyTypeIb.append(poly(coeffTypeIb, x))
                    yPolyTypeIc.append(poly(coeffTypeIc, x))
                    yPolyTypeIIOther.append(poly(coeffTypeIIOtherArray, x))
                    yPolyTypeLongGRB.append(poly(coeffLongGRBArray, x))
                    yPolyTypePairArray.append(poly(coeffPairArray, x))
                    yPolyTypeLowArray.append(poly(coeffLowArray, x))
                
                linestyle = '-'
                marker = 'x'
                if dashed:
                    linestyle = '--'
                    marker = '.'

                """
                ax1.plot(xArrayNum, yPolyTypeII, linestyle=linestyle, label='type IIP', color='blue')
                ax1.plot(xArrayNum, yPolyTypeIb, linestyle=linestyle, label='type Ib', color='red')
                ax1.plot(xArrayNum, yPolyTypeIc, linestyle=linestyle, label='type Ic', color='green')
                ax1.plot(xArrayNum, yPolyTypeIIOther, linestyle=linestyle, label='type IIP', color='blue')
                ax1.plot(xArrayNum, yPolyTypeLongGRB, linestyle=linestyle, label='type Ib', color='red')
                ax1.plot(xArrayNum, yPolyTypePairArray, linestyle=linestyle, label='type Ic', color='green')
                ax1.plot(xArrayNum, yPolyTypeLowArray, linestyle=linestyle, label='type Ic', color='green')
                """

                polyPlot.makePlotFromArray(ax1, 'Poly Fit: ' + singleBinary + ' IMF' + term.imfname, xArrayNum, 
                                    yPolyTypeII, "type IIP", yPolyTypeIb, "type Ib", yPolyTypeIc, "type Ic",
                                    yPolyTypeIIOther, "type II Other", yPolyTypeLongGRB, 'Long GRB', 
                                    yPolyTypePairArray, 'Pair', yPolyTypeLowArray, 'Low Mass')
                
                polyPlot.makeScatter(ax1, marker, 'Poly Fit: ' + singleBinary + ' IMF' + term.imfname, xArrayNum, 
                    yPolyTypeII, "type IIP", yPolyTypeIb, "type Ib", yPolyTypeIc, "type Ic",
                    yPolyTypeIIOther, "type II Other", yPolyTypeLongGRB, 'Long GRB', 
                    yPolyTypePairArray, 'Pair', yPolyTypeLowArray, 'Low Mass')
                
                """
                ax1.scatter(xArrayNum, typeIIArray, marker=marker, color='blue')
                ax1.scatter(xArrayNum, typeIbArray, marker=marker, color='red')
                ax1.scatter(xArrayNum, typeIcArray, marker=marker, color='green')
                ax1.scatter(xArrayNum, typeIIArray, marker=marker, color='blue')
                ax1.scatter(xArrayNum, typeIbArray, marker=marker, color='red')
                ax1.scatter(xArrayNum, typeIcArray, marker=marker, color='green')
                """
                
                ax1.set_xlabel('Z')
                ax1.set_ylabel('SN Rate from poly')
                ax1.set_title(f'Poly fit for {singleBinary + ' IMF' + term.imfname}')
                ax1.set_yscale('log')
                ax1.legend()

                metalPlot = myPlot.makePlotFromArray(ax2, 'Metal: ' + singleBinary + ' IMF' + term.imfname, xArray, 
                                                   typeIIArray, "type IIP", typeIbArray, "type Ib", typeIcArray, "type Ic",
                                                   typeIIOtherArray, "type II Other", typeLongGRBArray, 'Long GRB', 
                                                   typePairArray, 'Pair', typeLowArray, 'Low Mass')
                
                #metalPlot.show()
                metalPlot.yscale('log')
                returnPlot = metalPlot
        return returnPlot
    
    def agePlots(self, imf, mtl, sinbin, dashed: bool):
        singleBinary = 'single' if sinbin == 'sin' else 'binary'
        returnPlot = plt
        for term in self.supernovaArray:
            if term.imfname == imf:
                for subterm in term.data:
                    currdf = subterm.data
                    if subterm.sinbin == sinbin and subterm.mtl == mtl:
                        x = currdf['Log(age/yrs)']
                        myPlot = PlotHelper("-", "Age", "SN Rate")
                        if dashed: 
                            myPlot = PlotHelper("--", "Age", "SN Rate")
                        agePlot = myPlot.makePlot(
                            'Age: ' + singleBinary + ' IMF' + term.imfname + '_z' + subterm.mtl,
                            x, 
                            currdf['IIP'],
                            currdf['II-other'],
                            currdf['Ib'],
                            currdf['Ic'],
                            currdf['Long-GRB'],
                            currdf['Pair-Instab'],
                            currdf['Low-mass']
                            )
                        agePlot.yscale('log')
                        #agePlot.show()
                        returnPlot = agePlot
        return returnPlot
    
    def ionPlots(self, imf, mtl, sinbin, dashed: bool):
        returnPlot = plt
        singleBinary = 'single' if sinbin == 'sin' else 'binary'
        for data in self.combinedSupernovaIonArray:
            if data.sinbin == sinbin and data.imf == imf and data.mtl == mtl:
                df = data.data
                myPlot = PlotHelper("-", "Photon rate", "SN Rate")
                if dashed: 
                    myPlot = PlotHelper("--", "Photon rate", "SN Rate")
                ionPlot = myPlot.makePlot(
                        'Ion: ' + singleBinary + 'IMF' + data.imf + '_z' + data.mtl,
                        df['Log(N Ion/s)'], 
                        df['IIP'],
                        df['II-other'],
                        df['Ib'],
                        df['Ic'],
                        df['Long-GRB'],
                        df['Pair-Instab'],
                        df['Low-mass']
                        )
                
                #ionPlot.show()
                returnPlot = ionPlot
        return returnPlot