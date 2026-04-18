
"""
Helper class to generate plots using input data
scatter plots and nornal implemented 
functions return pyplot objects which can then be shown in the main code
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd

# region Plot Helper
class PlotHelper():
    def __init__(self, linestyle, xAxisName, yAxisName):
        self.linestyle = linestyle
        self.xAxisName = xAxisName
        self.yAxisName = yAxisName
        pass

    #colors = ['#FF5733', '#33FF57', '#3357FF', "#FFD012"]

    def makePlot(self, title, x, y1: pd.DataFrame, y2: pd.DataFrame, y3: pd.DataFrame, y4: pd.DataFrame, y5: pd.DataFrame, y6: pd.DataFrame, y7: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(x, y1, label=y1.name, linestyle=self.linestyle, color='#FF5733')
        ax.plot(x, y2, label=y2.name, linestyle=self.linestyle, color='#33FF57')
        ax.plot(x, y3, label=y3.name, linestyle=self.linestyle, color='#3357FF')
        ax.plot(x, y4, label=y4.name, linestyle=self.linestyle, color='#FFD012')
        ax.plot(x, y5, label=y5.name, linestyle=self.linestyle, color="#544051")
        ax.plot(x, y6, label=y6.name, linestyle=self.linestyle, color="#FF4FD6")
        ax.plot(x, y7, label=y7.name, linestyle=self.linestyle, color="#933823")

        ax.set_xlabel(self.xAxisName, fontsize=18)
        ax.set_ylabel(self.yAxisName, fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(title, fontsize=22)

        # Collect legend items from axes
        handles, labels = ax.get_legend_handles_labels()

        # Place legend on the figure instead
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,  fontsize=18)
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        #fig.tight_layout(rect=[0, 0., 1, 1])

        return fig, ax
    
    def makePlotFromArray(self, ax, title, x, y1, y1Name, y2, y2Name, y3, y3Name, y4, y4Name, y5, y5Name, y6, y6Name, y7, y7Name):
        #myPlot = plt

        ax.plot(x, y1, label=y1Name, linestyle=self.linestyle, color='Blue')
        ax.plot(x, y2, label=y2Name, linestyle=self.linestyle, color='Orange')
        ax.plot(x, y3, label=y3Name, linestyle=self.linestyle, color='Green')
        ax.plot(x, y4, label=y4Name, linestyle=self.linestyle, color='Red')
        ax.plot(x, y5, label=y5Name, linestyle=self.linestyle, color='Purple')
        ax.plot(x, y6, label=y6Name, linestyle=self.linestyle, color='Brown')
        ax.plot(x, y7, label=y7Name, linestyle=self.linestyle, color='Pink')
        
        #myPlot.xlabel(self.xAxisName)
        #myPlot.ylabel(self.yAxisName)
        #myPlot.title(title)

        ax.set_xlabel(self.xAxisName)
        ax.set_ylabel(self.yAxisName)
        ax.set_title(title)
        ax.legend()

        return plt
    
    def makeScatter(self, ax, marker, title, x, y1, y1Name, y2, y2Name, y3, y3Name, y4, y4Name, y5, y5Name, y6, y6Name, y7, y7Name):
        #myPlot = plt

        ax.scatter(x, y1, label='_nolegend_', marker=marker, color='Blue')
        ax.scatter(x, y2, label='_nolegend_', marker=marker, color='Orange')
        ax.scatter(x, y3, label='_nolegend_', marker=marker, color='Green')
        ax.scatter(x, y4, label='_nolegend_', marker=marker, color='Red')
        ax.scatter(x, y5, label='_nolegend_', marker=marker, color='Purple')
        ax.scatter(x, y6, label='_nolegend_', marker=marker, color='Brown')
        ax.scatter(x, y7, label='_nolegend_', marker=marker, color='Pink')
        
        #myPlot.xlabel(self.xAxisName)
        #myPlot.ylabel(self.yAxisName)
        #myPlot.title(title)

        ax.set_xlabel(self.xAxisName)
        ax.set_ylabel(self.yAxisName)
        ax.set_title(title)

        return plt