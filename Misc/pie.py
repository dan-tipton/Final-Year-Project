import matplotlib.pyplot as plt

labels = ['IIP [± 0.4%]', 'II-Other [± 2.0%]', 'Ib [± 1.5%]', 'Ic [± 0.4%]']
sizes = [43.9, 22.6, 22.7, 10.8]
colors = ['#FF5733', '#33FF57', '#3357FF', "#FFD012"]

plt.pie(
    sizes,
    labels=labels,              # show labes
    colors=colors,
    autopct='%1.1f%%',          # show percentages
    startangle=90,
    wedgeprops={
        'edgecolor': 'black',   # outline color
        'linewidth': 2          # outline thickness
    }
)

plt.axis('equal')  # keeps it circular
#plt.title("Supernova Fractions Across All Redshifts")
plt.show()