import matplotlib.pyplot as plt

labels1 = ['IIP', 'II-Other', 'Ib', 'Ic']
sizes1 = [43.88, 22.63, 22.68, 10.81]
errors1 = [0.44, 1.99, 1.52, 0.41]

labels2 = ['IIP', 'II-Other', 'Ib', 'Ic']
sizes2 = [47.38, 25.06, 18.60, 9.95]
errors2 = [0.86, 1.69, 1.66, 0.61]

colors = ['#FF5733', '#33FF57', "#4C6CFD", "#FFD012"]
fontsize = 14
# Create figure with 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Helper function for inside labels
def make_autopct(labels, errors):
    def inner(pct):
        idx = make_autopct.idx
        result = f"{labels[idx]}\n{pct:.2f}±{errors[idx]}%"
        make_autopct.idx += 1
        return result
    make_autopct.idx = 0
    return inner   

# First pie chart
make_autopct.idx = 0
axes[0].pie(
    sizes1,
    labels=None,
    colors=colors,
    autopct=make_autopct(labels1, errors1),
    pctdistance=0.6,              # distance of label from center
    startangle=90,
    wedgeprops={'edgecolor':'black', 'linewidth':2},
    textprops={'color':'white', 'weight':'bold', 'fontsize': fontsize}  # <-- white text
)
axes[0].set_title("A)")
axes[0].axis('equal')

# Second pie chart
make_autopct.idx = 0
axes[1].pie(
    sizes2,
    labels=None,
    colors=colors,
    autopct=make_autopct(labels2, errors2),
    pctdistance=0.6,              # distance of label from center
    startangle=90,
    wedgeprops={'edgecolor':'black', 'linewidth':2},
    textprops={'color':'white', 'weight':'bold', 'fontsize': fontsize}  # <-- white text
)
axes[1].set_title("B)")
axes[1].axis('equal')

plt.show()