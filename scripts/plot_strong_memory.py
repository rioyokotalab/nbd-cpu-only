import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import sys

csv = sys.argv[1]
pdf = sys.argv[2]

a = np.loadtxt(csv, delimiter=',', skiprows=1, usecols=[4, 12])
x = a[:, 0]
y = a[:, 1]
len = a.shape[0]

xref = np.array([1, 32])
yref = np.array([y[0], y[0] / 32])

plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots()
ax.set_xlabel('Number Processes')
ax.set_ylabel('Matrix mem (GiB)')

ax.plot(x, y, '^', label = "OUR CODE MPI", linestyle="-", color='blue')
ax.plot(xref, yref, label = "Ideal Strong Scale", linestyle="--", color='black')

ax.axis([1, 32, 0.01, 1000])
ax.loglog(basex=2,basey=10)

for axis in [ax.xaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)

ax.grid(True, which='both')
ax.legend(fontsize=11)

plt.subplots_adjust(left=0.15,right=0.9,bottom=0.15)
fig.savefig(pdf)
