import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import sys

csv = sys.argv[1]
pdf = sys.argv[2]

a = np.loadtxt(csv, delimiter=',', skiprows=1, usecols=[0, 12])
x = a[:, 0]
y = a[:, 1]
len = a.shape[0]

k = (y[len - 1] - y[0]) / (x[len - 1] - x[0])
b = y[len - 1] - k * x[len - 1]

xref = np.array([1024, 65536])
yref = k * xref

plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots()
ax.set_xlabel('N')
ax.set_ylabel('Matrix mem (GiB)')

ax.plot(x, y, '^', label = "OUR CODE serial", linestyle="-", color='blue')
ax.plot(xref, yref, label = "Ideal Linear", linestyle="--", color='black')

ax.axis([1024, 65536, 0.01, 1000])
ax.loglog(basex=2,basey=10)

for axis in [ax.xaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)

ax.grid(True, which='both')
ax.legend(fontsize=11)

plt.subplots_adjust(left=0.15,right=0.9,bottom=0.15)
fig.savefig(pdf)
