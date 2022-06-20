import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import sys

csv = sys.argv[1]
pdf = sys.argv[2]

a = np.loadtxt(csv, delimiter=',', skiprows=1, usecols=[0, 4, 7, 8])
x = a[:, 0]
x1 = a[:, 1]
y1 = a[:, 2]
y2 = a[:, 3]
len = a.shape[0]
x_str = []
y1 = y1 - y2

for i in range(len):
    x_str.append(str(int(x[i])) + "p" + str(int(x1[i])))

plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots()
ax.set_xlabel('N')
ax.set_ylabel('Factor Time (s)')

ax.bar(x_str, y2, 0.35, label='Fac Comm')
ax.bar(x_str, y1, 0.35, bottom=y2, label='Fac Compute')

ax.grid(True, which='both')
ax.legend(fontsize=11)

plt.subplots_adjust(left=0.15,right=0.9,bottom=0.15)
fig.savefig(pdf)
