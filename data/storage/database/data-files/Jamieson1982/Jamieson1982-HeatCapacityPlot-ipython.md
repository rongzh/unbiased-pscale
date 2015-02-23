import numpy as np
import matplotlib.pyplot as plt

Audat = np.loadtxt(fname='Jamieson1982-HeatCapacityPlot.md', delimiter='|', skiprows=3)

plt.ion()

plt.plot(Audat[:,0],Audat[:,1],'ko')
plt.xlabel('Temperature[kK]')
plt.ylabel('Heat Capacity[J/mg-kK]')
plt.title('Jamieson(1982) Au data - Heat Capacity')
plt.draw()