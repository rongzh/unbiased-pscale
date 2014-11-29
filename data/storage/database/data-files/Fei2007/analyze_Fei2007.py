import numpy as np
import matplotlib.pyplot as plt

Audat = np.loadtxt(fname='Au-Fig-1.md', delimiter='|', skiprows=3)
NaCldat = np.loadtxt(fname='NaCl-B2-Fig-4.md', delimiter='|', skiprows=3)

plt.ion()

plt.plot(Audat[:,1],Audat[:,2],'ko')
plt.xlabel('Pressure [GPa]')
plt.ylabel('Volume [Ang^3]')
plt.title('Fei(2007) Au data')
plt.draw()

plt.clf()
cmap = plt.get_cmap('bwr')
plt.scatter(NaCldat[:,1],NaCldat[:,2],50,NaCldat[:,0],'o',cmap=cmap,label='Fei2007-NaCl')
plt.colorbar(ticks=range(300,1001,100))
plt.clim([300, 1000])
plt.xlim([0,150])
plt.ylim([18,44])
legend = plt.legend(loc='upper right')
