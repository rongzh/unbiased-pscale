import numpy as np
import matplotlib.pyplot as plt

Audat = np.loadtxt(fname='Au-Los-Alamos.md', delimiter='|', skiprows=3)
Audat1 = np.loadtxt(fname='Au-Jones.md', delimiter='|',skiprows=3)
Audat2 = np.loadtxt(fname='Au-Altschuler.md',delimiter='|',skiprows=3)

plt.ion()

plt.plot(Audat[:,0],Audat[:,1],'ko')
plt.plot(Audat1[:,0],Audat1[:,1],'ko',markerfacecolor = 'red')
plt.plot(Audat2[:,0],Audat2[:,1],'ko',markerfacecolor = 'blue')
plt.xlabel('Up')
plt.ylabel('Us')
plt.xlim([0,4])
plt.ylim([2,12])
plt.title('Jamieson1982 Au Plot')
plt.draw()


import numpy as np
import matplotlib.pyplot as plt

Ptdat = np.loadtxt(fname='Pt.md', delimiter='|', skiprows=3)

plt.ion()

plt.plot(Ptdat[:,0],Ptdat[:,1],'ko')
plt.xlabel('Up')
plt.ylabel('Us')
plt.xlim([0,4])
plt.ylim([2,12])
plt.title('Jamieson1982 Pt Plot')
plt.draw()


import numpy as np
import matplotlib.pyplot as plt

MgOdat = np.loadtxt(fname='MgO-Los-Alamos.md', delimiter='|', skiprows=3)

plt.ion()

plt.plot(MgOdat[:,0],MgOdat[:,1],'ko')
plt.xlabel('Up')
plt.ylabel('Us')
plt.xlim([0,4])
plt.ylim([2,12])
plt.title('Jamieson1982 MgO Plot')
plt.draw()