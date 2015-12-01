#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import collections
from eos_mod import MGD_PowerLaw,debye_fun

def set_dic(a):
    param_d['V0'] = a[0]
    param_d['K0'] = a[1]
    param_d['K0p'] = a[2]
    param_d['theta0'] = np.exp(a[3])#
    param_d['gamma0'] = np.exp(a[4])
    param_d['q'] = a[5]


def set_const():
    param_d['const']['Natom'] = 4
    param_d['const']['kB'] =  8.6173324e-5 #eV per K
    param_d['const']['P_factor'] = 160.217657#GPa in 1 eV/Ang^3
    param_d['const']['R'] = 8.314462/1.6021764e-19 # eV/K per mol
    # 1eV/ang^3 = 160.217657GPa, 1eV = 1.6021764e-19Joules, 1Ang3 = e-30m^3
    param_d['const']['C_DP'] = 3*param_d['const']['R']#Dulong-Petit limit for Cv

#read data from file
dat = np.loadtxt(fname='UpUs-Pt.md', delimiter='|', skiprows=3)
print dat
Up = dat[:,0]
Us = dat[:,1]
plt.plot(Up,Us,'rx')
plt.xlabel('Up[km/s]')
plt.ylabel('Us[km/s]')
plt.show()
#print Up,Us

#set Pt density
rho1original = 21.472 #unit grams/cm3
atompermol = 6.022*(10**23) #at/mol
unitcell = 4 #at/unitcell
v_unitcell = 60.38*10**(-24) #unit cm3
rho1 = rho1original/10**24 #grams/ang^3
#compute pho2 based on conservation of Mass
rho2 = rho1*Us/(Us-Up) #unit grams/Ang^3
print "rho2:",rho2

#Atmospheric pressure is
p1 = 101325*10**(-9) #unit GPa
p2 = rho1*Us*Up*10**(24) + p1 #unit GPa
#edit units here: km and m
print "p2: ",p2

#let the initial internal energy E1 to be 0.
m_cell = 195*4/atompermol # (g/unitcell)
f_conv_E = m_cell/160.217657 # (g/cell)*(eV/(GPa*Ang^3))
E2 = 0.5*(p1+p2) * f_conv_E *(1/rho1-1/rho2)#unit eV/unitcell
print "E2: ", E2

param_d = collections.OrderedDict()
param_d['const'] = collections.OrderedDict()
set_const()
V_a = 195*param_d['const']['Natom']/atompermol/rho2 #unit Ang^3/unitcell
print "V_a" , V_a
# 1eV/ang^3 = 160.217657GPa, 1eV = 1.6021764e-19Joules, 1Ang3 = e-30m^3
fei_report = [60.38, 277,5.08,np.log(230),np.log(2.72),0.5]
set_dic(fei_report)

#compute Vinet energy
def energy_vinet( V_a, param_d ):
    V0 = param_d['V0']
    K0 = param_d['K0']
    K0p = param_d['K0p']
    P_factor = param_d['const']['P_factor']

    x = (V_a/V0)**(1.0/3)
    eta = 3.0*(K0p- 1)/2.0

    energy_a = V0*K0/P_factor*9.0/(eta**2.0) * (1 + (eta*(1-x)-1) * np.exp(eta*(1-x)))
    #print "vinet: " , energy_a
    return energy_a

#compute thermal part energy
def energy_mgd_powlaw( V_a, T_a, param_d ):
    # get parameter values
    theta0 = param_d['theta0']
    C_DP = param_d['const']['C_DP']
    P_factor = param_d['const']['P_factor']
    gamma= param_d['gamma0']*(V_a/param_d['V0'])**param_d['q']
    theta = param_d['theta0']*np.exp((-1)*(gamma-param_d['gamma0'])/param_d['q'])
    T_ref_a = 300 # here we use isothermal reference compression curve
    energy_therm_a = C_DP/atompermol*param_d['const']['Natom']*(T_a*debye_fun(theta/T_a) - T_ref_a*debye_fun(theta/T_ref_a ))
    return energy_therm_a

#return model total enerty
def energy_mod_total(T_a,V_a,param_d):
  return energy_vinet(V_a,param_d) + energy_mgd_powlaw(V_a,T_a,param_d)

def findroot(T_a,V_a,param_d,E):
  return energy_mod_total(T_a,V_a,param_d) - E

#find the temperature
result = []
for ind in range(len(V_a)):
  T_root = optimize.brentq(findroot, a=300,b=3000,args = (V_a[ind],param_d,E2[ind]),full_output = False)
  result.append(T_root)
  
print "result: ", result
#edit the dictionary
#plot energy from 300K to 3000K, compare with E2,

#eos_mod.py & eval_Pt_eos.py

P_resi = MGD_PowerLaw(V_a, T_root, param_d) - p2
print P_resi

plt.plot(p2,P_resi,'x')
