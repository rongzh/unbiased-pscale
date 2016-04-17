#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import collections
from eos_mod import MGD_PowerLaw

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


def debye_func(x):
    """
    Return debye integral value

    - calculation done using interpolation in a lookup table
    - interpolation done in log-space where behavior is close to linear
    - linear extrapolation is implemented manually
    """

    if(np.isscalar(x)):
        assert x >= 0, 'x values must be greater than zero.'
    else:
        #np.absolute(x)
        assert all(x >= 0), 'x values must be greater than zero.'
    # Lookup table
    # interpolate in log space where behavior is nearly linear
    debyex = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                       1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                       3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0,
                       5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    debyelogf = np.array([ 0.0, -0.03770187, -0.07580279, -0.11429475,
                          -0.15316866, -0.19241674, -0.2320279 , -0.27199378,
                          -0.31230405, -0.35294619, -0.39390815, -0.43518026,
                          -0.47674953, -0.51860413, -0.56072866, -0.64573892,
                          -0.73167389, -0.81841793, -0.90586032, -0.99388207,
                          -1.08236598, -1.17119911, -1.26026101, -1.34944183,
                          -1.43863241, -1.52771969, -1.61660856, -1.70519469,
                          -1.79338479, -1.88108917, -1.96822938, -2.05471771,
                          -2.14049175, -2.35134476, -2.55643273, -2.75507892,
                          -2.94682783, -3.13143746, -3.30880053, -3.47894273,
                          -3.64199587, -3.79820337, -3.94785746])
    # Create interpolation function
    logdebfun = interp1d(debyex, debyelogf, kind='cubic', bounds_error=False,
                         fill_value=np.nan)
    logfval = logdebfun(x)
    # Check for extrapolated values indicated by NaN
    #   - replace with linear extrapolation
    logfval = np.where(x > debyex[-1], debyelogf[-1] + (x - debyex[-1]) *
                       (debyelogf[-1]-debyelogf[-2])/(debyex[-1]-debyex[-2]),
                       logfval)
    # Exponentiate to get integral value
    return np.exp(logfval)


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
    energy_therm_a = C_DP/atompermol*param_d['const']['Natom']*(T_a*debye_func(theta/T_a) - T_ref_a*debye_func(theta/T_ref_a ))
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


