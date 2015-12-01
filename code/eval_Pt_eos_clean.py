#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import collections
from eos_mod import MGD_PowerLaw,debye_fun,Ptot_powerlaw,press_vinet
#####################################
        #Build in Functions#
#####################################
def set_const():
    param_dic['const']['Natom'] = 4
    param_dic['const']['kB'] =  8.6173324e-5 #eV per K
    param_dic['const']['P_factor'] = 160.217657#GPa in 1 eV/Ang^3
    param_dic['const']['R'] = 8.314462/1.6021764e-19 # eV/K per mol
    # 1eV/ang^3 = 160.217657GPa, 1eV = 1.6021764e-19Joules, 1Ang3 = e-30m^3
    param_dic['const']['C_DP'] = 3*param_dic['const']['R']#Dulong-Petit limit for Cv
    param_dic['const']['rho_Pt'] = 21.472 #unit grams/cm3
    param_dic['const']['Atmospheric_P'] = 101325*10**(-9) #unit GPa
    param_dic['const']['mol_mass'] = 195 #unit g/mol
    param_dic['const']['avogadro_const'] = 6.022*(10**23) #at/mol

def rho_to_vol(rho): #rho unit gram/cm^3 -> vol unit Ang^3/unitcell
    return param_dic['const']['mol_mass']*param_dic['const']['Natom']/ param_dic['const']['avogadro_const']/rho*1e24

def vol_to_rho(vol):
    return param_dic['const']['mol_mass']*param_dic['const']['Natom']/ param_dic['const']['avogadro_const']/vol*1e24 

def entropy(V, T, param_d):
    V0 = param_d['V0']
    g0 = 2.40
    thetaD = 200.0 #unit K
    thetaV = thetaD * np.exp(g0*(1-V/V0))
    T0 = 300.0
    fcv = 0.12786/0.12664 #the ratio of Cvmax/DP
    kT300 = 1.0/40 #unit eV
    #entropy = 3*fcv * kT300/300 * param_dic['const']['Natom'] *(4.0/3*debye_fun(thetaV/T) - np.log(1-np.exp(-thetaV/T)))
    d_s = 3.0*fcv * kT300/300.0*param_dic['const']['Natom'] *(4.0/3*(debye_fun(thetaV/T) - debye_fun(thetaD/T0)) - np.log((1-np.exp(-thetaV/T))/(1-np.exp(-thetaD/T0))))
    return d_s
def energy_vinet( V_a, param_d,T ):
    V0 = param_d['V0']
    K0 = param_d['K0']
    K0p = param_d['K0p']
    P_factor = param_dic['const']['P_factor']
    g0 = 2.40
    x = (V_a/V0)**(1.0/3)
    eta = 3.0*(K0p- 1)/2.0
    print 'eta', eta

    energy_a = V0*K0/P_factor*9.0/(eta**2.0) * (1 + (eta*(1-x)-1) * np.exp(eta*(1-x)))
    d_s = entropy(V_a,T,param_d)
    print "SSSSSSSSS",d_s*T
    return energy_a + d_s*T

def calc_hugoniot(Up,Us,param_dic):
    rho = param_dic['const']['rho_Pt']*Us/(Us-Up) #unit grams/cm^3
    P =  param_dic['const']['rho_Pt']*Us*Up + param_dic['const']['Atmospheric_P'] #unit GPa, and this is properly unit analyzed
    #conv_E = param_dic['const']['mol_mass'] *  param_dic['const']['Natom']/param_dic['const']['avogadro_const']/ param_dic['const']['P_factor']
    v = rho_to_vol(rho)
    V0 = rho_to_vol(param_dic['const']['rho_Pt'])
    P_factor = param_dic['const']['P_factor']
    E = 0.5*(P + param_dic['const']['Atmospheric_P'])*(V0-v)/P_factor
    #conv_E = param_dic['const']['mol_mass'] *  param_dic['const']['Natom']/ param_dic['const']['P_factor']
    #E = 0.5 * conv_E * (P + param_dic['const']['Atmospheric_P']) * (1/param_dic['const']['rho_Pt'] - 1/rho) #unit eV/unitcell
    return rho,P,E #unit grams/cm^3, GPa, eV/unitcell


def fit_ref_vinet(param0_a,P,V):
    def fit_vinet(param0_a,P0 = P,V0 = V):
        param_dic = {'V0':param0_a[0],'K0':param0_a[1],'K0p':param0_a[2]}
        Pmod_a = press_vinet(V,param_dic)
        resid_a = P-Pmod_a
        return resid_a
    paramfit_a = optimize.leastsq( fit_vinet, param0_a )
    param_dic['Jam_ref']['V0'] = paramfit_a[0][0]
    param_dic['Jam_ref']['K0'] = paramfit_a[0][1]
    param_dic['Jam_ref']['K0p'] = paramfit_a[0][2]
    return

def infer_mgd_temp_P(Pth,rho,param_dic): #input: thermal pressure, density and parameter dictionary
    def findT(T, P,rho):
        #rho0 = param_dic['const']['rho_Pt']
        rho0 = 21.4449 #unit g/cm^3
        V0 = rho_to_vol(rho0) #unit Ang^3/unitcell
        g0 = 2.40
        thetaD = 200.0 #unit K
        #thetaV = thetaD * np.exp(g0*(1-rho0/rho))
        thetaV = thetaD * np.exp(g0*(1-rho0/rho))
        T0 = 300.0
        fcv = 0.12786/0.12664 #the ratio of Cvmax/DP
        kT300 = 1.0/40 #unit eV
        return P - g0/V0 * 3 * fcv * kT300/300.0* param_dic['const']['Natom'] * T0 * (T/T0 * debye_fun(thetaV/T) - debye_fun(thetaV/T0)) * param_dic['const']['P_factor']
    Tfit = np.nan*np.ones(Pth.shape) #the result array
    for ind in range(len(Pth)): #loop through the array and find each correspondng T
        if mask_a[ind]:
            T_root = optimize.brentq(findT, a=300,b=10000,args = (Pth[ind],rho[ind]),full_output = False)
            Tfit[ind] = T_root
    return Tfit

def infer_mgd_temp_E(Eth,rho,param_dic):
    def energy_mgd_powlaw(  T, Eth, rho ):
        rho0 = 21.4449 #unit g/cm^3
        #rho0 = param_dic['const']['rho_Pt']
        P_factor = param_dic['const']['P_factor']
        g0 = 2.40
        thetaD = 200.0 #unit K
        thetaV = thetaD * np.exp(g0*(1-rho0/rho))
        T0 = 300.0
        fcv = 0.12786/0.12664 #the ratio of Cvmax/DP
        kT300 = 1.0/40 #unit eV
        energy_therm_a =3*fcv * kT300/300.0 * param_dic['const']['Natom'] * (T/T0 * debye_fun(thetaV/T) - debye_fun(thetaV/T0)) * param_dic['const']['P_factor']
        return Eth - energy_therm_a
    Tfit = np.nan*np.ones(Eth.shape) #the result array
    for ind in range(len(Eth)): #loop through the array and find each correspondng T
        if mask_a[ind]:
            T_root = optimize.brentq(energy_mgd_powlaw, a=300,b=10000,args = (Eth[ind],rho[ind]),full_output = False)
            Tfit[ind] = T_root
    return Tfit

########################################
         #Compute the Temperature#
########################################

#read data from file
dat = np.loadtxt(fname='UpUs-Pt.md', delimiter='|', skiprows=3)
print dat
Up = dat[:,0]
Us = dat[:,1]
plt.plot(Up,Us,'rx')
plt.xlabel('Up[km/s]')
plt.ylabel('Us[km/s]')
plt.show()
print Up,Us

param_dic = dict()
param_dic['const'] = dict()
set_const()
rho, P, E = calc_hugoniot(Up,Us,param_dic)

V =  rho_to_vol(rho) #Ang^3/unitcell
#print vol_to_rho(v)

#read Jamison's table to fit vinet model
V0 = rho_to_vol(param_dic['const']['rho_Pt']) #unit Ang^3/unitcell
dat = np.loadtxt("Fig.txt", delimiter = ",", skiprows = 1)
V300_a = (1 - dat[:,2]) * V0
P300_a = dat[:,1]
param0_a = np.array([V0, 100.0, 4.0]) #initial guess for ref_param
param_dic['Jam_ref'] = dict()
ref_a = fit_ref_vinet(param0_a,P300_a,V300_a)

#compute Vinet Pressure
P300 = press_vinet(V, param_dic['Jam_ref'])
mask_a = P > P300 #get rid of errorous data points
T_P = infer_mgd_temp_P(P-P300,rho,param_dic)
print T_P

#compute Vinet Energy
E300 = energy_vinet(V,param_dic['Jam_ref'],300.0)
E_test = energy_vinet(V0,param_dic['Jam_ref'],300.0)
mask_a = E > E300
T_E = infer_mgd_temp_E(E-E300,rho,param_dic)
print T_E


V0 = param_dic['Jam_ref']['V0']
K0 = param_dic['Jam_ref']['K0']
K0p = param_dic['Jam_ref']['K0p']
P_factor = param_dic['const']['P_factor']
rho0 = 21.4449 #unit g/cm^3
g0 = 2.40
thetaD = 200.0 #unit K
thetaV = thetaD * np.exp(g0*(1-rho0/rho))
T0 = 300.0
fcv = 0.12786/0.12664 #the ratio of Cvmax/DP
kT0 = 1.0/40 #unit eV
x = (V/V0)**(1.0/3)
#print 'x',x
eta = 3.0*(K0p- 1)/2.0
print 'eta', eta

energy_a = V0*K0/P_factor*9.0/(eta**2.0) * (1 + (eta*(1-x)-1) * np.exp(eta*(1-x)))/param_dic['const']['avogadro_const']
entropy = 3*fcv * kT0 *(4.0/3*(debye_fun(thetaV/300.0) - debye_fun(thetaD/T0)) - np.log((1-np.exp(-thetaV/300.0))/(1-np.exp(-thetaD/T0))))
from IPython import embed; embed(); import ipdb; ipdb.set_trace()
##########################