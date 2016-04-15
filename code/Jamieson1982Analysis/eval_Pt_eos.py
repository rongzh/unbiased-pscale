#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import collections
from eos_mod import MGD_PowerLaw,debye_fun,Ptot_powerlaw,press_vinet

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
print Up,Us

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
p2 = rho1*Us*Up*10**(24) + p1 #unit GPa, and this is properly unit analyzed
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
plt.figure()
plt.plot(p2,P_resi,'x')
plt.xlabel('absolute P')
plt.ylabel('resi_P')
plt.show()
plt.plot(result,P_resi,'rx')
plt.xlabel('T')
plt.ylabel('resi_P')
plt.show()
Pthermal = Ptot_powerlaw(V_a,result,param_d,300)
plt.plot(Pthermal,P_resi+Pthermal,'rx')
plt.xlabel('thermal')
plt.ylabel('resi_P')
plt.show()


##conversion from PVT to vs vp.
def mass_conserv(us,up,rho1, rho2):
    return rho1*us - rho2*(us-up)
def momentum_conserv(us,up,rho1,p1,p2):
    #return rho1*us*up + p1 - p2
    p1 = 101325*10**(-9) #unit GPa
    return p2 - rho1*us*up*10**(24) + p1 #unit GPa
def energy_conserv(us,up,p1,p2,rho1,rho2,E2):
    #return (p1+p2)*(1/rho1 + 1/rho2)/2 - E2
    m_cell = 195*4/atompermol # (g/unitcell)
    f_conv_E = m_cell/160.217657 # (g/cell)*(eV/(GPa*Ang^3))
    return E2 - 0.5*(p1+p2) * f_conv_E *(1/rho1-1/rho2)#unit eV/unitcell

mass_range = max(rho2*(Us-Up))
momentum_range = max(p2-p1)
energy_range = max(E2)

print "testhere: ",mass_range, momentum_range,energy_range

def fitfunc(u,rho1,rho2,p1,p2,E2):
    us = u[0]
    up = u[1]
    return np.array(mass_conserv(us,up,rho1,rho2)*mass_conserv(us,up,rho1,rho2)/mass_range/mass_range+ 
        momentum_conserv(us,up,rho1,p1,p2) * momentum_conserv(us,up,rho1,p1,p2)/momentum_range/momentum_range + 
        energy_conserv(us,up,p1,p2,rho1,rho2,E2) * energy_conserv(us,up,p1,p2,rho1,rho2,E2)/energy_range/energy_range,dtype = 'f8')

guess = [0.3,4]
#set other parameters
#print "test three functions: ", mass_conserv(Us,Up,rho1,rho2), momentum_conserv(Us,Up,rho1,p1,p2), energy_conserv(Us,Up,p1,p2,rho1,rho2,E2)
#rho1new = rho1 * np.ones(len(rho2))
#p1new = p1* np.ones(len(p2))
#for i in range(len(rho2)):
#    popt = optimize.minimize(fitfunc,guess[:],args=(rho1new[i],rho2[i],p1new[i],p2[i],E2[i]),full_output=1)
#    print popt
#print"done"


##ployfit rho and T 
#print "temperaure",result #unit K
#print "density and its corresponding volume", rho2, V_a# unit grams/Ang^3
##get Pmod and Emod
#Emod = energy_mod_total(result,V_a,param_d)
#Pmod = MGD_PowerLaw(V_a, result, param_d)
#print Emod, Pmod
#for i in range(len(rho2)):
#    popt = optimize.leastsq(fitfunc,guess[:],args=(rho1new[i],rho2[i],p1new[i],Pmod[i],Emod[i]),full_output=1)
    #print popt
#modfit = np.polyfit(result,rho2,3)
#print modfit
#p = np.poly1d(modfit)
#tem_range = np.array([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,2900,3000])
#print "using polyfit: ", p(tem_range),tem_range
#print "computed density and its corresponding temperature: ", rho2, result# unit grams/Ang^3


#####Compute Jamieson's Temperature

param_d['theta0'] = 200
param_d['gamma0'] = 2.40
param_d['q'] = 1
#compute thermal part energy
def energy_Jam_mgd_powlaw( V_a, T_a, param_d ):
    # get parameter values
    theta0 = param_d['theta0']
    gamma= param_d['gamma0']*(V_a/param_d['V0'])**param_d['q']
    theta = param_d['theta0']*np.exp((-1)*(gamma-param_d['gamma0'])/param_d['q'])
    T_ref_a = 298 # here we use isothermal reference compression curve
    energy_therm_a = 0.12786*(T_a*debye_fun(theta/T_a) - T_ref_a*debye_fun(theta/T_ref_a ))
    return energy_therm_a

def Jam_fit(T_a,P,param_d):
    return P - (21.449 * 2.40) * energy_Jam_mgd_powlaw(195*param_d['const']['Natom']/atompermol/21.449,T_a,param_d)

r = []
for ind in range(len(p2)):
  T_root = optimize.brentq(Jam_fit, a=300,b=3000,args = (p2[ind],param_d),full_output = False)
  r.append(T_root)
  
print "r: ", r

####
rho0 = 21.4449/10**24 #grams/Ang^3
rho2 = rho0*Us/(Us-Up) #unit grams/Ang^3
V_0 = 195*param_d['const']['Natom']/atompermol/rho0 #unit Ang^3/unitcell


#read Jam's data here and fit the vinet model for 300K
dat = np.loadtxt("Fig.txt", delimiter = ",", skiprows = 1)
V300_a = (1 - dat[:,2]) * V_0
P300_a = dat[:,1]
def V_fit(param_a, P_a=P300_a, V_a=V300_a):
    param_d = {'V0':param_a[0],'K0':param_a[1],'K0p':param_a[2]}
    Pmod_a = press_vinet(V_a,param_d)
    resid_a = P_a-Pmod_a
    return resid_a
param0_a = np.array([V_0, 100.0, 4.0])
print V_fit(param0_a)
paramfit_a = optimize.leastsq( V_fit, param0_a )
print "%%%%%%%%%%%%"
print "paramfit_a"
print paramfit_a

paramtrue_a = paramfit_a[0]
print "true params: ", paramtrue_a
#set true dictionary for Jam's vinet model
paramtrue = dict()
paramtrue = {'V0':paramtrue_a[0],'K0':paramtrue_a[1],'K0p':paramtrue_a[2]}

#using computed V_a to find the corresponding P_vinet
V_a = 195*param_d['const']['Natom']/atompermol/rho2 #unit Ang^3/unitcell
print "V_a: " , V_a
P300 = press_vinet(V_a, paramtrue)
print "P300 is: ", P300

#get the thermal pressure
Pth = p2-P300
print "pth ", Pth



#plt.plot(p2,V_a)
#plt.show()

mask_a = p2 > P300

print "now p2 is: ", p2
def findT(T, Pth,rho2):
    atompermol = 6.022*(10**23) #at/mol
    rho0 = 21.4449 #unit g/cm^3
    V_0 = 195*param_d['const']['Natom']/atompermol/rho0*1e24 #unit Ang^3/unitcell
    g0 = 2.40
    thetaD = 200 #unit K
    thetaV = thetaD * np.exp(g0*(1-rho0/rho2))
    T0 = 300
    fcv = 0.12786/0.12664 #the ratio of Cvmax/DP
    kT0 = 1.0/40 #unit eV
    unit_conv = 160.217657
    return Pth - g0/V_0 * 3 * fcv * kT0 * (T/T0 * debye_fun(thetaV/T) - debye_fun(thetaV/T0)) * unit_conv

#from IPython import embed; embed(); import ipdb; ipdb.set_trace()
print "findT: ", findT(300,Pth,rho2*1e24), findT(3000,Pth,rho2*1e24)
thetaV = 200 * np.exp(2.4*(1-rho0/(rho2*1e24)))
print "thetaV", thetaV

Tfit = np.nan*np.ones(p2.shape)
for ind in range(len(Pth)):
    if mask_a[ind]:
        T_root = optimize.brentq(findT, a=300,b=10000,args = (Pth[ind],rho2[ind]*1e24),full_output = False)
        Tfit[ind] = T_root
#print "Tfit",Tfit
#print "eta",(V_0-V_a)/V_0
#print "p2:",p2
pvt_a = np.vstack((p2,(V_0-V_a)/V_0,Tfit))
print pvt_a.T

plt.plot(Tfit/1000,p2,'rx')
plt.xlim(0,2)
plt.ylim(0,100)
plt.xlabel("temperature [kK]")
plt.ylabel("P [GPa]")
plt.show()


a = V_0*np.linspace(1,0.5,num = 100)
p = press_vinet(a,paramtrue)
plt.plot(p2,V_a,'rx', p,a,'b')
plt.xlim(0,200)
plt.show()