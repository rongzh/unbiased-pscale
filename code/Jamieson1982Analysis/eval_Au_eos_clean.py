#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.interpolate import interp1d, interp2d
import collections
from eos_mod import MGD_PowerLaw,debye_fun,Ptot_powerlaw,press_vinet
#####################################
        #Build in Functions#
#####################################
def set_const(param_dic):
    param_dic['const']['Natom'] = 4.0
    param_dic['const']['kB'] =  8.6173324e-5 #eV per K
    param_dic['const']['P_factor'] = 160.217657#GPa in 1 eV/Ang^3
    param_dic['const']['R'] = 8.314462/1.6021764e-19 # eV/K per mol
    # 1eV/ang^3 = 160.217657GPa, 1eV = 1.6021764e-19Joules, 1Ang3 = e-30m^3
    param_dic['const']['C_DP'] = 3.0*param_dic['const']['R']#Dulong-Petit limit for Cv
    param_dic['const']['fcv'] = 0.12500/0.12664#the ratio of Cvmax/DP
    param_dic['const']['Atmospheric_P'] = 101325*10**(-9) #unit GPa
    param_dic['const']['mol_mass'] = 196.96 #unit g/mol
    param_dic['const']['avogadro_const'] = 6.022*(10**23) #at/mol
    param_dic['const']['Cv_max_cell'] = param_dic['const']['C_DP']/param_dic['const']['avogadro_const']*param_dic['const']['Natom'] #ev/K/unitcell
    param_dic['const']['kT300'] = 1.0/40 #unit eV

def set_Jam_params(param_dic): #set parameters that Jamieson reported 
    param_dic['Jam_ref']['gamma0'] = 3.215 #reported by Jam
    param_dic['Jam_ref']['theta0'] = 170.0 #unit K
    param_dic['Jam_ref']['q'] = 1.0 #const
    param_dic['Jam_ref']['Cb'] = 2.975 #intercept for Up-Us line in fig1
    param_dic['Jam_ref']['S'] = 1.896 #slope for Up-Us line in fig1
    param_dic['Jam_ref']['rho0'] = 19.2827 #g/cm^3

def rho_to_vol(rho): #rho unit gram/cm^3 -> vol unit Ang^3/unitcell
    return param_dic['const']['mol_mass']*param_dic['const']['Natom']/ param_dic['const']['avogadro_const']/rho*1e24

def vol_to_rho(vol): #vol unit Ang^3/unitcell -> rho unit gram/cm^3
    return param_dic['const']['mol_mass']*param_dic['const']['Natom']/ param_dic['const']['avogadro_const']/vol*1e24 

def entropy(V, T, param_d): #compute the change of entropy d_s
    V0 = param_d['V0']
    gamma0 = param_d['gamma0']
    theta0 = param_d['theta0'] #unit K
    thetaV = theta0 * np.exp(gamma0*(1-V/V0))
    T0 = 300.0
    fcv = param_dic['const']['fcv']
    Cv_max_cell = param_dic['const']['Cv_max_cell']
    d_s = fcv * Cv_max_cell *(4.0/3*(debye_fun(thetaV/T) - debye_fun(theta0/T0)) - np.log((1.0-np.exp(-thetaV/T))/(1.0-np.exp(-theta0/T0))))
    return d_s

def energy_vinet( V_a, param_d,T ): #compute vinet energy
    V0 = param_d['V0']
    K0 = param_d['K0']
    K0p = param_d['K0p']
    P_factor = param_dic['const']['P_factor']
    gamma0 = param_d['gamma0']
    x = (V_a/V0)**(1.0/3)
    eta = 3.0*(K0p- 1)/2.0

    energy_a = V0*K0/P_factor*9.0/(eta**2.0) * (1 + (eta*(1-x)-1) * np.exp(eta*(1-x)))
    d_s = entropy(V_a,T,param_d)
    return energy_a + d_s*T

def calc_hugoniot(Up,Us,param_dic): #using Up Us to compute corresponding pressure, density and energy.
    rho = param_dic['Jam_ref']['rho0']*Us/(Us-Up) #unit grams/cm^3
    P =  param_dic['Jam_ref']['rho0']*Us*Up + param_dic['const']['Atmospheric_P'] #unit GPa, and this is properly unit analyzed
    v = rho_to_vol(rho) #convert to volume
    V_0 = rho_to_vol(param_dic['Jam_ref']['rho0'])
    P_factor = param_dic['const']['P_factor']
    E = 0.5*(P + param_dic['const']['Atmospheric_P'])*(V_0-v)/P_factor
    return rho,P,E #unit grams/cm^3, GPa, eV/unitcell


def fit_ref_vinet(param0_a,P,Eta_a):
    #input initial guess of the vinet parameters and data, return an array of vinet parameters(V0,K0,K0p)
    def fit_vinet(param0_a,P0 = P,Eta = Eta_a):
        x = Eta**(1./3)
        #compute vinet pressure
        Pmod_a = 3*param0_a[1]*(1.-x)*x**(-2)*np.exp(3./2.*(param0_a[2] - 1.)*(1. - x))
        resid_a = P-Pmod_a
        print resid_a
        return resid_a
    paramfit_a = optimize.leastsq( fit_vinet, param0_a , args = (P, Eta_a))
    return paramfit_a

def infer_mgd_temp_P(Pth,rho,param_dic): 
    #from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#input: thermal pressure, density and parameter dictionary, return an array of temperatures that corresponds to the pressure and volume using MGD model.
    def findT(T, P,rho):
        rho0 = param_dic['Jam_ref']['rho0']
        V_0 = rho_to_vol(rho0) #unit Ang^3/unitcell
        gamma0 = param_dic['Jam_ref']['gamma0']
        theta0 = param_dic['Jam_ref']['theta0'] #unit K
        thetaV = theta0 * np.exp(gamma0*(1-rho0/rho))
        T0 = 300.0
        fcv = param_dic['const']['fcv']
        kT300 = param_dic['const']['kT300'] #unit eV
        Cv_max_cell = param_dic['const']['Cv_max_cell']
        return P - gamma0/V_0 * fcv * Cv_max_cell* T0 * (T/T0 * debye_fun(thetaV/T) - debye_fun(theta0/T0)) * param_dic['const']['P_factor']
    Tfit = np.nan*np.ones(Pth.shape) #the result array
    for ind in range(len(Pth)): #loop through the array and find each correspondng T
        if mask_a[ind]:
            T_root = optimize.brentq(findT, a=100,b=1000000,args = (Pth[ind],rho[ind]),full_output = False)
            Tfit[ind] = T_root
    return Tfit

def infer_mgd_temp_E(Eth,rho,param_dic):
#input: thermal energy, density, and parameter dictionary, return an array of temperatures that recorresponds the energy and volume.
    def energy_mgd_powlaw(  T, Eth, rho,param_dic ):
        rho0 = param_dic['Jam_ref']['rho0']#g/cm^3
        gamma0 = param_dic['Jam_ref']['gamma0']
        theta0 = param_dic['Jam_ref']['theta0'] #unit K
        thetaV = theta0 * np.exp(gamma0*(1-rho0/rho))
        T0 = 300.0
        fcv = param_dic['const']['fcv'] #the ratio of Cvmax/DP
        Cv_max_cell = param_dic['const']['Cv_max_cell'] #eV/K/unitcell
        energy_therm_a =fcv * Cv_max_cell * (T* debye_fun(thetaV/T) -T0* debye_fun(theta0/T0))
        return Eth - energy_therm_a
    Tfit = np.nan*np.ones(Eth.shape) #the result array
    for ind in range(len(Eth)): #loop through the array and find each correspondng T
        if mask_a[ind]:
            T_root = optimize.brentq(energy_mgd_powlaw, a=100,b=1000000,args = (Eth[ind],rho[ind],param_dic),full_output = False)
            Tfit[ind] = T_root
    return Tfit

########################################
         #Compute the Temperature#
########################################

#read data from file
shock_dat = np.loadtxt(fname='UpUs-Au.md', delimiter='|', skiprows=2)
print shock_dat
Up = shock_dat[:,0]
Us = shock_dat[:,1]
plt.plot(Up,Us,'rx')
plt.xlabel('Up[km/s]')
plt.ylabel('Us[km/s]')
plt.show()
print Up,Us

##set up the dictionaries
param_dic = dict()
param_dic['const'] = dict()
param_dic['Jam_ref'] = dict()
set_const(param_dic)
set_Jam_params(param_dic)
#compute the pressure, density and energy using hugoniont equations
rho, P, E = calc_hugoniot(Up,Us,param_dic)

V =  rho_to_vol(rho) #Ang^3/unitcell #convert hugoniont density to volumes

#read Jamison's table to fit vinet model

Table4_dat = np.loadtxt(fname='Table4-Au.md', delimiter='|', skiprows=1)
P300_TBL_a = Table4_dat[4:,2]
Eta300_TBL_a = (1 - Table4_dat[4:,0])
#should not compute V now, since V0 is changing every time
V0 = rho_to_vol(param_dic['Jam_ref']['rho0']) #unit Ang^3/unitcell
param0_a = np.array([V0, 160.0, 5.0]) #initial guess for ref_param
Jam_ref_a = fit_ref_vinet(param0_a,P300_TBL_a,Eta300_TBL_a)

#put parameter values into dictioanry
print "Jam_ref_a", Jam_ref_a
param_dic['Jam_ref']['V0'] = Jam_ref_a[0][0]
param_dic['Jam_ref']['K0'] = Jam_ref_a[0][1]
param_dic['Jam_ref']['K0p'] = Jam_ref_a[0][2]
V0 = param_dic['Jam_ref']['V0'] #now all V0 start using the inferred one: ['Jam_ref']['V0']

V300_a = Eta300_TBL_a * V0
P_TBL_mod = press_vinet(V300_a,param_dic['Jam_ref'])
print "P300 difference: ", P_TBL_mod-P300_TBL_a 
#from IPython import embed; embed(); import ipdb; ipdb.set_trace()
#compute Vinet Pressure
P300 = press_vinet(V, param_dic['Jam_ref'])
mask_a = P > P300 #get rid of errorous data points
#print "P300 difference: ", P300-P300_TBL_a 
T_P = infer_mgd_temp_P(P-P300,rho,param_dic)
print "T_P",T_P

#compute Vinet Energy
E300 = energy_vinet(V,param_dic['Jam_ref'],300.0)
E_test = energy_vinet(V0,param_dic['Jam_ref'],300.0)
mask_a = E > E300
T_E = infer_mgd_temp_E(E-E300,rho,param_dic)
print P
print "T_E",T_E

print "the ratio", T_E*1.0/T_P



##########Testing#########################
# V0 = param_dic['Jam_ref']['V0']
# K0 = param_dic['Jam_ref']['K0']
# K0p = param_dic['Jam_ref']['K0p']
# P_factor = param_dic['const']['P_factor']
# rho0 = param_dic['Jam_ref']['rho0']
# gamma0 = param_dic['Jam_ref']['gamma0']
# theta0 = param_dic['Jam_ref']['theta0'] #unit K
# thetaV = theta0 * np.exp(gamma0*(1-rho0/rho))
# T0 = 300.0
# fcv = param_dic['const']['fcv'] #the ratio of Cvmax/DP
# #kT0 = 1.0/40 #unit eV
# x = (V/V0)**(1.0/3)
# eta = 3.0*(K0p- 1)/2.0

# energy_a = V0*K0/P_factor*9.0/(eta**2.0) * (1 + (eta*(1-x)-1) * np.exp(eta*(1-x)))/param_dic['const']['avogadro_const']

##plot s-T
# T_a = np.linspace(300,3000,500)
# s_1 = entropy(V0/5, T_a, param_dic['Jam_ref'])
# s_2 = entropy(2*V0/5, T_a, param_dic['Jam_ref'])
# s_3 = entropy(3*V0/5, T_a, param_dic['Jam_ref'])
# s_4 = entropy(4*V0/5, T_a, param_dic['Jam_ref'])
# s_5 = entropy(5*V0/5, T_a, param_dic['Jam_ref'])
# plt.plot(T_a,s_1,'r',T_a,s_2,'b',T_a,s_3,'g',T_a,s_4,'y',T_a,s_5)
# plt.xlabel("Temperature")
# plt.ylabel("Entropy")
# plt.show()

# ##derivative of Free energy
# v_a = np.linspace(V0/2,V0,100)
# F = energy_vinet(v_a,param_dic['Jam_ref'],300.0)
# P_a = press_vinet(v_a,param_dic['Jam_ref'])
# diff_v = v_a[1:] - v_a[:-1]
# diff_F = F[1:] - F[:-1]
# p_est = diff_F/diff_v
# plt.plot(v_a,np.gradient(F,v_a[1]-v_a[0])*(-1)*160.217657,'bx')
# plt.plot(v_a,P_a,'r-')


# T_a = np.linspace(300,3000,500)
# rho_1 = vol_to_rho(V0/5.0)
# rho_2 = vol_to_rho(V0*2/5.0)
# rho_3 = vol_to_rho(V0*3/5.0)
# rho_4 = vol_to_rho(V0*4/5.0)
# rho_5 = vol_to_rho(V0*5/5.0)
# theta_1 = theta0 * np.exp(gamma0*(1-rho0/rho_1))
# theta_2 = theta0 * np.exp(gamma0*(1-rho0/rho_2))
# theta_3 = theta0 * np.exp(gamma0*(1-rho0/rho_3))
# theta_4 = theta0 * np.exp(gamma0*(1-rho0/rho_4))
# theta_5 = theta0 * np.exp(gamma0*(1-rho0/rho_5))

# s_1 = entropy(V0/5.0, T_a, param_dic['Jam_ref'])
# s_2 = entropy(2*V0/5.0, T_a, param_dic['Jam_ref'])
# s_3 = entropy(3*V0/5.0, T_a, param_dic['Jam_ref'])
# s_4 = entropy(4*V0/5.0, T_a, param_dic['Jam_ref'])
# s_5 = entropy(5*V0/5.0, T_a, param_dic['Jam_ref'])
# #print "C_DP/avogadro_const", param_dic['const']['C_DP']/param_dic['const']['avogadro_const']
# Cv_max_cell = param_dic['const']['Cv_max_cell']
# Cv_1 = fcv *Cv_max_cell*(4.0*debye_fun(theta_1/T_a) - 3.0*theta_1/T_a/(np.exp(theta_1/T_a)-1))
# Cv_2 = fcv *Cv_max_cell*(4.0*debye_fun(theta_2/T_a) - 3.0*theta_2/T_a/(np.exp(theta_2/T_a)-1))
# Cv_3 = fcv *Cv_max_cell*(4.0*debye_fun(theta_3/T_a) - 3.0*theta_3/T_a/(np.exp(theta_3/T_a)-1))
# Cv_4 = fcv *Cv_max_cell*(4.0*debye_fun(theta_4/T_a) - 3.0*theta_4/T_a/(np.exp(theta_4/T_a)-1))
# Cv_5 = fcv *Cv_max_cell*(4.0*debye_fun(theta_5/T_a) - 3.0*theta_5/T_a/(np.exp(theta_5/T_a)-1))
# plt.plot(T_a,Cv_1,'r',T_a,Cv_2,'b',T_a,Cv_3,'g',T_a,Cv_4,'y',T_a,Cv_5,'k')
# plt.xlabel("Temperature")
# plt.ylabel("Cv")

# plt.plot(T_a,T_a*np.gradient(s_1,T_a[1]-T_a[0]),'rx',T_a,T_a*np.gradient(s_2,T_a[1]-T_a[0]),'bx',T_a,T_a*np.gradient(s_3,T_a[1]-T_a[0]),'gx',T_a,T_a*np.gradient(s_4,T_a[1]-T_a[0]),'yx',T_a,T_a*np.gradient(s_5,T_a[1]-T_a[0]),'kx')
# plt.show()

# plt.scatter(P,T_P,c = np.log(T_E*1.0/T_P))
# plt.colorbar()
# plt.show()
#from IPython import embed; embed(); import ipdb; ipdb.set_trace()
##########################

########Check with Jamieson's Table4 Value
Table4_dat = np.loadtxt(fname='Table4-Au.md', delimiter='|', skiprows=1)
Table4_t = Table4_dat[0,1:]*1000 #unit K
Table4_v = (1-Table4_dat[4:,0])*V0 #unit Ang^3/unitcell
p_diff_Jam = np.zeros((len(Table4_v), len(Table4_t)))
for ind, i in enumerate(Table4_t):
    P_PowerLaw = MGD_PowerLaw(Table4_v, i, param_dic['Jam_ref'],Natom = param_dic['const']['Natom'])
    P_Table4 = Table4_dat[4:,ind+1]
    p_diff_Jam[:,ind] = P_PowerLaw - P_Table4
    # plt.scatter(Table4_v,np.ones(len(Table4_v))*i,c = P_PowerLaw-P_Table4,marker='+')
    # plt.clim([-0.3,0.3])
r_max = p_diff_Jam.max(axis=0)
r_min = p_diff_Jam.min(axis = 0)
r = np.column_stack((r_min,r_max))
print r
v_plt_array = np.zeros((len(Table4_v), len(Table4_t)))
for ind,i in enumerate(Table4_t):
    v_plt_array[:,ind] = Table4_v
t_plt_array = np.zeros((len(Table4_v), len(Table4_t)))
for ind, i in enumerate(Table4_v):
    t_plt_array[ind,:] = Table4_t
#print np.shape(v_plt_array),np.shape(t_plt_array)
plt.scatter(v_plt_array,t_plt_array,c = p_diff_Jam,marker='+')
plt.colorbar()
plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Jamieson")
plt.show()
Fei_dic = dict()
fei_report = [67.85, 167,6.0,170.0,2.97,0.6]
Fei_dic['V0'] = fei_report[0]
Fei_dic['K0'] = fei_report[1]
Fei_dic['K0p'] = fei_report[2]
Fei_dic['gamma0'] = fei_report[4]
Fei_dic['theta0'] = fei_report[3]
Fei_dic['q'] = fei_report[5]
p_diff_Fei = np.zeros((len(Table4_v), len(Table4_t)))
for ind, i in enumerate(Table4_t):
    P_PowerLaw = MGD_PowerLaw(Table4_v, i, Fei_dic,Natom = param_dic['const']['Natom'])
    P_Table4 = Table4_dat[4:,ind+1]
    p_diff_Fei[:,ind] = P_PowerLaw - P_Table4
    # plt.scatter(Table4_v,np.ones(len(Table4_v))*i,c = P_PowerLaw-P_Table4,marker='+')
    # plt.clim([-1.0,3.0])
r_max = p_diff_Fei.max(axis=0)
r_min = p_diff_Fei.min(axis = 0)
r = np.column_stack((r_min,r_max))
print r
plt.scatter(v_plt_array,t_plt_array,c = p_diff_Fei,marker='+')
plt.colorbar()
plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Fei")
plt.show()

############# verify that comparison between Jamieson EOS model and Jamieson Table 4 are EXACT for the 300K column######
###Errors are not within the range of 0.01
P_TBL_model = press_vinet((1-Table4_dat[4:,0])*V0,param_dic['Jam_ref'])
P_TBL_300 = Table4_dat[4:,2]
print "P_TBL_model - P_TBL_300", P_TBL_model - P_TBL_300

P_TBL_MGD_300 = MGD_PowerLaw((1-Table4_dat[4:,0])*V0, 300.0, param_dic['Jam_ref'],Natom = param_dic['const']['Natom'])
print "P_TBL_MGD_300 - P_TBL_300", P_TBL_MGD_300 - P_TBL_300

#############Verify consistency of 300K hugoniot
#######use Jamieson's linear fit to Up-Us values (reported as cb and S in Table 2) to obtain average Up-Us values along hugoniot
Cb = param_dic['Jam_ref']['Cb']
S = param_dic['Jam_ref']['S']

Up_sample = np.linspace(0.01,1.03,100)
Us_sample = S * Up_sample + Cb
#########transform into P,V,E values using Rankine Hugoniot expressions
rho_sample, P_sample, E_sample = calc_hugoniot(Up_sample,Us_sample,param_dic)

##########Determine corresponding temperature according to Jamieson using digitized Figure 1 
Fig1_Au_dat = np.loadtxt(fname='Fig1-Au.md', delimiter=',', skiprows=1)
Fig1_Au_T = Fig1_Au_dat[:,0]*1000.0
Fig1_Au_P = Fig1_Au_dat[:,1]
f = interp1d(Fig1_Au_P, Fig1_Au_T)
infer_T_Fig1 = f(P_sample)
print "infer_T", infer_T_Fig1
#from IPython import embed; embed(); import ipdb; ipdb.set_trace()
##########Compare with your own inferred temperatures using Jamieson EOS model
V_sample = rho_to_vol(rho_sample)
E300_sample = energy_vinet(V_sample,param_dic['Jam_ref'],300.0)
P300_sample = press_vinet(V_sample, param_dic['Jam_ref'])
mask_a = P_sample > 0

infer_T_E = infer_mgd_temp_E(E_sample-E300_sample,rho_sample,param_dic)
print "infer_T_E", infer_T_E
infer_T_P = infer_mgd_temp_P(P_sample-P300_sample,rho_sample,param_dic)
print "infer_T_P", infer_T_P

#infer for the corresponding volume
def findV(V,T,P,param_dic):
    return P - MGD_PowerLaw(V,T,param_dic['Jam_ref'],Natom = param_dic['const']['Natom'])
Vfit = np.ones(P_sample.shape) #the result array
for ind in range(len(P_sample)): #loop through the array and find each correspondng T
    V_root = optimize.brentq(findV, a=30.0,b=40000.0,args = (infer_T_Fig1[ind],P_sample[ind],param_dic),full_output = False)
    Vfit[ind] = V_root

#plt.plot(P_sample, infer_T, 'r', P, T_P,'bx',P,T_E,'gx', P_sample, infer_T_E,'k', P_sample, infer_T_P,'y')
plt.plot( infer_T_Fig1/1000., P_sample,'b-',T_P/1000., P, 'kx',T_E/1000.,P,'rx', infer_T_E/1000.,P_sample, 'r',infer_T_P/1000., P_sample, 'k')
plt.xlabel("Temperature(kK)")
plt.ylabel("Pressure(GPa)")
plt.show()
from IPython import embed; embed(); import ipdb; ipdb.set_trace()
