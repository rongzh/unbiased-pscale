#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.interpolate import interp1d, interp2d
import collections
from eos_mod import MGD_PowerLaw,debye_fun,Ptot_powerlaw,press_vinet, MGD_PowerLawShifted
#####################################
        #Build in Functions#
#####################################
def set_const(param_dic):
    param_dic['const']['Natom'] = 8.0
    param_dic['const']['kB'] =  8.6173324e-5 #eV per K
    param_dic['const']['P_factor'] = 160.217657#GPa in 1 eV/Ang^3
    param_dic['const']['R'] = 8.314462/1.6021764e-19 # eV/K per mol
    # 1eV/ang^3 = 160.217657GPa, 1eV = 1.6021764e-19Joules, 1Ang3 = e-30m^3
    param_dic['const']['C_DP'] = 3.0*param_dic['const']['R']#Dulong-Petit limit for Cv
    param_dic['const']['fcv'] = 0.123754/0.12664#the ratio of Cvmax/DP ###???????
    param_dic['const']['Atmospheric_P'] = 101325*10**(-9) #unit GPa
    param_dic['const']['mol_mass'] = 40.3044 #unit g/mol
    param_dic['const']['avogadro_const'] = 6.022*(10**23) #at/mol
    param_dic['const']['Cv_max_cell'] = param_dic['const']['C_DP']/param_dic['const']['avogadro_const']*param_dic['const']['Natom'] #ev/K/unitcell
    param_dic['const']['kT300'] = 1.0/40 #unit eV

def set_Jam_params(param_dic): #set parameters that Jamieson reported 
    param_dic['Jam_ref']['gamma0'] = 1.32 #reported by Jam
    param_dic['Jam_ref']['theta0'] = 760.0 #unit K
    param_dic['Jam_ref']['q'] = 1.0 #const
    param_dic['Jam_ref']['Cb'] = 6.597 #intercept for Up-Us line in fig1
    param_dic['Jam_ref']['S'] = 1.369 #slope for Up-Us line in fig1
    param_dic['Jam_ref']['rho0'] = 3.585 #g/cm^3

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
        return resid_a
    paramfit_a = optimize.leastsq( fit_vinet, param0_a , args = (P, Eta_a))
    return paramfit_a

def infer_mgd_temp_P(Pth,rho,param_dic): 
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
            T_root = optimize.brentq(findT, a=300,b=10000,args = (Pth[ind],rho[ind]),full_output = False)
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
            T_root = optimize.brentq(energy_mgd_powlaw, a=300,b=10000,args = (Eth[ind],rho[ind],param_dic),full_output = False)
            Tfit[ind] = T_root
    return Tfit

########################################
         #Compute the Temperature#
########################################

#read data from file
shock_dat = np.loadtxt(fname='UpUs-MgO.md', delimiter='|', skiprows=3)
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
print param_dic
set_Jam_params(param_dic)
#compute the pressure, density and energy using hugoniont equations
rho, P, E = calc_hugoniot(Up,Us,param_dic)

V =  rho_to_vol(rho) #Ang^3/unitcell #convert hugoniont density to volumes

#read Jamison's table to fit vinet model

Table4_dat = np.loadtxt(fname='Table4-MgO.md', delimiter='|', skiprows=1)
P300_TBL_a = Table4_dat[1:,2]
Eta300_TBL_a = (1 - Table4_dat[1:,0])

#should not compute V now, since V0 is changing every time
V0 = rho_to_vol(param_dic['Jam_ref']['rho0']) #unit Ang^3/unitcell
param0_a = np.array([V0, 100.0, 4.0]) #initial guess for ref_param
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

#compute Vinet Pressure
P300 = press_vinet(V, param_dic['Jam_ref'])
mask_a = P > P300 #get rid of errorous data points

T_P = infer_mgd_temp_P(P-P300,rho,param_dic)
print "T_P",T_P

#compute Vinet Energy
E300 = energy_vinet(V,param_dic['Jam_ref'],300.0)
E_test = energy_vinet(V0,param_dic['Jam_ref'],300.0)
mask_a = E > E300
T_E = infer_mgd_temp_E(E-E300,rho,param_dic)
print "T_E",T_E
#from IPython import embed; embed(); import ipdb; ipdb.set_trace()
print "the ratio", T_E*1.0/T_P


########Check with Jamieson's Table4 Value
print param_dic['Jam_ref']
const_array = [param_dic['const']['Natom'],param_dic['const']['Cv_max_cell']*param_dic['const']['fcv']*param_dic['const']['P_factor'] ]
Table4_dat = np.loadtxt(fname='Table4-MgO.md', delimiter='|', skiprows=1)
Table4_t = Table4_dat[0,1:]*1000 #unit K
Table4_v = (1-Table4_dat[4:,0])*V0 #unit Ang^3/unitcell
p_diff_Jam = np.zeros((len(Table4_v), len(Table4_t)))
Jam_therm_P = np.zeros((len(Table4_v), len(Table4_t)))
Eos_therm_P = np.zeros((len(Table4_v), len(Table4_t)))
Ratio = np.zeros((len(Table4_v), len(Table4_t)))
for ind, i in enumerate(Table4_t):
    print i
    P_PowerLaw = MGD_PowerLaw(Table4_v, i, param_dic['Jam_ref'],Natom = param_dic['const']['Natom'])
    print param_dic['const']
    P_Table4 = Table4_dat[4:,ind+1]
    print P_PowerLaw - P_Table4
    p_diff_Jam[:,ind] = P_PowerLaw - P_Table4
    Jam_therm_P[:,ind] = P_Table4-Table4_dat[4:,2]
    Eos_therm_P[:,ind] = P_PowerLaw-press_vinet(Table4_v, param_dic['Jam_ref'])
    Ratio[:,ind] = Jam_therm_P[:,ind] / Eos_therm_P[:,ind]
    print "Ratio of thermal Pressure at ", i, "K:", Jam_therm_P[:,ind] / Eos_therm_P[:,ind]

print np.shape(p_diff_Jam)
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
print np.shape(v_plt_array),np.shape(t_plt_array)
plt.scatter(v_plt_array,t_plt_array,c = p_diff_Jam,marker='+')


plt.colorbar()
plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Jamieson")
plt.show()
plt.scatter(v_plt_array,t_plt_array,c = Jam_therm_P,marker='+')
plt.colorbar()
plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Jamieson thermal Pressure")
plt.show()
plt.scatter(v_plt_array,t_plt_array,c = Eos_therm_P,marker='+')
plt.colorbar()
plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Eos thermal Pressure")
plt.show()
plt.scatter(v_plt_array,t_plt_array,c = Ratio,marker='+')
plt.colorbar()
plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Eos thermal Pressure")
plt.show()
####Dewaele used Powerlaw shifted, thus it cannot be used to compare

Dewaele_dic = dict()
Dewaele_report = [74.71*4, 161.0,4.01,800,1.45,0.8]
Dewaele_dic['V0'] = Dewaele_report[0]
Dewaele_dic['K0'] = Dewaele_report[1]
Dewaele_dic['K0p'] = Dewaele_report[2]
Dewaele_dic['gamma0'] = Dewaele_report[4]
Dewaele_dic['theta0'] = Dewaele_report[3]
Dewaele_dic['q'] = Dewaele_report[5]
p_diff_Dew = np.zeros((len(Table4_v), len(Table4_t)))
for ind, i in enumerate(Table4_t):
    P_PowerLaw = MGD_PowerLawShifted(Table4_v, i, Dewaele_dic, Natom = param_dic['const']['Natom'])
    P_Table4 = Table4_dat[4:,ind+1]
    p_diff_Dew[:,ind] = P_PowerLaw - P_Table4

r_max = p_diff_Dew.max(axis=0)
r_min = p_diff_Dew.min(axis = 0)
r = np.column_stack((r_min,r_max))
print r
plt.scatter(v_plt_array,t_plt_array,c = p_diff_Dew,marker='+')
plt.colorbar()

plt.xlabel("volume Ang^3")
plt.ylabel("Temperature K")
plt.title("Dewaele")
plt.show()

############# verify that comparison between Jamieson EOS model and Jamieson Table 4 are EXACT for the 300K column######
###Errors are not within the range of 0.01
P_TBL_model = press_vinet((1-Table4_dat[4:,0])*V0,param_dic['Jam_ref'])
P_TBL_300 = Table4_dat[4:,2]
print "P_TBL_model - P_TBL_300", P_TBL_model - P_TBL_300

P_TBL_MGD_300 = MGD_PowerLaw((1-Table4_dat[4:,0])*V0, 300.0, param_dic['Jam_ref'],Natom =param_dic['const']['Natom'])
print "P_TBL_MGD_300 - P_TBL_300", P_TBL_MGD_300 - P_TBL_300

#############Verify consistency of 300K hugoniot
#######use Jamieson's linear fit to Up-Us values (reported as cb and S in Table 2) to obtain average Up-Us values along hugoniot
Cb = param_dic['Jam_ref']['Cb']
S = param_dic['Jam_ref']['S']

#from IPython import embed; embed(); import ipdb; ipdb.set_trace()
Up_sample = np.linspace(0.05,2.7,100)
Us_sample = S * Up_sample + Cb
#########transform into P,V,E values using Rankine Hugoniot expressions
rho_sample, P_sample, E_sample = calc_hugoniot(Up_sample,Us_sample,param_dic)
##########Determine corresponding temperature according to Jamieson using digitized Figure 1 
Fig1_MgO_dat = np.loadtxt(fname='Fig1-MgO.md', delimiter=',', skiprows=1)
Fig1_MgO_T = Fig1_MgO_dat[:,0]*1000.0
Fig1_MgO_P = Fig1_MgO_dat[:,1]
f = interp1d(Fig1_MgO_P, Fig1_MgO_T)
infer_T_Fig1 = f(P_sample)
print "infer_T", infer_T_Fig1

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
    V_root = optimize.brentq(findV, a=30.0,b=10000.0,args = (infer_T_Fig1[ind],P_sample[ind],param_dic),full_output = False)
    Vfit[ind] = V_root


plt.plot( infer_T_Fig1/1000., P_sample,'b-',T_P/1000., P, 'kx',T_E/1000.,P,'rx', infer_T_E/1000.,P_sample, 'r',infer_T_P/1000., P_sample, 'k')
plt.xlabel("Temperature(kK)")
plt.ylabel("Pressure(GPa)")
plt.show()
from IPython import embed; embed(); import ipdb; ipdb.set_trace()
