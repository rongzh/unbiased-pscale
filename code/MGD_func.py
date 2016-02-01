#!/usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
def debye_fun(x):
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


def MGD_PowerLaw(volume, temperature,p_eos, Natom): # MGD_PowerLaw
  
  if np.isscalar(temperature):
    temperature = temperature*np.ones(len(volume))

  assert len(p_eos)==6, 'EOS parameter array must have correct length of 6'
  assert len(volume)==len(temperature), 'temperature should be a scalar or its length should be equal to volume'

  Kb = 13.806488*10**(-24)  #J per K
  P_conv_Fac= 160.217657*6.24*10**(18) #GPa in 1 J/Ang^3
  C_DP = 3*Kb*Natom*P_conv_Fac
  # Natom = # of atoms in unitcell
  # V = volume of unitcell
	#the cold function(Vinet function) Vinet function parameters
	#V0_0Fe = 162.12
	#K0_0Fe = 262.3
	#Kp_0Fe = 4.044
  #V0_13Fe = 163.16
	#K0_13Fe = 243.8
  #Kp_13Fe = 4.160
  V0 = p_eos[0]
  K0 = p_eos[1]
  Kp = p_eos[2]

	#Thermal function parameters
	#Theta0_0Fe = 1000
	#Gamma0_0Fe = 1.675
	#q_0Fe = 1.39
  #Theta0_13Fe = 1000
	#Gamma0_13Fe = 1.400
  #q_13Fe = 0.56
	
  Theta0 = p_eos[3]
  Gamma0 = p_eos[4]
  q = p_eos[5]
  RefT = 300.0 #unit is K
	
  x = (volume/V0)**(1./3)
  Vinet = 3*K0*(1.-x)*x**(-2)*np.exp(3./2.*(Kp - 1.)*(1. - x))

  gamma= Gamma0 *(volume/V0)**q
  theta = Theta0*np.exp((-1)*(gamma-Gamma0)/q)


#compute the P_thermal(V,300K)
  Debye_Int = debye_fun(theta/RefT)
  P_th_ref = C_DP*gamma*RefT*Debye_Int/volume

#compute P_th in different temperatures
  P_th = (C_DP*gamma*temperature*debye_fun(theta/temperature))/volume
    	
#compute P(V,T)
  MGD = Vinet + P_th - P_th_ref
  return MGD

def MGD_PowerLawShifted(volume, temperature, p_eos, Natom):

  # Natom = # of atoms in unitcell
  # V = volume of unitcell
  P_conv_Fac= 160.217657 #GPa in 1 eV/Ang^3
  Kb = 8.6173324e-5 #eV per K
  C_DP = 3*Kb*Natom#Dulong-Petit limit for Cv

	#Vinet function parameters
  #sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
  V0 = p_eos[0] #V0_Ne = 22.234
  K0 = p_eos[1] #K0_Ne = 1.070
  Kp = p_eos[2] #Kp_Ne = 8.40
	#Thermal function parameters
  Theta0 = p_eos[3] #Theta0_Ne = 75.1
  Gamma0 = p_eos[4] #Gamma0_Ne = 2.442
  q = p_eos[5] #q_Ne = 0.97
	#RefT = 0

  x = (volume/V0)**(1./3)
  Vinet = 3.*K0*(1-x)*x**(-2)*np.exp(3./2.*(Kp - 1.)*(1-x)) #Pcold = Vinet_Ne
  
  gammaV = Gamma0*x**(3*q)+1./2
  thetaV = Theta0*x**(-3./2)*np.exp(Gamma0/q*((1-x**(3.*q))))

  debye_Int = debye_fun(thetaV/temperature)
  P_th = (C_DP*temperature*gammaV/volume*debye_Int)*P_conv_Fac

  #compute P(V,T)
  MGD = Vinet + P_th
  return MGD

"""
#test Dewaele's table
p_eos = np.array([22.234,1.070,8.40,75.1,2.442,0.97])
volume = np.array([13.69743329, 12.31533725, 10.845, 10.305, 7.827])
temperature = np.array([298,298,500,750,900])
print (MGD_PowerLawShifted(volume, temperature,p_eos,4))
"""

#plot Dewaele's table
 #sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
"""
p_eos = np.array([22.234,1.070,8.40,75.1,2.442,0.97])
Nat = 1
Nedat = np.loadtxt(fname='Ne.md', delimiter='|', skiprows=3)
#temp_298 = np.zeros([34])
vol_298 = np.zeros([34])
ob_298 = np.zeros([34])
#temp_500 = np.zeros([5])
vol_500 = np.zeros([5])
ob_500 = np.zeros([5])
#temp_600 = np.zeros([6])
vol_600 = np.zeros([6])
ob_600 = np.zeros([6])
#temp_750 = np.zeros([5])
vol_750 = np.zeros([5])
ob_750 = np.zeros([5])
#temp_900 = np.zeros([6])
vol_900 = np.zeros([6])
ob_900 = np.zeros([6])
i_298 = 0
i_500 = 0
i_600 = 0
i_750 = 0
i_900 = 0
for ind in range(len(Nedat)):
    if Nedat[ind,0] == 298:
      ob_298[i_298] = Nedat[ind,1]
      vol_298[i_298] = Nedat[ind,2]
      i_298 = i_298 + 1
    if Nedat[ind,0] > 499 and Nedat[ind,0] < 502:
      ob_500[i_500] = Nedat[ind,1]
      vol_500[i_500] = Nedat[ind,2]
      i_500 = i_500 + 1
    if Nedat[ind,0] == 600:
      ob_600[i_600] = Nedat[ind,1]
      vol_600[i_600] = Nedat[ind,2]
      i_600 = i_600 + 1
    if Nedat[ind,0] == 750:
      ob_750[i_750] = Nedat[ind,1]
      vol_750[i_750] = Nedat[ind,2]
      i_750 = i_750 + 1
    if Nedat[ind,0] == 900:
      ob_900[i_900] = Nedat[ind,1]
      vol_900[i_900] = Nedat[ind,2]
      i_900 = i_900 + 1

volume1 = np.linspace(0.2,1.05,200)*p_eos[0]
T = np.array([298, 500, 600, 750, 900])

model_298 = MGD_PowerLawShifted(volume1,T[0]*np.ones(volume1.shape),p_eos,Nat)
model_500 = MGD_PowerLawShifted(volume1,T[1]*np.ones(volume1.shape),p_eos,Nat)
model_600 = MGD_PowerLawShifted(volume1,T[2]*np.ones(volume1.shape),p_eos,Nat)
model_750 = MGD_PowerLawShifted(volume1,T[3]*np.ones(volume1.shape),p_eos,Nat)
model_900 = MGD_PowerLawShifted(volume1,T[4]*np.ones(volume1.shape),p_eos,Nat)

plt.plot(model_298,volume1,'k',label = '298 Model')
plt.plot(model_500,volume1,'c',label = '500 Model')
plt.plot(model_600,volume1,'r',label = '600 Model')
plt.plot(model_750,volume1,'m',label = '750 Model')
plt.plot(model_900,volume1,'y',label = '900 Model')


plt.plot(ob_298,vol_298, 'ko',label = '298')
plt.plot(ob_500,vol_500, 'co',label = '500')
plt.plot(ob_600,vol_600,  'ro',label = '600')
plt.plot(ob_750,vol_750, 'mo',label = '750')
plt.plot(ob_900,vol_900,  'yo',label = '900')
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.xlabel('Pressure [GPa]')


plt.legend()
plt.show()

test298 = MGD_PowerLawShifted(vol_298,T[0]*np.ones(vol_298.shape),p_eos,1)
#print "vol_298",vol_298
#print test298
print test298 - ob_298
#print model_500 - ob_500
"""
#test Dr Wolf's table
#volume = np.array([146.59, 145.81, 144.97, 144.32, 146.35, 131.26,142.52,133.96,125.42,133.86,133.91,133.71,125.42,125.40,124.05])
#temperature = np.array([300,300,300,300,1700,300,1924,2375,2020,1755,1780,1740,2228,2240,2045])
#p_eos_0Fe = np.array([162.12,262.3,4.044,1000,1.675,1.39])
#print (MGD_Vinet(volume, temperature, p_eos1,20))

#plot 13% Fe
"""
p_eos = np.array([163.16,243.8,4.160,1000,1.400,0.56])
Pvdat = np.loadtxt(fname='Fe_13.md', delimiter='|', skiprows=3)
temp_300 = np.zeros([49])
vol_300 = np.zeros([49])
ob_300 = np.zeros([49])
i_300 = 0
for ind in range(len(Pvdat)):
    if Pvdat[ind,0] == 300:
      temp_300[i_300] = Pvdat[ind,0]
      ob_300[i_300] = Pvdat[ind,1]
      vol_300[i_300] = Pvdat[ind,2]
      i_300 = i_300 + 1
#p_300 = MGD_Vinet(vol_300, temp_300,p_eos_13Fe,20)
plt.plot(ob_300,vol_300, 'ko',label = '300')
volume1 = np.linspace(0.2,1.05,200)*p_eos[0]
temp1 = np.zeros(len(volume1))
model_300 = MGD_PowerLaw(volume1,temp1+300,p_eos,20)
print(model_300)
plt.plot(model_300,volume1,'k',label = '300 Model')
plt.xlim([20,140])
plt.ylim([122,155])
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.xlabel('Pressure [GPa]')
plt.legend()
plt.show()
"""



####color plot Wolf's PVTMgPvTange.txt
"""
Pvdat = np.loadtxt(fname='PVTMgPvTange.txt', skiprows=1)
volume = Pvdat[:,5]
experiment_P = Pvdat[:,1]
p_eos = np.array([162.12,262.3,4.044,1000,1.675,1.39])
volume1 = np.linspace(0.2,1.05,200)*p_eos[0]
T = np.array([300,500,700,900,1700,1900,2100,2300,2500])
model_P_300 = MGD_PowerLaw(volume1,T[0]*np.ones(volume1.shape),p_eos,20)
model_P_500 = MGD_PowerLaw(volume1,T[1]*np.ones(volume1.shape),p_eos,20)
model_P_700 = MGD_PowerLaw(volume1,T[2]*np.ones(volume1.shape),p_eos,20)
model_P_900 = MGD_PowerLaw(volume1,T[3]*np.ones(volume1.shape),p_eos,20)
model_P_1700 = MGD_PowerLaw(volume1,T[4]*np.ones(volume1.shape),p_eos,20)
model_P_1900 = MGD_PowerLaw(volume1,T[5]*np.ones(volume1.shape),p_eos,20)
model_P_2100 = MGD_PowerLaw(volume1,T[6]*np.ones(volume1.shape),p_eos,20)
model_P_2300 = MGD_PowerLaw(volume1,T[7]*np.ones(volume1.shape),p_eos,20)
model_P_2500 = MGD_PowerLaw(volume1,T[8]*np.ones(volume1.shape),p_eos,20)

plt.ylabel('Volume[' r'$A^{3}$'']')
plt.xlabel('Pressure [GPa]')

plt.clf()
cmap = plt.get_cmap('gist_rainbow')
plt.scatter(experiment_P,volume,30,Pvdat[:,3],'o',cmap=cmap,label='Pressure')

plt.colorbar(ticks=range(300,2500,500))
plt.clim([300, 2500])
plt.xlim([20,140])
plt.ylim([122,155])
legend = plt.legend(loc='upper right')
plt.legend()
plt.plot(model_P_300,volume1,c = cmap(30))
plt.plot(model_P_500,volume1,c = cmap(50))
plt.plot(model_P_700,volume1,c = cmap(70))
plt.plot(model_P_900,volume1,c = cmap(90))
plt.plot(model_P_1700,volume1,c = cmap(170))
plt.plot(model_P_1900,volume1,c = cmap(190))
plt.plot(model_P_2100,volume1,c = cmap(210))
plt.plot(model_P_2300,volume1,c = cmap(230))
plt.plot(model_P_2500,volume1,c = cmap(250))


plt.show()

"""
####color plot Wolf's PVTMgFePvWolf.txt
"""
Pvdat = np.loadtxt(fname='PVTMgFePvWolf.txt', skiprows=1)
volume = Pvdat[:,5]
experiment_P = Pvdat[:,1]
p_eos = np.array([163.16,243.8,4.160,1000,1.400,0.56])
volume1 = np.linspace(0.2,1.05,200)*p_eos[0]
T = np.array([300,500,700,900,1700,1900,2100,2300,2500])
model_P_300 = MGD_PowerLaw(volume1,T[0]*np.ones(volume1.shape),p_eos,20)
model_P_500 = MGD_PowerLaw(volume1,T[1]*np.ones(volume1.shape),p_eos,20)
model_P_700 = MGD_PowerLaw(volume1,T[2]*np.ones(volume1.shape),p_eos,20)
model_P_900 = MGD_PowerLaw(volume1,T[3]*np.ones(volume1.shape),p_eos,20)
model_P_1700 = MGD_PowerLaw(volume1,T[4]*np.ones(volume1.shape),p_eos,20)
model_P_1900 = MGD_PowerLaw(volume1,T[5]*np.ones(volume1.shape),p_eos,20)
model_P_2100 = MGD_PowerLaw(volume1,T[6]*np.ones(volume1.shape),p_eos,20)
model_P_2300 = MGD_PowerLaw(volume1,T[7]*np.ones(volume1.shape),p_eos,20)
model_P_2500 = MGD_PowerLaw(volume1,T[8]*np.ones(volume1.shape),p_eos,20)
"""
"""
plt.plot(experiment_P,volume,'ko',label = 'experiment')
plt.plot(model_P_300,volume1,'b',label = '300K')
plt.plot(model_P_500,volume1,'g',label = '500K')
plt.plot(model_P_700,volume1,'r',label = '700K')
plt.plot(model_P_900,volume1,'y',label = '900K')
plt.plot(model_P_1700,volume1,'b',label = '1700K')
plt.plot(model_P_1900,volume1,'g',label = '1900K')
plt.plot(model_P_2100,volume1,'r',label = '2100K')
plt.plot(model_P_2300,volume1,'y',label = '2300K')
plt.plot(model_P_2500,volume1,'b',label = '2500K')
#plt.xlim([20,140])
#plt.ylim([122,155])
"""
"""
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.xlabel('Pressure [GPa]')

plt.clf()
"""

"""
plt.scatter(model_P_300,volume1,30,T[0]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_500,volume1,30,T[1]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_700,volume1,30,T[2]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_900,volume1,30,T[3]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_1700,volume1,30,T[4]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_1900,volume1,30,T[5]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_2100,volume1,30,T[6]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_2300,volume1,30,T[7]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
plt.scatter(model_P_2500,volume1,30,T[8]*np.ones(volume1.shape),'-',cmap=cmap,label='Pressure')
"""

###original plotting script
"""
cmap = plt.get_cmap('gist_rainbow')
plt.scatter(experiment_P,volume,30,Pvdat[:,3],'o',cmap=cmap,label='Pressure')

plt.colorbar(ticks=range(300,2500,500))
plt.clim([300, 2500])
plt.xlim([20,140])
plt.ylim([122,155])
legend = plt.legend(loc='upper right')
plt.legend()
plt.plot(model_P_300,volume1,c = cmap(30))
plt.plot(model_P_500,volume1,c = cmap(50))
plt.plot(model_P_700,volume1,c = cmap(70))
plt.plot(model_P_900,volume1,c = cmap(90))
plt.plot(model_P_1700,volume1,c = cmap(170))
plt.plot(model_P_1900,volume1,c = cmap(190))
plt.plot(model_P_2100,volume1,c = cmap(210))
plt.plot(model_P_2300,volume1,c = cmap(230))
plt.plot(model_P_2500,volume1,c = cmap(250))


plt.show()
"""
###test plotting below
"""
cmap = plt.get_cmap('gist_rainbow')

climvals = [300,2500]

plt.colorbar(ticks=range(300,2500,500))
plt.clim(climvals)

plt.xlim([20,140])
plt.ylim([122,155])
legend = plt.legend(loc='upper right')
Tcolbar = np.linspace(climvals[0],climvals[1],len(cmap))
Indcolbar = np.range(0,len(cmap))
plt.scatter(experiment_P,volume,30,Pvdat[:,3],'o',cmap=cmap,label='Pressure')
for ind, Tval in enumerate(T):
  indcmap = np.interp1d(Tcolbar,Indcolbar,Tval,kind='nearest')
  plt.plot(model_P[ind],volume1,c = cmap[indcmap])


plt.plot(model_P_500,volume1,c = cmap(50))
plt.plot(model_P_700,volume1,c = cmap(70))
plt.plot(model_P_900,volume1,c = cmap(90))
plt.plot(model_P_1700,volume1,c = cmap(170))
plt.plot(model_P_1900,volume1,c = cmap(190))
plt.plot(model_P_2100,volume1,c = cmap(210))
plt.plot(model_P_2300,volume1,c = cmap(230))
plt.plot(model_P_2500,volume1,c = cmap(250))


plt.show()

"""
