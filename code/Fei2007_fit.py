#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from MGD_func import MGD_PowerLaw


#define functions here
def lsqfun(p_eos,volume,temperature,Natom,P):
	resid_scl = MGD_PowerLaw(volume, temperature, p_eos,Natom) - P
	return resid_scl

def fitfun(p_eos,volume,temperature,Natom,P,error):
	resid_scl = (MGD_PowerLaw(volume, temperature, p_eos,Natom) - P)/error
	return resid_scl

#V0, gamma0 and theta0 are fixed here.
def lsqfun_Fixed(p_fit, p_fix ,volume,temperature,Natom,P):
	pall = [p_fix[0],p_fit[0],p_fit[1],p_fix[1], p_fix[2],p_fit[2]]
	resid_scl = MGD_PowerLaw(volume, temperature, pall,Natom) - P
	return resid_scl

def lsqfun_Fixedfit(p_fit, p_fix ,volume,temperature,Natom,P,error):
	pall = [p_fix[0],p_fit[0],p_fit[1],p_fix[1], p_fix[2],p_fit[2]]
	resid_scl = (MGD_PowerLaw(volume, temperature, pall,Natom) - P)/error
	return resid_scl
"""
#Au-fitting
Audat = np.loadtxt(fname='Au-Fig-1.md', delimiter='|', skiprows=3)
T_Au = Audat[:,0]
P_Au = Audat[:,1]
V_Au = Audat[:,2]
Natom_Au = 4

plt.plot(P_Au,V_Au,'ro')
plt.xlabel('Pressure[GPa]')
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.show()

#sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
guess = [67.85,167,6.00,170,2.97,0.6]
popt = optimization.leastsq(lsqfun,guess[:],args=(V_Au,T_Au,Natom_Au,P_Au),full_output = 0)
print popt
#(array([  71.20655615,   86.21463461,    8.5687309 ,  169.99994558, -1.70495262,    0.7049154 ]), 1)
"""

"""
#NaCl-B2-Fig-4_fitting
#we read the text
NaCldat = np.loadtxt(fname='NaCl-B2-Fig-4.md', delimiter='|', skiprows=3)
T_NaCl = NaCldat[:,0]
P_NaCl = NaCldat[:,1]
V_NaCl = NaCldat[:,2]
Natom_NaCl = 4

plt.plot(P_NaCl,V_NaCl,'ro')
plt.xlabel('Pressure[GPa]')
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.show()

#sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
guess_fix = [41.35,290,1.7]#V0,theta0 and gamma0
guess = [26.86,5.25,0.5]#K0,Kp and q
popt = optimization.leastsq(lsqfun_Fixed,guess[:],args=(guess_fix[:],V_NaCl,T_NaCl,Natom_NaCl,P_NaCl),full_output = 0)
print popt
#(array([ 26.14377661,   5.33390463,   2.07041529]), 1)
"""
"""
#Neon using Au scale
Ne_Audat = np.loadtxt(fname='Neon-Au-Fig-5.md', delimiter='|', skiprows=4)
T_Ne_Au = Ne_Audat[:,0]
P_Ne_Au = Ne_Audat[:,1]
V_Ne_Au = Ne_Audat[:,2]
NatomNe_Au = 4

plt.plot(P_Ne_Au,V_Ne_Au,'ro')
plt.xlabel('Pressure[GPa]')
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.show()

#sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
guessNe_fix = [88.967,75.1,2.05]#V0,theta0 and gamma0
guessNe = [1.16,8.23,0.6]#K0,Kp and q
popt = optimization.leastsq(lsqfun_Fixed,guessNe[:],args=(guessNe_fix[:],V_Ne_Au,T_Ne_Au,NatomNe_Au,P_Ne_Au),full_output = 0)
print popt
#(array([ 144.64862632,   -2.13616756,    9.9976124 ]), 1)
"""
"""
#Neon
Ne_Ptdat = np.loadtxt(fname='Neon-Pt-Fig-5.md', delimiter='|', skiprows=4)
Ne_Audat = np.loadtxt(fname='Neon-Au-Fig-5.md', delimiter='|', skiprows=4)
Hemley_dat = np.loadtxt(fname='Hemley1989-Neon.md', delimiter='|', skiprows=3)
T_Ne = np.concatenate([Ne_Ptdat[:,0], Ne_Audat[:,0],Hemley_dat[:,0]])
P_Ne = np.concatenate([Ne_Ptdat[:,1], Ne_Audat[:,1],Hemley_dat[:,1]])
HemleyV = (Hemley_dat[:,2]**3)
V_Ne = np.concatenate([Ne_Ptdat[:,2], Ne_Audat[:,2],HemleyV])
NatomNe = 4

report = [88.967,1.16,8.23,75.1,2.05,0.6]
FeiPt_residual = lsqfun(report,Ne_Ptdat[:,2],Ne_Ptdat[:,0],NatomNe,Ne_Ptdat[:,1])
FeiPt_errorbar = np.sqrt(sum(FeiPt_residual*FeiPt_residual)/(len(Ne_Ptdat[:,2])-6))
FeiAu_residual = lsqfun(report,Ne_Audat[:,2],Ne_Audat[:,0],NatomNe,Ne_Audat[:,1])
FeiAu_errorbar = np.sqrt(sum(FeiAu_residual*FeiAu_residual)/(len(Ne_Audat[:,2])-6))
Hemley_residual = lsqfun(report,HemleyV,Hemley_dat[:,0],NatomNe,Hemley_dat[:,1])
Hemley_errorbar = np.sqrt(sum(Hemley_residual*Hemley_residual)/(len(Hemley_dat[:,0])-6))

error = np.concatenate([FeiPt_errorbar*np.ones(len(Ne_Ptdat[:,0])), FeiAu_errorbar*np.ones(len(Ne_Audat[:,0])), Hemley_errorbar*np.ones(len(Hemley_dat[:,0]))])

plt.plot(P_Ne,V_Ne,'ro')
plt.xlabel('Pressure[GPa]')
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.show()

#sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
guessNe_fix = [88.967,75.1,2.05]#V0,theta0 and gamma0
guessNe = [1.16,8.23,0.6]#K0,Kp and q

popt = optimization.leastsq(lsqfun_Fixedfit,guessNe[:],args=(guessNe_fix[:],V_Ne,T_Ne,NatomNe,P_Ne,error),full_output = 1)
#print popt
cov = popt[1]
mean = popt[0]
print mean,cov

V = np.linspace(24,50,100)
a = np.random.multivariate_normal(mean,cov)
para = [guessNe_fix[0], a[0],a[1],guessNe_fix[1],guessNe_fix[2],a[2]]
print para
pressure300 = MGD_PowerLaw(V,300*np.ones(len(V)), para, NatomNe)
pressure1000 = MGD_PowerLaw(V,1000*np.ones(len(V)), para, NatomNe)
pressure2000 = MGD_PowerLaw(V,2000*np.ones(len(V)), para, NatomNe)
plt.figure(facecolor="white")
plt.plot(pressure300,V,label = "300K",color=[0.5,0.5,1])
plt.plot(pressure1000,V,label = "1000K",color=[1,0.5,0.5])
plt.plot(pressure2000,V,label = "2000K",color=[0.5,1,0.5])
for i in range(1,9):
	a = np.random.multivariate_normal(mean,cov)
	para = [guessNe_fix[0], a[0],a[1],guessNe_fix[1],guessNe_fix[2],a[2]]
	#print(a)
	#print V
	pressure300 = MGD_PowerLaw(V,300*np.ones(len(V)), para, NatomNe)
	pressure1000 = MGD_PowerLaw(V,1000*np.ones(len(V)), para, NatomNe)
	pressure2000 = MGD_PowerLaw(V,2000*np.ones(len(V)), para, NatomNe)
	#print pressure300
	#print pressure1700
	plt.plot(pressure300,V,'-',color=[0.5,0.5,1])
	plt.plot(pressure1000,V,'-',color=[1,0.5,0.5])
	plt.plot(pressure2000,V,'-',color=[0.5,1,0.5])

plt.plot(Ne_Ptdat[0:8,1],Ne_Ptdat[0:8,2], 'ob')
plt.plot(Ne_Ptdat[9:14,1],Ne_Ptdat[9:14,2], 'or')
plt.plot(Ne_Audat[:,1],Ne_Audat[:,2], 'ob')
plt.plot(Hemley_dat[:,1],HemleyV, 'ob')


plt.xlabel('Pressure[GPa]',fontsize = 20)
plt.ylabel('Volume[' r'$A^{3}$'']',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(prop={'size':15})
#plt.figure(facecolor="white")
plt.title("Uncertainty Interval of Ne from Fei2007[1]",fontsize = 20)
plt.show()
"""

"""
punc = np.sqrt(np.diag(popt[1]))
print "parameter uncert: ", punc

"""

#Pt
Fei_Ptdat = np.loadtxt(fname='Fei2004-Pt.md', delimiter='|', skiprows=3)
Dewaele_Ptdat = np.loadtxt(fname='Dewaele-Pt.md', delimiter='|', skiprows=3)
T_Pt = np.concatenate([Fei_Ptdat[:,0], Dewaele_Ptdat[:,0]])
P_Pt = np.concatenate([Fei_Ptdat[:,1], Dewaele_Ptdat[:,1]])
FeiV = Fei_Ptdat[:,2]**3
DewaeleV = Dewaele_Ptdat[:,2]*4
V_Pt = np.concatenate([FeiV,DewaeleV])
Natom_Pt = 4

#Fei's reported parameters for Pt
fei_report = [60.38, 277,5.08,230,2.72,0.5]
#get residuals here
Dewaele_residual = lsqfun(fei_report,DewaeleV,Dewaele_Ptdat[:,0],Natom_Pt,Dewaele_Ptdat[:,1])
Dewaele_errorbar = np.sqrt(sum(Dewaele_residual*Dewaele_residual)/(len(Dewaele_Ptdat[:,2])-6))
Fei_residual = lsqfun(fei_report,FeiV,Fei_Ptdat[:,0],Natom_Pt,Fei_Ptdat[:,1])
Fei_errorbar = np.sqrt(sum(Fei_residual*Fei_residual)/(len(Fei_Ptdat[:,0])-6))
#print Fei_residual
print Fei_errorbar

error = np.concatenate([Fei_errorbar*np.ones(len(Fei_Ptdat[:,0])), Dewaele_errorbar*np.ones(len(Dewaele_Ptdat[:,2]))])

plt.plot(P_Pt,V_Pt,'ro')
plt.xlabel('Pressure[GPa]')
plt.ylabel('Volume[' r'$A^{3}$'']')
plt.show()

#sequence of the p_eos: V0, K0, Kp, theta0, gamma0, q
guess = [60.38,277,5.08,230,2.72,0.5]
popt = optimization.leastsq(fitfun,guess[:],args=(V_Pt,T_Pt,Natom_Pt,P_Pt,error), ftol = Fei_errorbar,full_output = 1)
#pt,pcov = optimization.curve_fit(curfitfun, V_Pt, P_Pt, p0 = guess, sigma = error)
cov = popt[1]
mean = popt[0]
#mean = pt
#cov = pcov
print mean, cov
#print(np.sqrt(cov[0][0]))

V = np.linspace(47,65,70)
a = np.random.multivariate_normal(mean,cov)
pressure300 = MGD_PowerLaw(V,300*np.ones(len(V)), a, Natom_Pt)
pressure1473 = MGD_PowerLaw(V,1473*np.ones(len(V)), a, Natom_Pt)
pressure1873 = MGD_PowerLaw(V,1873*np.ones(len(V)), a, Natom_Pt)
plt.figure(facecolor="white")
plt.plot(pressure300,V,label = "300K",color=[0.5,0.5,1])
plt.plot(pressure1473,V,label = "1473K", color=[1,0.5,0.5])
plt.plot(pressure1873,V,label = "1873K",color=[0.5,1,0.5])

for i in range(1,9):
	a = np.random.multivariate_normal(mean,cov)
	pressure300 = MGD_PowerLaw(V,300*np.ones(len(V)), a, Natom_Pt)
	pressure1473 = MGD_PowerLaw(V,1473*np.ones(len(V)), a, Natom_Pt)
	pressure1873 = MGD_PowerLaw(V,1873*np.ones(len(V)), a, Natom_Pt)
	plt.plot(pressure300,V,color=[0.5,0.5,1])
	plt.plot(pressure1473,V,color=[1,0.5,0.5])
	plt.plot(pressure1873,V,color=[0.5,1,0.5])

plt.plot(Dewaele_Ptdat[:,1],DewaeleV,'bx')
plt.plot(Fei_Ptdat[0:6,1],FeiV[0:6],'bx')
plt.plot(Fei_Ptdat[7:12,1],FeiV[7:12],'rx')
plt.plot(Fei_Ptdat[20:,1],FeiV[20:],'gx')



plt.xlabel('Pressure[GPa]',fontsize = 20)
plt.ylabel('Volume[' r'$A^{3}$'']',fontsize = 20)
plt.legend(prop={'size':15})
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([0,135])
plt.ylim([47,61])
plt.title("Uncertainty Interval of Pt from Fei2007[1]",fontsize = 20)
plt.show()


""""
	pressure_0 = MGD_PowerLaw(V_Pt, t[0]*np.ones(V_Pt.shape), a ,Natom_Pt)
	pressure_1 = MGD_PowerLaw(V_Pt, t[1]*np.ones(V_Pt.shape), a ,Natom_Pt)
	pressure_2 = MGD_PowerLaw(V_Pt, t[2]*np.ones(V_Pt.shape), a ,Natom_Pt)
	pressure_3 = MGD_PowerLaw(V_Pt, t[3]*np.ones(V_Pt.shape), a ,Natom_Pt)
	pressure_4 = MGD_PowerLaw(V_Pt, t[4]*np.ones(V_Pt.shape), a ,Natom_Pt)
	pressure_5 = MGD_PowerLaw(V_Pt, t[5]*np.ones(V_Pt.shape), a ,Natom_Pt)
	pressure_6 = MGD_PowerLaw(V_Pt, t[6]*np.ones(V_Pt.shape), a ,Natom_Pt)
"""


#draw random draw by cov matrix (np.)
#calculate isotherm for range of volumes
#interp1d to get volumes at specific pressures
#repeat 1000 times
#end up with a big matirx [1000*pn(the number of specific pressures)] np.quantile to get the middle 68% value -> a vector of lower bound and upper bound of the pressures.
#temperature as an input parameters.