import numpy as np 
import matplotlib.pyplot as plt
from Parameters import *
from math import pi, exp, sqrt, log
import sys

#Helper functions:

def H(a):
	return (H0/(3.085678*10**19)) * sqrt( (omega_b+omega_m)*a**(-3) + omega_r*a**(-4) + omega_lambda ) #Dimensions: 1/s

def n_b(a):
	ro_c = 3*( (H0/(3.085678*10**19)) **2) / (8*pi*G)
	return omega_b*ro_c / (m_H * a**3) #Dimensions: 1/m3

def T_b(a):
	return T_0 / a

def SahaEquation(a):
	coefficient = (1/n_b(a)) * (2*pi*m_e*kBoltzmannSI*T_b(a))**(1.5) / (h**3)  *  exp(-epsilon_0 /(T_b(a) * kBoltzmann))
	return 0.5 * (-coefficient + sqrt(coefficient**2 + 4*coefficient)) #Dimensionless

def phi2(a):
	if T_b(a) > 1000000:
		return 0
	else:
		return 0.448*np.log(epsilon_0/(kBoltzmann*T_b(a))) #Dimensionless

def alpha2(a):
	return 64*pi/sqrt(27*pi) * (alphaFina**2 * h2**2)/(m_e**2 * c) * sqrt(epsilon_0/(kBoltzmann*T_b(a))) * phi2(a) #m3/s

def beta(a):
	return (m_e*kBoltzmannSI*T_b(a) / (2*pi*h2**2))**1.5 * exp(-epsilon_0/(kBoltzmann*T_b(a))) * alpha2(a) #Dimensions: 1/m3

def beta2(a):
	return beta(a) * exp(3*epsilon_0/(4*kBoltzmann*T_b(a))) #Dimensions: 1/m3

def n1s(Xe, a):
	return (1 - Xe)*n_b(a) #1/m3

def LambdaAlpha(Xe, a):
	#¿Qué constante de Planck usar?
	return H(a) * ((3*epsilon_0SI)**3) / ((8*pi)**2 * n1s(Xe, a) * h2**3 * c**3) #Dimensions: 1/s

def C_r(Xe, a):
	return (Lambda21 + LambdaAlpha(Xe, a)) / (Lambda21 + LambdaAlpha(Xe, a) + beta2(a)) #Dimensionless

def XeDerivative(Xe, a):
	return C_r(Xe, a)/(a*H(a)) * (beta(a) * (1 - Xe) - n_b(a)*alpha2(a)*Xe**2) #Dimensionless

def rungeKutta4(x, derivX, t = None, deltat=0.1):
	"""
		Runge-Kutta 4th order numerical ODE, FIXED STEPSIZE.
		- x is a numpy array with all the variables to integrate 
		- derivX is a function that maps each variable in x to its derivative and returns them in a numpy array
		- deltat is the time step to use in the algorithm (defaults to 0.1)
		- **kwargs parameters of derivX function
	"""
	k1 = derivX(x,t)*deltat
	k2 = derivX(x + 0.5*k1, t + deltat/2)*deltat
	k3 = derivX(x + 0.5*k2, t + deltat/2)*deltat
	k4 = derivX(x + k3, t)*deltat
	return x + (k1+2*k2+2*k3+k4)/6.0

#Main program:

#Create two grids for Saha and Peebles redshift intervals:

z1 = np.linspace(2500, 1587.4, 1200) #Valid Saha interval, 100 points, Xe becomes 0.99 at redshift 1587.4 based on Saha's approximation
z2 = np.linspace(1587.4, 65, 100000) #Valid Peebles interval, 100000 points


#Create empty arrays to store intermediate variables for plotting Peebles equation parameters:

XeGrid = np.empty(100000)
n_bGrid = np.empty(100000)
T_bGrid = np.empty(100000)
phi2Grid = np.empty(100000)
alpha2Grid = np.empty(100000)
betaGrid = np.empty(100000)
beta2Grid = np.empty(100000)
n1sGrid = np.empty(100000)
LambdaAlphaGrid = np.empty(100000)
C_rGrid = np.empty(100000)
XeDerivativeGrid = np.empty(100000)

#Initial values for Peebles equation are given from Saha's equation:

XeGrid[0] = SahaEquation(1/(1 + 1587.4))
n_bGrid[0] = n_b(1/(1 + 1587.4))
T_bGrid[0] = T_b(1/(1 + 1587.4))
phi2Grid[0] = phi2(1/(1 + 1587.4))
alpha2Grid[0] = alpha2(1/(1 + 1587.4))
betaGrid[0] = beta(1/(1 + 1587.4))
beta2Grid[0] = beta2(1/(1 + 1587.4))
n1sGrid[0] = n1s(XeGrid[0], 1/(1 + 1587.4))
LambdaAlphaGrid[0] = LambdaAlpha(XeGrid[0], 1/(1 + 1587.4))
C_rGrid[0] = C_r(XeGrid[0], 1/(1 + 1587.4))
XeDerivativeGrid[0] = XeDerivative(XeGrid[0], 1/(1 + 1587.4))

#Runge-Kutta grid to integrate Xe over the Peebles interval:

for i in range(1,len(z2)):
	Xe = XeGrid[i-1]
	Xe = rungeKutta4(Xe, XeDerivative, t = 1/(1+z2[i-1]), deltat = (np.absolute(1/(1+z2[i]) - 1/(1+z2[i-1]))))

	current_a = 1/(1+z2[i])

	XeGrid[i] = Xe 
	n_bGrid[i] = n_b(current_a)
	T_bGrid[i] = T_b(current_a)
	phi2Grid[i] = phi2(current_a)
	alpha2Grid[i] = alpha2(current_a)
	betaGrid[i] = beta(current_a)
	beta2Grid[i] = beta2(current_a)
	n1sGrid[i] = n1s(Xe, current_a)
	LambdaAlphaGrid[i] = LambdaAlpha(Xe, current_a)
	C_rGrid[i] = C_r(Xe, current_a)
	XeDerivativeGrid[i] = XeDerivative(Xe, current_a)

	print("z = " + str(z2[i]), ", Xe = " + str(XeGrid[i]))


#Plotting and saving to .txt:

#First window, plot for Saha and Peebles' equations vs redshift

fig1 = plt.figure()

b = np.zeros(1200+100000)

for i in range(len(z1)):
	b[i] = SahaEquation(1/(1+z1[i]))

for i in range(len(z2)):
	b[i+1200] = SahaEquation(1/(1+z2[i]))

plt.semilogy(z1, b[:1200], color='b')
##plt.plot(z1, b[:100], color='b')
plt.semilogy(z2, XeGrid, color = 'b')
plt.semilogy(z2, b[1200:(1200+100000)], linestyle='--', color='b', linewidth=1)
##plt.plot(z2, b[100:(100+100000)], linestyle='--', color='b', linewidth=1)
plt.xlim(np.max(z1), np.min(z2))
##plt.xlim(np.max(z1), 1550)
#plt.ylim(np.min(XeGrid), 10)
plt.ylim(10**(-4), 1.5)
##plt.ylim(0.98, 1.001)
plt.axvline(1587.4, color = 'red', linewidth=1)
plt.ylabel("Xe")
plt.xlabel("z")

#Save to txt:

np.savetxt("XeGrid.txt", np.concatenate((z1,z2)))
np.savetxt("XeResults.txt", np.concatenate((b[:1200], XeGrid)))

#Second window, non-logarithmic scale:

fig2 = plt.figure()

plt.plot(z1, b[:1200], color='b')
plt.plot(z2, XeGrid, color = 'b')
plt.plot(z2, b[1200:(1200+100000)], linestyle='--', color='b', linewidth=1)
plt.xlim(np.max(z1), np.min(z2))
plt.ylim(0, 1.1)
plt.axvline(1587.4, color = 'red', linewidth=1)
plt.ylabel("Xe")
plt.xlabel("z")

#Third window, plot for everything in Peebles' equation:

fig3 = plt.figure()

plt.subplot(251)
plt.title("n_b")
plt.plot(z2, n_bGrid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(252)
plt.title("T_b")
plt.plot(z2, T_bGrid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(253)
plt.title("phi2")
plt.plot(z2, phi2Grid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(254)
plt.title("alpha2")
plt.plot(z2, alpha2Grid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(255)
plt.title("beta")
plt.plot(z2, betaGrid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(256)
plt.title("beta2")
plt.plot(z2, beta2Grid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(257)
plt.title("n1s")
plt.plot(z2, n1sGrid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(258)
plt.title("LambdaAlpha, Lambda21")
plt.plot(z2, LambdaAlphaGrid)
plt.axhline(Lambda21, color = 'r')
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(259)
plt.title("C_r")
plt.plot(z2, C_rGrid)
plt.xlim(np.max(z2), np.min(z2))

plt.subplot(2,5,10)
plt.title("XeDerivative")
plt.plot(z2, XeDerivativeGrid)
plt.xlim(np.max(z2), np.min(z2))

#####

#plt.figure()

# plt.rc('text', usetex=True)
# plt.title(r"$\Lambda_\alpha$, $\Lambda_{2\rightarrow1}$")
#plt.plot(z2, LambdaAlphaGrid)
# plt.axhline(Lambda21, color = 'r')
# plt.xlim(np.max(z2), np.min(z2))
# plt.xlabel(r"Redshift $z$")
# plt.ylabel(r"Velocidad de la reaccion $s^{-1}$")

#plt.rc('text', usetex=True)
#plt.title(r'Densidad de H en estado fundamental $(n_{1s})$')
#plt.plot(z2, n1sGrid)
#plt.xlim(np.max(z2), np.min(z2))
#plt.xlabel("Redshift (z)")
#plt.ylabel("Densidad $m^{-3}$")

plt.show()
