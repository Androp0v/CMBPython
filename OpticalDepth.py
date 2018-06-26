import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import splrep, splev
from Parameters import *
from math import exp

#Useful functions:

def H(a):
	return (H0/(3.085678*10**19)) * sqrt( (omega_b+omega_m)*a**(-3) + omega_r*a**(-4) + omega_lambda ) #Dimensions: 1/s

#Read the results from Recombination.py:

XeGrid = np.flip(np.loadtxt("XeGrid.txt"), 0)
XeValues = np.flip(np.loadtxt("XeResults.txt"),0)

#Minimum smoothing factor s for interpolation that doesn't crash:

splCoefficients = splrep(XeGrid, XeValues, s = 0.000000001)

#We redefine Xe to a function adjusting bounds to return min/max outside interpolated interval:

def Xe(z):
	if z < 65:
		return splev(65.0, splCoefficients)
	elif z > 2000:
		return splev(2000, splCoefficients)
	else:
		return splev(z, splCoefficients)

def n_e(a):
	rho_c = 3*( (H0/(3.085678*10**19)) **2) / (8*pi*G)
	return Xe(1/a - 1)*omega_b*rho_c / ((1 - Xe(1/a - 1)) * m_H * a**3)

def tauDerivative(tau, a):
	return - n_e(a)*sigma_T*c/(a*H(a))

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

#Setting up grid of redshifts z and corresponding scale factors a and ln(a) = x grids:

gridSize = 10000 #Optimal gridsize = 1000000

zGrid = np.linspace(0, 2200, gridSize)
aGrid = 1/(1 + zGrid)
xGrid = np.log(aGrid)

#Additional grid to store tau values for several redshifts before interpolation:

tauGrid = np.zeros(gridSize)

#Integrate from z = 0 to z = 2200:

tauGrid[0] = 0 #Initial conditions, current tau = 0

for i in range(1,len(zGrid)):
	tau = tauGrid[i-1]
	tau = rungeKutta4(tau, tauDerivative, t = 1/(1+zGrid[i-1]), deltat = (np.absolute(1/(1+zGrid[i]) - 1/(1+zGrid[i-1]))))
	tauGrid[i] = tau

#Now interpolating tau:

splCoefficientsTau = splrep(zGrid, tauGrid, s = 0.0000001)

def tau(a):
	return np.abs(splev(1/a - 1, splCoefficientsTau))

#Define a visibility function:

def visibility(z):
	a = 1/(1+z)
	return -a*H(a)*tauDerivative(None, a)*np.exp(-tau(a))/c



#Plotting:

#Figure 1: Tau as a function of a:
plt.figure()
plt.plot(aGrid, np.abs(tauGrid))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Factor de escala (a)")
plt.ylabel("Profundidad óptica (tau)")

#Figure 2: Visibility as a function of redshift z:
plt.figure()
zGridReduced = np.linspace(0,2200,10000) #Smaller grid for plotting
visibilityGrid = [] #For plotting visibility

for z in zGridReduced:
	visibilityGrid.append(visibility(z))

visibilityGrid = np.asarray(visibilityGrid)

plt.plot(zGridReduced, visibilityGrid)
plt.xlabel("Redshift (z)")
plt.ylabel("Visibility (g)")
plt.xlim(np.max(zGridReduced), np.min(zGridReduced))
plt.axvline(1087.6, color = 'red', linewidth=1)

#Figure 3: Interpolated tau as a function of a, interpolated:
plt.figure()
plt.plot(aGrid, tau(1/(1+zGrid)))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Factor de escala (a)")
plt.ylabel("Profundidad óptica interpolada (tau)")

plt.show()