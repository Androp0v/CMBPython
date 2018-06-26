import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt
from scipy.interpolate import splrep, splev
from Parameters import *

#Functions:

rungeKuttaCoefficient = 1.0/6.0
def rungeKutta4(x, derivX, deltat=0.1, **kwargs):
	"""
		Runge-Kutta 4th order numerical ODE, FIXED STEPSIZE.
		- x is a numpy array with all the variables to integrate
		- derivX is a function that maps each variable in x to its derivative and returns them in a numpy array
		- deltat is the time step to use in the algorithm (defaults to 0.1)
		- **kwargs parameters of derivX function
	"""
	k1 = derivX(x,**kwargs)*deltat
	k2 = derivX(x + 0.5*k1, **kwargs)*deltat
	k3 = derivX(x + 0.5*k2, **kwargs)*deltat
	k4 = derivX(x + k3, **kwargs)*deltat
	return x + rungeKuttaCoefficient*(k1+2*k2+2*k3+k4)

def H(a):
	return H0*sqrt( (omega_b+omega_m)*a**(-3) + omega_r*a**(-4) + omega_lambda)

def nuDerivative(nu,a):
	return 1/(H0*sqrt((omega_m + omega_b)*a + omega_r + omega_lambda*a**4))

def conformalTime(scaleFactorGrid, conformalTimeGrid):
	pass


if __name__ == "__main__":

	#Create the two-part redshift grid and join them:

	zgrid1 = np.linspace(1630,614,200)
	zgrid2 = np.linspace(613,0,300)

	zgrid = np.concatenate((zgrid1, zgrid2), axis = 0)

	#Compute the corresponging scale factor grid:

	scaleFactorGrid = np.asarray([1/(1+z) for z in zgrid])

	#Create another (independent) x = log(a) grid:

	agridB = np.log(np.linspace(10**(-10),1,1000))

	#Compute the conformal time for each grid point:

	conformalTimeGrid = np.empty(200+300)

	j = 0
	for a in scaleFactorGrid:
		atemp = 0
		nu = 0
		deltaa = a/1000
		while atemp < a:
			nu = rungeKutta4(nu,nuDerivative, deltat = deltaa, a = atemp)
			atemp += deltaa

		conformalTimeGrid[j] = nu
		j += 1

	#Convert units to seconds:

	conformalTimeGrid *= 3.085678*10**19

	#Print current conformal time in seconds:

	print(conformalTimeGrid[200+300-1])

	#Use SciPy cubic spline interpolation to obtain a function for conformal time:

	conformalTimeInterpolatedCoefficients = splrep(scaleFactorGrid, conformalTimeGrid)

	def conformalTimeInterpolated(scaleFactors):
		return splev(scaleFactors, conformalTimeInterpolatedCoefficients)

	#Plotting:

	#Create another grid to show interpolation:

	zgrid1plotting = np.linspace(1630,614,2000)
	zgrid2plotting = np.linspace(615,0,3000)

	zgridplotting = np.concatenate((zgrid1plotting, zgrid2plotting), axis = 0)

	scaleFactorGridplotting = np.asarray([1/(1+z) for z in zgridplotting])

	#Call plot functions
	plt.scatter(scaleFactorGrid, conformalTimeGrid, s = 5, zorder = 1)
	plt.plot(scaleFactorGridplotting, conformalTimeInterpolated(scaleFactorGridplotting), linestyle='--', color='black', linewidth = 1 , zorder = 0)
	plt.xlabel("Factor de escala (a)")
	plt.ylabel("Tiempo conformal (η), segundos")
	plt.show()