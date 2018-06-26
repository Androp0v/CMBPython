from math import pi, sqrt

#Constants:

H0 = 68.0 #Hubble parameter, km/(s·Mpc), Planck
omega_r = 5.042*10**(-5) #Fixed by CMB temperature today (T = 2.73K)
omega_b = 0.04878 #Baryon density, Planck
omega_m = 0.313 #Physical matter density, Planck
omega_lambda = 1 - (omega_r + omega_b + omega_m) #Adjusted for a flat universe (omega_r + omega_b + omega_m + omega_lambda = 1), dark energy
c = 299792458 #Speed of light
kBoltzmann = 8.6173303*10**(-5) #Boltzmann constant, eV / K
kBoltzmannSI = 1.38064852*10**(-23) #Boltzmann constant, m2 kg s-2 K-1
m_e = 9.10938356*10**(-31) #Electron mass, kg
epsilon_0 = 13.605698 #Hydrogen ionization energy, eV
epsilon_0SI = 2.17896 * 10**(-18) #Hydrogen ionization energy, Joules
m_H = 1.6726219*10**(-27) #Hydrogen mass, kg
G = 6.67408 * 10**(-11) #Gravitational constant, N m2 / kg2
T_0 = 2.725 #Current CMB temperature, K
h = 6.62607004*10**(-34) #Planck constant, m2 kg / s
h2 = 1.054571628 * 10**(-34) #Planck constant / 2*pi, J·s or m2 kg /s
alphaFina = 7.297352568 * 10**(-3) #Fine structure constant, adimensional
e = (1.60217662 * 10**(-19)) #/ (4*pi*8.8541878176*10**(-12)) #Nm2
Lambda21 = 8.227 #1/s
sigma_T = 6.652462 * 10**(-29) #m2
