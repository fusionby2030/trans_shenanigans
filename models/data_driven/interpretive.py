""" 
A purely interpretive model of the pedestal transport. 

Assume that the pedestal transport is given by the following equation:

- partial_t T = partial_x [x Chi partial_x T ] + S_T
- partial_t n = partial_x [ D partial_x n + V n ] + S_n 

Assumption 1: Sources of heat and particle transport are prescribed based on an input parameter, Power_IN [MW]. 

The data used to calculate D, V, and Chi is based on the ELM-Averaged MTANH fit of the pedestal from the JET pedestal database. 
Given the profiles and sources above, the transport coefficients are calculated such that at steady state (d/dt = 0), the profiles are consistent with the input profiles:
This requires solving linear non-homogenous equations for D, V, and Chi.
-> Heat Diffusivity: Chi`  - p(x) Chi = r(x)
    - where p(x) = 1/x + d2T/dx2 / (x dT/dx), r(x) = -S_T(x) / (x dT/dx), and solving with Chi(0.8) = Chi_GB(0.8)
-> Particle Diffusivity: D` - p(x) D = r(x)
    - where p(x) = d2/dx2 n, r(x) = -S_n/(d/dx n) - V - (d/dx V )n / (d/dx n)
    - Assume a given Pinch profile, V(x) = x**5

"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scienceplots 
plt.style.use(['grid', 'science'])

# Define T(x) as the mtanh function
def mtanh(x, h1, h0, s, w, p):
    r = (p - x) / (w / 2)
    return (h1 - h0) / 2 * ((1 + s * r) * (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r)) + 1) + h0

def dmtanhdx(x, h1, h0, s, w, p):
    numerator = (h0 - h1)*(np.exp(8*p/w)*s*w - np.exp(8*x/w)*s*w + 4*np.exp(4*(p+x)/w)*(2*p*s + w - 2*s*x))
    denominator = (np.exp(4*p/w) + np.exp(4*x/w))**2 * w**2
    return numerator/denominator

def d2mtanhdx2(x, h1, h0, s, w, p): 
    numerator = (h0-h1)*16*np.exp(4*(p+x) / w) *(np.exp(4*x/w)*(2*p*s + w + s*w - 2*s*x) + np.exp(4*p/w)*(-2*p*s + (-1+s)*w + 2*s*x))
    denominator = (np.exp(4*p/w) + np.exp(4*x/w))**3 * w**3
    return -numerator/denominator
    
def chi_intraelm(x, chi_0, w, p, c_crash=1.1, scaling_factor=1.0):
    # Gausssian  
    amplitude = c_crash / (np.sqrt(2*np.pi*w**2))
    mu = p - w
    sigma = w 
    u = amplitude*np.exp( - (x - mu)**2 / (2 * sigma**2))*scaling_factor
    return u + chi_0

def d_intraelm(x, d_0, w, p, c_crash=1.0, scaling_factor=1.0): 
    amplitude = c_crash
    mu = p - w
    sigma = w 
    u = amplitude*np.exp( - (x - mu)**2 / (2 * sigma**2))*scaling_factor
    return u + d_0

def pinch_term(x): 
    return x**5

def dpinchdx(x): 
    return 5*x**4

def source(x, A, mu, sigma, scalefactor):
    return scalefactor * A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def p_chi(x, h1, h0, s, w, p_val):
    return (1 / x) + (d2mtanhdx2(x, h1, h0, s, w, p_val) / (x * dmtanhdx(x, h1, h0, s, w, p_val)))

def r_chi(x, h1, h0, s, w, p_val, A, mu, sigma, scalefactor):
    return -source(x, A, mu, sigma, scalefactor) / (x * dmtanhdx(x, h1, h0, s, w, p_val))

def p_d(x, h1, h0, s, w, p_val): 
    return d2mtanhdx2(x, h1, h0, s, w, p_val) / dmtanhdx(x, h1, h0, s, w, p_val)

def r_d(x, h1, h0, s, w, p_val, A, mu, sigma, scalefactor): 
    return - source(x, A, mu, sigma, scalefactor)/dmtanhdx(x, h1, h0, s, w, p_val) - pinch_term(x) - dpinchdx(x)*mtanh(x, h1, h0, s, w, p_val) / dmtanhdx(x, h1, h0, s, w, p_val)

def calculate_pressure(temperature, density): 
    return (temperature*11604)*(density/10.0)*1.38064852 / 1000.0


# Define the Linear non-homogenous ODE
def ode_system(x, y, p, r, p_params, r_params):
    px = p(x, **p_params)
    rx = r(x, **r_params)
    return -px * y + rx


if __name__ == '__main__': 
    # Domain for x
    x_start = 0.8
    x_end = 1.0
    x_eval = np.linspace(x_start, x_end, 200)

    """ CHI DETERMINATION """

    # Parameter values, Consumed from Data
    h1Value = 0.5
    h0Value = 0
    sValue = 0.14
    wValue = 0.0438
    pValue = 0.9834

    steady_state_te = mtanh(x_eval, h1Value, h0Value, sValue, wValue, pValue)

    AValue = 10
    muValue = 0.6
    sigmaValue = 0.1
    scalefactorValue = 0.1

    heat_source    = source(x_eval, AValue, muValue, sigmaValue, scalefactorValue)
    CHI_GB = 3.0
    C_INTRA = 1.1
    p_params = {'h1': h1Value, 'h0': h0Value, 's': sValue, 'w': wValue, 'p_val': pValue}
    r_params = {'h1': h1Value, 'h0': h0Value, 's': sValue, 'w': wValue, 'p_val': pValue, 'A': AValue, 'mu': muValue, 'sigma': sigmaValue, 'scalefactor': scalefactorValue}

    # Solve the ODE using solve_ivp
    chi_solution = solve_ivp(
        ode_system, 
        [x_start, x_end], 
        [CHI_GB], 
        args=(p_chi, r_chi, p_params, r_params),
        t_eval=x_eval,
        method='RK45'
    )

    chi_vals = chi_solution.y[0]
    chi_intraelm_vals = chi_intraelm(x_eval, CHI_GB, wValue, pValue, C_INTRA)

    """ D DETERMINATION """

    h1Value = 3.44
    h0Value = 0.0
    sValue  = 0.03
    wValue  = 0.056
    pValue  = 0.9834

    stead_state_ne = mtanh(x_eval, h1Value, h0Value, sValue, wValue, pValue)

    AValue = 5
    muValue = 1.05
    sigmaValue = 0.02
    scalefactorValue = 0.1

    particle_source    = source(x_eval, AValue, muValue, sigmaValue, scalefactorValue)

    D_BOHM = 0.05

    p_params = {'h1': h1Value, 'h0': h0Value, 's': sValue, 'w': wValue, 'p_val': pValue}
    r_params = {'h1': h1Value, 'h0': h0Value, 's': sValue, 'w': wValue, 'p_val': pValue, 'A': AValue, 'mu': muValue, 'sigma': sigmaValue, 'scalefactor': scalefactorValue}

    # Solve the ODE using solve_ivp
    D_solution = solve_ivp(
        ode_system, 
        [x_start, x_end], 
        [D_BOHM], 
        args=(p_d, r_d, p_params, r_params),
        t_eval=x_eval,
        method='RK45'
    )

    D_vals = D_solution.y[0]
    D_intraelm_vals = d_intraelm(x_eval, D_BOHM, wValue*5, pValue, C_INTRA, scaling_factor=0.5)
    V_intraelm_vals = 1.0*np.ones_like(x_eval)

    """ Plotting """


    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()

    axs[0].plot(x_eval, steady_state_te, label='T(x)', color='blue')
    axs[0].set_title('Temperature [keV]')

    axs[1].plot(x_eval, chi_vals, label='Chi(x)', color='red')
    axs[1].plot(x_eval, chi_intraelm_vals, label=f'Chi_intraelm(x), c_intra={C_INTRA}', color='black')
    axs[1].legend()
    axs[1].set_title(r'$\chi$ Diffusivity  [m$^2$/s]')

    axs[2].plot(x_eval, heat_source, label='S_T(x)', color='green')
    axs[2].set_title('Heat Source [MW / m$^2$]')

    axs[3].plot(x_eval, stead_state_ne, color='blue')
    axs[3].set_title('Density [1e19 e/s]')

    axs[4].plot(x_eval, D_vals, color='red', label='D')
    axs[4].plot(x_eval, D_intraelm_vals, color='black', label='D Intra', ls='--')
    twindv = axs[4].twinx()
    twindv.plot(x_eval, pinch_term(x_eval), label='V')
    twindv.plot(x_eval, V_intraelm_vals, label='V Intra', linestyle='--')
    twindv.legend()
    axs[4].legend()
    axs[4].set_title('Particle Diffusivity, D \nPinch Term, V')

    axs[5].plot(x_eval, particle_source, color='green')
    axs[5].set_title('Particle Source')

    for ax in axs: 
        ax.set_xlabel(r'x ($\rho$ like)')


    fig.suptitle('Pre-ELM \"Steady State\" Transport Profiles \nassuming a perscribed density, temperature, and source profiles')
    plt.show()


