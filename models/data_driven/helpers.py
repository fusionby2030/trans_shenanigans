import numpy as np 
from dataclasses import dataclass 
from scipy.integrate import solve_ivp
from load_data import FitParams, MachineParameters, KineticProfileStruct
from enum import Enum 


@dataclass 
class Domain: 
    phi_norm: np.ndarray 
    R: np.ndarray
    phi: np.ndarray
    dphi_normdr: np.ndarray

class BCS(Enum): 
    INTRA        = 0 
    INTER        = 1
    INTER_FLUXIN = 2
    INTRA_FLUXIN = 3
    INTER_FIXED  = 4
    INTRA_FIXED  = 5
    INSIDE_FIXED = 6

def calculate_bohmgrybohm_from_jardin(R: np.ndarray, rho: np.ndarray, rho_norm: np.ndarray, T0: np.ndarray, mps: MachineParameters):
    dTdrho = np.gradient(T0, rho_norm) / (rho.max() - rho.min())
    dTdr   = dTdrho*(2*R) # Units of keV / m
    gr  = gyro_radius(T0, mps.BT)
    chi = chi_gb(T0, dTdr, mps.BT)
    return chi, gr

def calculate_bohmgrybohm_from_jardin_dc(xdomain: Domain,  T0: np.ndarray, mps: MachineParameters):
    # R: np.ndarray, rho: np.ndarray, rho_norm: np.ndarray,
    R = xdomain.R
    rho = xdomain.R
    rho_norm = xdomain.phi_norm

    dTdrho = np.gradient(T0, rho_norm) / (rho.max() - rho.min())
    dTdr   = dTdrho*(2*R) # Units of keV / m
    gr  = gyro_radius(T0, mps.BT)
    chi = chi_gb(T0, dTdr, mps.BT)
    return chi, gr



def calculate_bohmdiffusion(T0, mps: MachineParameters):
    # D_BOHM = (1.0/16.0) * (k_b *T) / (e * B)
    kev2j    = 1.60218e-17    # J
    e        = 1.60218e-19    # C
    D_BOHM = (1.0/16.0) * (kev2j*T0) / (e * mps.BT)
    return D_BOHM
def calculate_bohmgyrbohm_particlediffusion(T0: np.ndarray, mps: MachineParameters):
    kev2j    = 1.60218e-17    # J
    e        = 1.60218e-19    # C
    D_BOHM = (1.0/16.0) * (kev2j*T0) / (e * mps.BT)
    gr = gyro_radius(T0, mps.BT)
    rhostar = gr / mps.Rmin # (R - mps.Rmaj)
    return D_BOHM * rhostar
def gyro_radius(Te, BT: float=3.0):
    # Larmor radius in meters, given by thermal velocity over cyclotron frequency
    return 1.07E-4 * np.sqrt(Te) / BT

def chi_gb(Te, dTdr, BT: float=3.0):
    # BT units tesla
    ev2j     = 1.60218e-19    # J
    c        = 3.0E8          # m/s
    mp       = 1.6726219e-27  # kg
    kev2j    = 1.60218e-16    # J
    me       = 9.10938356e-31 # kg
    e        = 1.60218e-19    # C
    alpha_GB = 3.5E-2         # From NUCLEAR FUSION, Vol. 38, No. 7 (1998)
    larmor_rad = np.sqrt(me*Te) / (e*BT)
    larmor_rad = gyro_radius(Te, BT)
    return abs(alpha_GB * (kev2j*Te / (e*BT))*larmor_rad*(dTdr/Te))* 1000.0


@dataclass
class TransParams: 
    D: np.ndarray
    V: np.ndarray
    S_N: np.ndarray
    S_T: np.ndarray
    CHI: np.ndarray
    _C: float 
    ne_fitparams: FitParams
    te_fitparams: FitParams
    bcs: BCS
    heatfluxin: float = 0.0
    particlefluxin: float = 0.0
    particlefluxout: float = 0.0
    te_inner_lim: float = 0.0
    ne_inner_lim: float = 0.0
    ne_outer_lim: float = 0.0
    te_outer_lim: float = 120.0
    dphi_normdr : np.ndarray = np.empty(0)
    mps         : MachineParameters = None
    def _pinch_term(self, x: float) -> float: 
        return x**5
    
    def _dpinchdx(self, x: float) -> float: 
        return 5*x**4

    def _source(self, x: float): 
        return 0.0
    
    def _mtanh(self, x: float, fitparam: FitParams):
        p, h1, h0, s, w = fitparam.p, fitparam.h1, fitparam.h0, fitparam.s, fitparam.w
        r = (p - x) / (w / 2)
        return (h1 - h0) / 2 * ((1 + s * r) * (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r)) + 1) + h0
    
    def _dmtanhdx(self, x: float, fitparam: FitParams):
        p, h1, h0, s, w = fitparam.p, fitparam.h1, fitparam.h0, fitparam.s, fitparam.w
        numerator = (h0 - h1)*(np.exp(8*p/w)*s*w - np.exp(8*x/w)*s*w + 4*np.exp(4*(p+x)/w)*(2*p*s + w - 2*s*x))
        denominator = (np.exp(4*p/w) + np.exp(4*x/w))**2 * w**2
        return numerator/denominator
    
    def _d2mtanhdx2(self, x:float, fitparam: FitParams):
        p, h1, h0, s, w = fitparam.p, fitparam.h1, fitparam.h0, fitparam.s, fitparam.w
        numerator = (h0-h1)*16*np.exp(4*(p+x) / w) *(np.exp(4*x/w)*(2*p*s + w + s*w - 2*s*x) + np.exp(4*p/w)*(-2*p*s + (-1+s)*w + 2*s*x))
        denominator = (np.exp(4*p/w) + np.exp(4*x/w))**3 * w**3
        return -numerator/denominator
    
    def _dphi_norm_dr(self, phi_norm_i: float): 
        # for getting dphi/dr for a given phi_norm
        # R = np.linspace(Rmaj, Rmaj + Rmin, N_POINTS)
        # phi = R^2
        # phi_norm = (phi - phi_min) / (phi_max - phi_min)
        # dphi/dr = 2R / (phi_max - phi_min)
        phi_unormed = phi_norm_i*(self.mps.Rmaj + self.mps.Rmin) + self.mps.Rmaj
        R = np.sqrt(phi_unormed)
        dphi_dR = 2*R
        return dphi_dR
        # return 2*R / (self.mps.Rmaj + self.mps.Rmin)
    
# Define T(x) as the mtanh function
def mtanh(x, h1, h0, s, w, p):
    r = (p - x) / (w / 2)
    return (h1 - h0) / 2 * ((1 + s * r) * (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r)) + 1) + h0

def calculate_pressure(temperature, density): 
    return (temperature*11604)*(density/10.0)*1.38064852 / 1000.0

def T_model(x, T, params: TransParams): 
    dTdx = np.gradient(T, x)
    flux = x*params.CHI*dTdx
    fluxgrad = np.gradient(flux, x)
    dTdt = fluxgrad + params.S_T
    # Boundary conditions, fixed temperature at the boundaries
    if params.bcs == BCS.INTRA:
        dTdt[0]  = 0.0 # params._C*(params.CHI[0]*x[0]*dTdx[0]) 
        dTdt[-1] = 0.0
    elif params.bcs == BCS.INTER:
        dTdt[0] = 0.0
        dTdt[-1] = 0.0
    elif params.bcs == BCS.INTER_FLUXIN:  
        # print(dTdt[0], flux[0], params.fluxin, (params.fluxin- flux[0])/(x[1] - x[0])) # ( # ), (params.fluxin- flux[0])/(x[1] - x[0]))
        # forward difference with params.fluxin
        if T[0] >= params.te_inner_lim: 
            dTdt[0] = 0.0
        else: 
            dTdt[0] = (params.heatfluxin- flux[0]) # / (x[1] - x[0])
        dTdt[-1] = 0.0
    elif params.bcs == BCS.INTRA_FLUXIN: 
        if T[0] <= params.te_inner_lim: 
            dTdt[0] = 0.0
        else: 
            dTdt[0] = (params.heatfluxin- flux[0]) # / (x[1] - x[0])
        dTdt[-1] = 0.0
    elif (params.bcs == BCS.INSIDE_FIXED) or (params.bcs == BCS.INTER_FIXED) or (params.bcs == BCS.INTRA_FIXED): 
        dTdt[0] = 0.0
        dTdt[-1] = 0.0
    return dTdt 

def n_model_old(x, n, params: TransParams): 
    # params: D, V, S_N
    dndx = np.gradient(n, x)*params.dphi_normdr
    flux = params.D*dndx + params.V*n
    fluxgrad = np.gradient(flux, x)*params.dphi_normdr
    dndt = fluxgrad + params.S_N

    # Left boundary (x = 0), assume the loss is proportional to the density
    # This should be against the relatex flux in 
    if params.bcs == BCS.INTRA:
        dndt[0]  = params._C*1000*(params.D[0] * dndx[0] + params.V[0] * n[0])
        dndt[-1] = params._C*3.0*(params.D[-1] * dndx[-1] + params.V[-1] * n[-1])
    elif params.bcs == BCS.INTER:
        dndt[-1]  = -params._C*(params.D[-1] * dndx[-1] + params.V[-1] * n[-1])
        dndt[0] =  -params._C*0.1*(params.D[0] * dndx[0] + params.V[0] * n[0])
    elif params.bcs == BCS.INTER_FLUXIN: 
        if n[0] >= params.ne_inner_lim:
            dndt[0] = 0.0
        else: 
            dndt[0] = (params.particlefluxin - flux[0]) # / (x[1] - x[0])
        if n[-1] >= params.ne_outer_lim: 
            dndt[-1] = 0.0
        else:
            dndt[-1] = (params.particlefluxout - flux[-1])
    elif params.bcs == BCS.INTRA_FLUXIN:
        if n[0] <= params.ne_inner_lim:
            dndt[0] = 0.0
        else: 
            dndt[0] = (params.particlefluxin - flux[0])
        if n[-1] <= params.ne_outer_lim:
            dndt[-1] = 0.0
        else:
            dndt[-1] = (params.particlefluxout - flux[-1])

    


    # An influx would have a negative sign. 
    # print(f"Left Boundary: dndx[0] = {dndx[0]:.4}, D[0] = {params.D[0]:.4}, V[0] = {params.V[0]:.4}, dndt[0] = {dndt[0]:.4}, n[0] = {n[0]:.4}")
    
    # Right boundary (x = L), similar condition
    

    return dndt

def n_model(x, n, params: TransParams): 
    # params: D, V, S_N
    dndx = np.gradient(n, x)
    flux = np.gradient(params.D*dndx + params.V*n, x)
    dndt = flux + params.S_N

    # Left boundary (x = 0), assume the loss is proportional to the density
    # This should be against the relatex flux in 
    if params.bcs == BCS.INTRA:
        dndt[0]  = params._C*1000*(params.D[0] * dndx[0] + params.V[0] * n[0])
        dndt[-1] = params._C*3.0*(params.D[-1] * dndx[-1] + params.V[-1] * n[-1])
    elif params.bcs == BCS.INTER:
        dndt[-1]  = -params._C*(params.D[-1] * dndx[-1] + params.V[-1] * n[-1])
        dndt[0] =  -params._C*0.1*(params.D[0] * dndx[0] + params.V[0] * n[0])
    elif params.bcs == BCS.INTER_FLUXIN: 
        if n[0] >= params.ne_inner_lim:
            dndt[0] = 0.0
        else: 
            dndt[0] = (params.particlefluxin - flux[0]) # / (x[1] - x[0])
        if n[-1] >= params.ne_outer_lim: 
            dndt[-1] = 0.0
        else:
            dndt[-1] = (params.particlefluxout - flux[-1])
    elif params.bcs == BCS.INTRA_FLUXIN:
        if n[0] <= params.ne_inner_lim:
            dndt[0] = 0.0
        else: 
            dndt[0] = (params.particlefluxin - flux[0])
        if n[-1] <= params.ne_outer_lim:
            dndt[-1] = 0.0
        else:
            dndt[-1] = (params.particlefluxout - flux[-1])
    elif params.bcs == BCS.INTER_FIXED: 
        dndt[0] = 0.0
        if n[-1] >= params.ne_outer_lim:
            dndt[-1] = 0.0
        else:
            dndt[-1] = (params.particlefluxout - flux[-1]) 
    elif params.bcs == BCS.INTRA_FIXED: 
        dndt[0] = 0.0
        if n[-1] <= params.ne_outer_lim:
            dndt[-1] = 0.0
        else:
            dndt[-1] = (params.particlefluxout - flux[-1]) 


    # An influx would have a negative sign. 
    # print(f"Left Boundary: dndx[0] = {dndx[0]:.4}, D[0] = {params.D[0]:.4}, V[0] = {params.V[0]:.4}, dndt[0] = {dndt[0]:.4}, n[0] = {n[0]:.4}")
    
    # Right boundary (x = L), similar condition
    

    return dndt


def solve_pde(x, u0, tinterval: list[float], transparams: TransParams, modeltosolve, tkeep=None):
    ode_to_handle = lambda t, y: modeltosolve(x, y, transparams)
    sol = solve_ivp(ode_to_handle, tinterval, u0, t_eval=tkeep, )
    return sol.y 

def normal_distribution(x, mu, sigma, c_crash, baseline=0.0): 
    amplitude = c_crash / (np.sqrt(2*np.pi*sigma**2))
    return amplitude*np.exp( - (x - mu)**2 / (2 * sigma**2)) + baseline


def p_d_old(x, params: TransParams):
    d2ndx2 = params._d2mtanhdx2(x, params.ne_fitparams) 
    dndx = params._dmtanhdx(x, params.ne_fitparams)
    return d2ndx2 / dndx

def r_d_old(x: np.ndarray, params: TransParams): 
    # dndx = np.gradient(density_steady_state, x)
    # dpinchdx = np.gradient(pinch_term, x)
    # return - source_term/dndx - pinch_term - dpinchdx*density_steady_state/dndx
    dndx = params._dmtanhdx(x, params.ne_fitparams)
    dpinchdx = params._dpinchdx(x)
    return - params._source(x)/dndx - params._pinch_term(x) - dpinchdx*params._mtanh(x, params.ne_fitparams)/dndx

def p_d(x, params: TransParams): 
    # Includes the full derivative, dphi_normdr
    dndx = params._dmtanhdx(x, params.ne_fitparams)*params._dphi_norm_dr(x)
    d2ndx2 = params._d2mtanhdx2(x,params.ne_fitparams)*params._dphi_norm_dr(x)
    return d2ndx2 / dndx

def r_d(x, params: TransParams): 
    # Includes the full derivative, dphi_normdr 
    dndx = params._dmtanhdx(x, params.ne_fitparams)*params._dphi_norm_dr(x)
    dpinchdx = params._dpinchdx(x)*params._dphi_norm_dr(x)
    return - params._source(x)/dndx - params._pinch_term(x) - dpinchdx*params._mtanh(x, params.ne_fitparams)/dndx

def p_chi(x, params: TransParams): 
    # dTdx = np.gradient(temperature_steady_state, x)
    # d2Tdx2 = np.gradient(dTdx, x)
    dTdx = params._dmtanhdx(x, params.te_fitparams)
    d2Tdx2 = params._d2mtanhdx2(x, params.te_fitparams)
    return 1/x + d2Tdx2 / (x*dTdx)

def r_chi(x, params: TransParams): 
    # dTdx = np.gradient(temperature_steady_state, x)
    # return -source_term / (x*dTdx)
    dTdx = params._dmtanhdx(x, params.te_fitparams)
    return -params._source(x) / (x*dTdx)

def ode_system(x, y, p_func, r_func, p_params, r_params): 
    px = p_func(x, **p_params)
    rx = r_func(x, **r_params)
    return -px*y + rx 

# 
def get_chi_inter_ped(x, initial_condition, temperature_steady_state, transparams: TransParams):
    ped_bool = x > transparams.te_fitparams.p - transparams.te_fitparams.w
    x_eval = x[ped_bool]
    scaled_solution = np.ones_like(x)*initial_condition
    # Only solve for x > te_fit_params.p - te_fit_params.w
    solution =  solve_ivp(
        ode_system, 
        [x_eval[0], x_eval[-1]],
        [initial_condition],
        args = (p_chi, r_chi, {'params': transparams}, {'params': transparams}),
        t_eval=x_eval,
        method='RK45'
    )
    factor = lambda phi_norm, c_scale: normal_distribution(phi_norm, transparams.te_fitparams.p, transparams.te_fitparams.w / 2.0, c_scale, 1.0) 
    ped_scaled_solution = solution.y[0] / factor(x_eval, transparams._C)
    scaled_solution[ped_bool] = ped_scaled_solution
    return scaled_solution

def get_chi_inter(x, initial_condition, temperature_steady_state, transparams: TransParams):
    # Only solve for x > te_fit_params.p - te_fit_params.w
    solution =  solve_ivp(
        ode_system, 
        [x[0], x[-1]],
        [initial_condition],
        args = (p_chi, r_chi, {'params': transparams}, {'params': transparams}),
        t_eval=x,
        method='RK45'
    )
    factor = lambda phi_norm, c_scale: normal_distribution(phi_norm, transparams.te_fitparams.p, transparams.te_fitparams.w / 2.0, c_scale, 1.0) 
    if transparams._C > 0.0: 
        scaled_solution = solution.y[0] / factor(x, transparams._C)
    else: 
        scaled_solution = solution.y[0] * factor(x, abs(transparams._C))
    return scaled_solution

def get_chi_inter_unscaled(x, initial_condition, temperature_steady_state, transparams: TransParams):
    solution =  solve_ivp(
        ode_system, 
        [x[0], x[-1]],
        [initial_condition],
        args = (p_chi, r_chi, {'params': transparams}, {'params': transparams}),
        t_eval=x,
        method='RK45'
    )
    return solution.y[0]

def get_d_inter_ped(x, initial_condition, density_steady_state, transparams: TransParams):
    ped_bool = x > transparams.ne_fitparams.p - transparams.ne_fitparams.w
    x_eval = x[ped_bool]
    scaled_solution = np.ones_like(x)*initial_condition
    # Only solve for x > te_fit_params.p - te_fit_params.w
    solution =  solve_ivp(
        ode_system, 
        [x_eval[0], x_eval[-1]],
        [initial_condition],
        args = (p_d, r_d, {'params': transparams}, {'params': transparams}),
        t_eval=x_eval,
        method='RK45'
    )
    factor = lambda phi_norm, c_scale: normal_distribution(phi_norm, transparams.ne_fitparams.p, transparams.ne_fitparams.w / 2.0, c_scale, 1.0)
    ped_scaled_solution = solution.y[0] / factor(x_eval, transparams._C)
    scaled_solution[ped_bool] = ped_scaled_solution
    return scaled_solution

def get_d_inter_unscaled(x, initial_condition, density_steady_state, transparams: TransParams): 
    solution = solve_ivp(
        ode_system, 
        [x[0], x[-1]],
        [initial_condition],
        args = (p_d, r_d, {'params': transparams}, {'params': transparams}),
        t_eval=x,
        method='RK45'
    )
    return solution.y[0]

def get_d_inter(x, initial_condition, density_steady_state, transparams: TransParams): 
    solution = solve_ivp(
        ode_system, 
        [x[0], x[-1]],
        [initial_condition],
        args = (p_d, r_d, {'params': transparams}, {'params': transparams}),
        t_eval=x,
        method='RK45'
    )
    factor = lambda phi_norm, c_scale: normal_distribution(phi_norm, transparams.ne_fitparams.p, transparams.ne_fitparams.w / 2.0, c_scale, 1.0) 
    if transparams._C > 0.0: 
        scaled_solution = solution.y[0] / factor(x, transparams._C)
    else: 
        scaled_solution = solution.y[0] * factor(x, abs(transparams._C))
    return scaled_solution

def solve_time_evolution(initial_density, initial_temperature, inter_params: TransParams, intra_params: TransParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    # Returns time evolution
    # t, density, temperature, pressure
    t_interval = [0.0, 1.0]
    tau_intraelm = 200E-6
    num_time_steps = int((t_interval[1] - t_interval[0]) / (tau_intraelm / 2.0))
    t_keep     = np.linspace(t_interval[0], t_interval[1], num_time_steps)

    psi_eval = np.linspace(0.8, 1.0, 100)
    temperature_profiles_all = np.empty((num_time_steps, len(initial_temperature)))
    density_profiles_all     = np.empty((num_time_steps, len(initial_temperature)))

    temperature_profiles_all[0] = initial_temperature
    density_profiles_all[0]     = initial_density

    # Start with interelm 
    params = inter_params

    t0 = initial_temperature
    n0 = initial_density

    ALPHA_CRIT = 90.0
    ELM_ACTIVE = False 
    T_LAST_ELM = 0.0 

    for i in range(num_time_steps-1): 
        t_init, t_end = t_keep[i], t_keep[i+1]
        t_interval = [t_init, t_end]

        solutions_intraelm_te = solve_pde(psi_eval, t0, t_interval, params, T_model)
        solutions_intraelm_ne = solve_pde(psi_eval, n0, t_interval, params, n_model)

        temperature_profiles_all[i+1] = solutions_intraelm_te[:, -1]
        density_profiles_all[i+1]     = solutions_intraelm_ne[:, -1]
        t0 = solutions_intraelm_te[:, -1]
        n0 = solutions_intraelm_ne[:, -1]

        p0 = calculate_pressure(t0, n0)

        # Check pressure gradient condition 

        alpha_exp = max(abs(np.gradient(p0, psi_eval)))
        if (alpha_exp > ALPHA_CRIT) and ELM_ACTIVE is False:
            params = intra_params
            ELM_ACTIVE = True

        if ELM_ACTIVE is True and T_LAST_ELM >= tau_intraelm: 
            T_LAST_ELM = 0.0
            ELM_ACTIVE = False 
            params =  inter_params
        
        # elif ELM_ACTIVE is True and T_LAST_ELM <= tau_intraelm: 
        T_LAST_ELM += (t_end - t_init)

        print(t_end, ELM_ACTIVE, T_LAST_ELM, alpha_exp)
    pressure_profiles_all = calculate_pressure(temperature_profiles_all, density_profiles_all)

    return t_keep, density_profiles_all, temperature_profiles_all, pressure_profiles_all

from scipy.signal import find_peaks

def calculate_elm_frequency_from_normalised_alpha(alpha_exp_norm: np.ndarray, t: np.ndarray) -> float: 
    peaks, _ = find_peaks(alpha_exp_norm, height=0.999)
    # only look at the peaks in the last half 
    idx_of_interest = np.argmin(abs(t - t[-1]/2))
    peaks_of_interest = peaks[peaks > idx_of_interest]
    elm_frequency = len(peaks_of_interest) / (t[-1] - t[-1]/2)
    return peaks, elm_frequency

from tqdm import tqdm 

def do_mll_estimation_over_crash(collected_data: list[tuple[float, TransParams, np.ndarray, np.ndarray, np.ndarray]], 
                                 xdomain: Domain, 
                                 t_minus_t_elm_expdata: np.ndarray, 
                                 relevant_profiles: KineticProfileStruct, exp_profiles_offset: float, spacing: float) -> tuple[float, tuple[np.ndarray, np.ndarray]]: 
    windows = [(x, x + spacing) for x in np.arange(0.8, 1.0, spacing)]
    window_likelihoods = np.zeros((len(collected_data), 3, len(windows)))
    log_liklihood = lambda x_i, mu, sigma: -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (x_i - mu)**2 / sigma**2

    for i, t in enumerate(t_minus_t_elm_expdata*1000):
        if t > 1.5 or t < 0.0: 
            continue 

        for k, window in enumerate(windows): 
            window_bool = np.logical_and(relevant_profiles.hrts_psi[i] + exp_profiles_offset > window[0], relevant_profiles.hrts_psi[i] + exp_profiles_offset < window[1])
            window_ne = relevant_profiles.ne[i, window_bool]*1E-19
            window_te = relevant_profiles.te[i, window_bool]/1000.0
            window_pe = calculate_pressure(window_te, window_ne)

            for c, collect in enumerate(collected_data): 
                C_INTRA, INTRA_PARAMS, solutions_intraelm_te, solutions_intraelm_ne, solutions_intraelm_pe = collect
                sim_window_bool = np.logical_and(xdomain.phi_norm > window[0], xdomain.phi_norm < window[1])
                sim_window_ne = solutions_intraelm_ne[sim_window_bool, -1]
                sim_window_te = solutions_intraelm_te[sim_window_bool, -1]
                sim_window_pe = solutions_intraelm_pe[sim_window_bool, -1]
                sim_window_ne_mean = np.mean(sim_window_ne)
                sim_window_te_mean = np.mean(sim_window_te)
                sim_window_pe_mean = np.mean(sim_window_pe)
                sim_window_ne_std = np.std(sim_window_ne)
                sim_window_te_std = np.std(sim_window_te)
                sim_window_pe_std = np.std(sim_window_pe)

                # Liklihood calculation
                ne_likelihood = np.sum([log_liklihood(x_i, sim_window_ne_mean, sim_window_ne_std) for x_i in window_ne])
                te_likelihood = np.sum([log_liklihood(x_i, sim_window_te_mean, sim_window_te_std) for x_i in window_te])
                pe_likelihood = np.sum([log_liklihood(x_i, sim_window_pe_mean, sim_window_pe_std) for x_i in window_pe])

                window_likelihoods[c, 0, k] += ne_likelihood
                window_likelihoods[c, 1, k] += te_likelihood
                window_likelihoods[c, 2, k] += pe_likelihood

    window_likelihoods /= len(t_minus_t_elm_expdata)

    return window_likelihoods 

def setup_domain(mps: MachineParameters) -> Domain: 
    # Setup Domain 
    N_POINTS    = 1000
    R           = np.linspace(mps.Rmaj, mps.Rmaj + mps.Rmin, N_POINTS)
    phi         = R**2
    phi_norm    = (phi - phi.min()) / (phi.max() - phi.min())
    pedestal_region =  phi_norm > 0.65
    phi_norm    = phi_norm[pedestal_region]
    phi         = phi[pedestal_region]
    R           = R[pedestal_region]
    dphi_normdr = 2*R / (phi.max() - phi.min())

    xdomain = Domain(phi_norm, R, phi, dphi_normdr)
    return xdomain

def setup_intra_params(C_CRASH: float, xdomain: Domain, te_fit_params: FitParams, ne_fit_params: FitParams, chi_gb: float, dbohm: float) -> TransParams:
    scale_tewidth = 1.0
    scale_newidth = 1.25
    scale_chi0_intra = 2.0
    scale_d0_intra = 5.0
    CHI0_INTRA = chi_gb 
    D0_INTRA = dbohm

    _INTRA_PARAMS = TransParams(
        CHI=normal_distribution(xdomain.phi_norm, te_fit_params.p - te_fit_params.w, te_fit_params.w*scale_tewidth, C_CRASH, CHI0_INTRA*scale_chi0_intra), 
        D=normal_distribution(xdomain.phi_norm, ne_fit_params.p - ne_fit_params.w, ne_fit_params.w*scale_newidth, C_CRASH , D0_INTRA*scale_d0_intra), 
        V=np.ones_like(xdomain.phi_norm),
        S_N=np.zeros_like(xdomain.phi_norm),
        S_T=np.zeros_like(xdomain.phi_norm),
        _C = C_CRASH,
        ne_fitparams=ne_fit_params, 
        te_fitparams=te_fit_params, 
        bcs=BCS.INTRA_FIXED, 
        dphi_normdr=xdomain.dphi_normdr
        )
    _INTRA_PARAMS.particlefluxout = (_INTRA_PARAMS.D[-1])*np.gradient(ne_fit_params.steady_state_arr, xdomain.phi_norm)[-1]*xdomain.dphi_normdr[-1] + _INTRA_PARAMS.V[-1]*ne_fit_params.steady_state_arr[-1]
    _INTRA_PARAMS.ne_outer_lim = 0.0
    return _INTRA_PARAMS

def single_c_crash(intra_params: TransParams, current_te: np.ndarray, current_ne: np.ndarray): 

    pass 

def find_optimal_c_crash(xdomain: Domain, 
                         te_fit_params: FitParams, ne_fit_params: FitParams, 
                         CHI0_INTRA: float, D0_INTRA: float, 
                         t_minus_t_elm_expdata: np.ndarray, 
                         relevant_profiles: KineticProfileStruct, 
                         exp_profiles_offset: float ) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    scale_tewidth = 1.0
    scale_newidth = 1.25
    scale_chi0_intra = 2.0
    scale_d0_intra = 5.0

    max_c_crash = 6.0
    min_c_crash = 0.01
    tau_intraelm = 200E-6 # 200 us
    t_interval = [0, tau_intraelm]
    t_keep     = np.linspace(0, tau_intraelm, 10)

    collected = []
    scan_crashes = np.linspace(min_c_crash, max_c_crash, 30)

    for C_CRASH in tqdm(scan_crashes): 
        # Setup INTRA PARAMS 
        _INTRA_PARAMS = TransParams(
        CHI=normal_distribution(xdomain.phi_norm, te_fit_params.p - te_fit_params.w, te_fit_params.w*scale_tewidth, C_CRASH, CHI0_INTRA*scale_chi0_intra), 
        D=normal_distribution(xdomain.phi_norm, ne_fit_params.p - ne_fit_params.w, ne_fit_params.w*scale_newidth, C_CRASH , D0_INTRA*scale_d0_intra), 
        V=np.ones_like(xdomain.phi_norm),
        S_N=np.zeros_like(xdomain.phi_norm),
        S_T=np.zeros_like(xdomain.phi_norm),
        _C = C_CRASH,
        ne_fitparams=ne_fit_params, 
        te_fitparams=te_fit_params, 
        bcs=BCS.INTRA_FIXED, 
        dphi_normdr=xdomain.dphi_normdr
        )

        _INTRA_PARAMS.particlefluxout = (_INTRA_PARAMS.D[-1])*np.gradient(ne_fit_params.steady_state_arr, xdomain.phi_norm)[-1]*xdomain.dphi_normdr[-1] + _INTRA_PARAMS.V[-1]*ne_fit_params.steady_state_arr[-1]
        _INTRA_PARAMS.ne_outer_lim = 0.0

        solutions_intraelm_te = solve_pde(xdomain.phi_norm, te_fit_params.steady_state_arr, t_interval, _INTRA_PARAMS, T_model, t_keep)
        solutions_intraelm_ne = solve_pde(xdomain.phi_norm, ne_fit_params.steady_state_arr, t_interval, _INTRA_PARAMS, n_model, t_keep)
        solutions_intraelm_pe = calculate_pressure(solutions_intraelm_te, solutions_intraelm_ne)
        collect = (C_CRASH, _INTRA_PARAMS, solutions_intraelm_te, solutions_intraelm_ne, solutions_intraelm_pe)
        collected.append(collect)
    
    # Do MLL estimation 
    spacing = 0.025 # NOTE: this should be increased if the nothing is found...
    window_likelihoods = do_mll_estimation_over_crash(collected, xdomain, t_minus_t_elm_expdata, relevant_profiles, exp_profiles_offset, spacing)

    neg_sumover_windows_and_netepe = -np.sum(window_likelihoods, axis=(1, 2))
    
    best_c_crash_idx = np.argmin(neg_sumover_windows_and_netepe)
    best_c_crash = scan_crashes[best_c_crash_idx]

    data_from_best_c_crash = collected[best_c_crash_idx]

    return best_c_crash, (scan_crashes, neg_sumover_windows_and_netepe), data_from_best_c_crash

def get_c_inter_over_time(t: float, cinter: float, cinter_initial: float=-0.05):
    return cinter_initial + (cinter + cinter_initial)*t

def calculate_maxalpha(pressure: np.ndarray, xdomain: Domain) -> np.ndarray: 
    in_pedestal = xdomain.phi_norm > 0.94
    return np.max(abs(np.gradient(pressure[in_pedestal], xdomain.phi_norm[in_pedestal]))) 

def setup_inter_params(xdomain: Domain, te_fit_params: FitParams, ne_fit_params: FitParams, machineparams: MachineParameters, CHI0_INTER: float, D0_INTER) -> TransParams:
    _fake_cgrowth = 3.0
    INTER_PARAMS = TransParams(
        CHI=np.empty_like(xdomain.phi_norm), 
        D=np.empty_like(xdomain.phi_norm), 
        V=xdomain.phi_norm**5, 
        S_N=np.zeros_like(xdomain.phi_norm),
        S_T=np.zeros_like(xdomain.phi_norm),
        _C = get_c_inter_over_time(0.0, _fake_cgrowth),
        ne_fitparams=ne_fit_params, 
        te_fitparams=te_fit_params, 
        bcs=BCS.INTER_FIXED, 
        dphi_normdr=xdomain.dphi_normdr,
        mps = machineparams
    )
    INTER_PARAMS.particlefluxout = -((D0_INTER)*np.gradient(ne_fit_params.steady_state_arr, xdomain.phi_norm)[-1]*xdomain.dphi_normdr[-1] + INTER_PARAMS.V[-1]*ne_fit_params.steady_state_arr[-1])
    INTER_PARAMS.ne_outer_lim = ne_fit_params.steady_state_arr[-1] 
    return INTER_PARAMS

def single_c_growth_run_from_postelm_crash(xdomain: Domain, C_GROWTH: float,
                                           post_elm_ne: np.ndarray, post_elm_te: np.ndarray, 
                                           te_fit_params: FitParams, ne_fit_params: FitParams, 
                                           CHI0_INTER: float, D0_INTER: float, machineparams: MachineParameters, 
                                           printout: bool = False) -> float: 
    
    INTER_PARAMS = TransParams(
        CHI=np.empty_like(xdomain.phi_norm), 
        D=np.empty_like(xdomain.phi_norm), 
        V=xdomain.phi_norm**5, 
        S_N=np.zeros_like(xdomain.phi_norm),
        S_T=np.zeros_like(xdomain.phi_norm),
        _C = get_c_inter_over_time(0.0, C_GROWTH),
        ne_fitparams=ne_fit_params, 
        te_fitparams=te_fit_params, 
        bcs=BCS.INTER_FIXED, 
        dphi_normdr=xdomain.dphi_normdr,
        mps = machineparams
    )

    tau_interelm = 0.0001
    in_pedestal = xdomain.phi_norm > 0.94

    steady_state_pe = calculate_pressure(te_fit_params.steady_state_arr, ne_fit_params.steady_state_arr)
    ALPHA_CRIT = np.max(abs(np.gradient(steady_state_pe[in_pedestal], xdomain.phi_norm[in_pedestal])))
    current_ne = post_elm_ne
    current_te = post_elm_te
    current_pe = calculate_pressure(current_te, current_ne)

    current_time = 0.0
    alpha_exp = np.max(abs(np.gradient(current_pe, xdomain.phi_norm)))
    collected = []
    while True: 
        if alpha_exp >= ALPHA_CRIT:
            break 
        dt = tau_interelm
        t_internal = [current_time, current_time + dt]
        
        INTER_PARAMS._C  = get_c_inter_over_time(current_time, C_GROWTH)
        INTER_PARAMS.CHI = get_chi_inter(xdomain.phi_norm, CHI0_INTER, te_fit_params.steady_state_arr, INTER_PARAMS)
        INTER_PARAMS.D   = get_d_inter(xdomain.phi_norm, D0_INTER, ne_fit_params.steady_state_arr, INTER_PARAMS)
        
        INTER_PARAMS.particlefluxout = -((D0_INTER)*np.gradient(ne_fit_params.steady_state_arr, xdomain.phi_norm)[-1]*xdomain.dphi_normdr[-1] + INTER_PARAMS.V[-1]*ne_fit_params.steady_state_arr[-1])
        INTER_PARAMS.ne_outer_lim = ne_fit_params.steady_state_arr[-1] 

        solutions_interelm_te = solve_pde(xdomain.phi_norm, current_te, t_internal, INTER_PARAMS, T_model)
        solutions_interelm_ne = solve_pde(xdomain.phi_norm, current_ne, t_internal, INTER_PARAMS, n_model)

        current_te = solutions_interelm_te[:, -1]
        current_ne = solutions_interelm_ne[:, -1]
        current_pe = calculate_pressure(current_te, current_ne)
        alpha_exp = np.max(abs(np.gradient(current_pe, xdomain.phi_norm)))
        current_time += dt
        # Check if alpha_exp is greater than ALPHA_CRIT
        if printout: 
            print(f"Current Time: {current_time:.4}, Alpha Exp: {alpha_exp:.4}, Alpha Crit: {ALPHA_CRIT:.4}, C_INTER: {INTER_PARAMS._C:.4}")
        collect = (current_time, current_te, current_ne, current_pe)
        collected.append(collect)

    all_times = np.empty(len(collected))
    all_te = np.empty((len(collected), len(xdomain.phi_norm)))
    all_ne = np.empty((len(collected), len(xdomain.phi_norm)))
    all_pe = np.empty((len(collected), len(xdomain.phi_norm)))
    for nt, collect in enumerate(collected): 
        time, te, ne, pe = collect
        all_times[nt] = time
        all_te[nt] = te
        all_ne[nt] = ne
        all_pe[nt] = pe
    colection = (all_times, all_te, all_ne, all_pe)
    return current_time, colection

def find_optimal_c_growth(tau_interelm_from_exp: float,
                          xdomain: Domain,
                          post_elm_ne: np.ndarray, post_elm_te: np.ndarray, 
                          te_fit_params: FitParams, ne_fit_params: FitParams, 
                          CHI0_INTER: float, D0_INTER: float, machineparams: MachineParameters, 
                          ) -> tuple[float, tuple[np.ndarray, np.ndarray]]:

    max_c_growth = 5.0
    min_c_growth = 1.0

    scan_growths = np.linspace(min_c_growth, max_c_growth, 20)
    tau_interelm_from_sim = np.empty_like(scan_growths)
    collection = []
    for n, C_GROWTH in enumerate(tqdm(scan_growths)): 
        time_to_alpha_crit, collect = single_c_growth_run_from_postelm_crash(xdomain, C_GROWTH, post_elm_ne.copy(), post_elm_te.copy(), te_fit_params, ne_fit_params, CHI0_INTER, D0_INTER, machineparams)
        tau_interelm_from_sim[n] = time_to_alpha_crit    
        collection.append(collect)
    # find c_growth most similar to tau_interelm_from_exp
    best_c_growth_idx = np.argmin(abs(tau_interelm_from_sim - tau_interelm_from_exp))
    best_c_growth = scan_growths[best_c_growth_idx]

    return best_c_growth, (scan_growths, tau_interelm_from_sim), collection[best_c_growth_idx]

