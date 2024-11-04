import jax.numpy as jnp
from load_data import FitParams
from scipy.integrate import solve_ivp 
from dataclasses import dataclass
from load_data import MachineParameters
import numpy as np

@dataclass
class TransParams: 
    D: np.ndarray
    V: np.ndarray
    S_N: np.ndarray
    n_fitparams: FitParams
    t_fitparams: FitParams
    machine_params: MachineParameters
    C_ETB: float
    C_CRASH: float
    _CURRENT_C_ETB: float = 0.0
    def _pinch_term(self, x: float) -> float: 
        return x**5
    
    def _dpinchdx(self, x: float) -> float: 
        return 5*x**4

    def _source(self, x: float, fitparam: FitParams, density: np.ndarray = None, cetb: float = None): 
        # 1 / (n ** C_ETB)
        # return 0.0

        if cetb is not None: 
            m_parameter = 0.7 / cetb
            k_parameter = 10.0*cetb
        else: 
            m_parameter = 0.2 # / self.C_ETB
            k_parameter = 9.0# *self.C_ETB
        if density is None: 
            density = self._mtanh(x, fitparam)
        # S = 1 / ((density ** (self.C_ETB)))*1.0e2
        # scale = 100e2
        # scaling_factor = np.exp(-k_parameter*(1.0 - x)**m_parameter)

        # scaling_factor = (1 + k_parameter*(1.0 - x)**m_parameter)**-1
        # Scaling factor, as ne is in the order of 1e19
        # and S is on order of 1E21
        S = self.C_ETB*np.exp(x / density)
        return S # (S * scaling_factor)
    
    def _mtanh(self, x: float, fitparam: FitParams):
        p, h1, h0, s, w = fitparam.p, fitparam.h1, fitparam.h0, fitparam.s, fitparam.w
        r = (p - x) / (w / 2)
        return (h1 - h0) / 2 * ((1 + s * r) * (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r)) + 1) + h0
    
    @staticmethod
    def mtanh(x: float, fitparam: FitParams):
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
    
def ode_system(x, y, p_func, r_func, p_params, r_params): 
    px = p_func(x, **p_params)
    rx = r_func(x, **r_params)
    return -px*y + rx 

def p_func_density(x, params: TransParams): 
    dndx = params._dmtanhdx(x, params.n_fitparams)
    d2ndx2 = params._d2mtanhdx2(x, params.n_fitparams)
    return d2ndx2 / dndx 

def r_func_density(x, params: TransParams): 
    dndx = params._dmtanhdx(x, params.n_fitparams)
    dpinchdx = params._dpinchdx(x)
    return - params._source(x, params.n_fitparams)/dndx - params._pinch_term(x) - dpinchdx*params._mtanh(x, params.n_fitparams)/dndx


def get_d_inter(x: np.ndarray, initial_condition: float, params: TransParams): 
    solution = solve_ivp(
        ode_system, 
        [x[0], x[-1]], 
        [initial_condition], 
        args=(p_func_density, r_func_density, {"params": params}, {"params": params}), 
        t_eval=x, 
        method="RK45"
    )

    return solution.y[0]

def get_c_inter_over_time(t: float, cinter: float, cinter_initial: float=0.1):
    return cinter_initial + (cinter + cinter_initial)*t

def scale_d_inter(x, base_d_inter, params: TransParams): 
    factor = lambda phi_norm, c_scale: normal_distribution(phi_norm, params.n_fitparams.p, params.n_fitparams.w / 2.0, c_scale, 1.0) 
    if params._CURRENT_C_ETB > 0.0: 
        scaled_solution = base_d_inter / factor(x, params._CURRENT_C_ETB)
    else: 
        scaled_solution = base_d_inter * factor(x, abs(params._CURRENT_C_ETB))
    return scaled_solution

def normal_distribution(x, mu, sigma, c_crash, baseline=0.0): 
    amplitude = c_crash / (np.sqrt(2*np.pi*sigma**2))
    return amplitude*np.exp( - (x - mu)**2 / (2 * sigma**2)) + baseline


from load_data import load_single_pulse_struct, get_fit_params_from_pdbshot, PulseStruct, MachineParameters, KineticProfileStruct, SignalStruct
from physics_helpers import calculate_bohmdiffusion


def setup_base_intratransparams(shot_num: int, C_CRASH: float, C_ETB: float) -> TransParams:
    BASE_PULSE_DIR = JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"
    PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")
    pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
    te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)
    x_eval = np.linspace(0.7, 1.0, 100)
    steady_state_te = TransParams.mtanh(x_eval, te_fit_params)
    steady_state_ne = TransParams.mtanh(x_eval, ne_fit_params)
    Dbohm = calculate_bohmdiffusion(steady_state_te, machineparams)
    
    D0_INTRA = Dbohm[0]
    scale_newidth= 1.25
    scale_d0_intra = 5.0
    C_CRASH = 0.3

    init_params = TransParams(
        D = normal_distribution(x_eval, ne_fit_params.p - ne_fit_params.w, ne_fit_params.w*scale_newidth, C_CRASH , D0_INTRA*scale_d0_intra),
        V = np.ones_like(Dbohm),
        S_N = np.zeros_like(Dbohm),
        n_fitparams = ne_fit_params,
        t_fitparams=te_fit_params,
        C_ETB = C_ETB,
        C_CRASH=C_CRASH,
        machine_params=machineparams
    )
    # Zero source in INTRA elm phase 
    return init_params

def setup_base_intertransparams(shot_num: int, C_ETB: float, C_CRASH: float) -> TransParams:
    BASE_PULSE_DIR = JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"
    PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")
    pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
    te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)
    x_eval = np.linspace(0.7, 1.0, 100)
    steady_state_te = TransParams.mtanh(x_eval, te_fit_params)
    steady_state_ne = TransParams.mtanh(x_eval, ne_fit_params)
    Dbohm = calculate_bohmdiffusion(steady_state_te, machineparams)
    
    init_params = TransParams(
        D = Dbohm, 
        V = np.zeros_like(Dbohm),
        S_N = np.zeros_like(Dbohm),
        n_fitparams = ne_fit_params,
        t_fitparams=te_fit_params,
        C_ETB = C_ETB,
        C_CRASH=C_CRASH,
        machine_params=machineparams
    )
    S_N = init_params._source(x_eval, ne_fit_params)
    V   = init_params._pinch_term(x_eval)
    init_params.S_N = S_N
    init_params.V = V

    D_INTER = get_d_inter(x_eval, Dbohm[0]*1.5, init_params)
    init_params.D = D_INTER
    return init_params


# Testing of d_inter
import os 
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

if __name__ == "__main__": 
    BASE_PULSE_DIR = JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"
    
    shot_num = 83624
    C_ETB = 0.5
    C_CRASH = 0.3
    
    # PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")
    # pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
    # te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)
    # steady_state_te = TransParams.mtanh(x_eval, te_fit_params)
    x_eval = np.linspace(0.7, 1.0, 100)
    init_params = setup_base_intertransparams(shot_num, C_CRASH, C_ETB)
    steady_state_ne = TransParams.mtanh(x_eval, init_params.n_fitparams)
    steady_state_te = TransParams.mtanh(x_eval, init_params.t_fitparams)

    Dbohm = calculate_bohmdiffusion(steady_state_te, init_params.machine_params)
    
    D_INTER = get_d_inter(x_eval, Dbohm[0], init_params)

    init_params_intra = setup_base_intratransparams(shot_num, C_CRASH, C_ETB)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4)) 
    axs[0].plot(x_eval, steady_state_ne, color='red', label='pre-ELM')
    axs[0].legend()
    axs[0].set_title("Density")
    axs[1].plot(x_eval, Dbohm / Dbohm[0], label="Bohm")
    axs[1].plot(x_eval, D_INTER/ Dbohm[0], label='INTER')
    axs[1].plot(x_eval, init_params_intra.D / Dbohm[0], label='INTRA')

    axs[1].legend()
    axs[1].set_title("D / $D_{Bohm}^{core}$")
    axs[2].plot(x_eval, init_params.S_N)
    axs[2].set_yscale('log')
    axs[2].set_ylim(1, 500)
    axs[2].set_yticks([0.01, 0.1, 1, 10, 100], ["$10^{17}$", "$10^{18}$", "$10^{19}$", "$10^{20}$", "$10^{21}$"])
    # scaling_factor = np.exp(-k_parameter*(1.0 - x)**m_parameter)
    axs[2].set_title("Source Term: (m$^{-3}$ / s)\n" + r"$(1 / n^{C_{ETB}}) (\text{exp} (-k (1-x)^m))$")
    axs[3].plot(x_eval, init_params.V)
    axs[3].set_title("Pinch Term, $x^5$")

    for ax in axs: 
        ax.axvline(init_params.n_fitparams.p - init_params.n_fitparams.w, color='black', ls='--')
    fig.suptitle(f"C_CRASH = {C_CRASH:.4} | C_ETB = {C_ETB:.4}")
    # Loop through various C_ETB and compare the source and compute inter terms 
    C_ETB = np.linspace(0.5, 1.5, 10)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for c_etb in C_ETB: 
        init_params.C_ETB = c_etb
        S_N = init_params._source(x_eval, init_params.n_fitparams)
        D_INTER = get_d_inter(x_eval, Dbohm[0]*1.5, init_params)
        axs[0].plot(x_eval, S_N, label=f"{c_etb}")
        axs[1].plot(x_eval, D_INTER / Dbohm[0], label=f"{c_etb}")


    axs[0].set_title("Source Term")
    axs[0].set_yscale('log')
    # axs[0].set_ylim(1, 500)
    # axs[0].set_yticks([1, 10, 100], ["$10^{19}$", "$10^{20}$", "$10^{21}$"])

    axs[1].set_title("D / $D_{Bohm}^{core}$")
    axs[1].legend(loc=(1.0, 0.0), title='$C_{ETB}$')
    # axs[1].set_title("Computed INTER")
    plt.show()
