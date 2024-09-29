from load_data import * 
from interpretive import mtanh, source, pinch_term, ode_system, p_chi, r_chi, p_d, r_d, chi_intraelm, calculate_pressure, d_intraelm
from interpretive_solve import solve_pde, solve_npde 

from scipy.integrate import solve_ivp 
import matplotlib.pyplot as plt 
import scienceplots 
import sys 
plt.style.use(['science', 'grid'])

PULSE_STRUCT_DIR="/home/akadam/EUROfusion/2024/data/jet_83628"
JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"

""" 
1. Load pulse struct 
2. Fit parameters based on pulse struct, 
    - c_elm based on variation of ne and te in profile
"""

psi_eval = np.linspace(0.0, 1.0, 100)

pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
relevant_profiles = pulse.return_profiles_in_time_windows()

te_fit_params, ne_fit_params, pe_fit_params  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)

steady_state_te = mtanh(psi_eval, te_fit_params.h1, te_fit_params.h0, te_fit_params.s, te_fit_params.w, te_fit_params.p)
steady_state_ne = mtanh(psi_eval, ne_fit_params.h1, ne_fit_params.h0, ne_fit_params.s, ne_fit_params.w, ne_fit_params.p)
steady_state_pe = mtanh(psi_eval, pe_fit_params.h1, pe_fit_params.h0, pe_fit_params.s, pe_fit_params.w, pe_fit_params.p)


if sys.argv[1] == 'plot': 
    offset = 0.052
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.ravel()

    axs[0].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.ne*1E-19, color='grey', edgecolors='black')
    axs[0].plot(psi_eval, steady_state_ne, color='red')

    axs[1].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.te*1E-3, color='grey', edgecolors='black')
    axs[1].plot(psi_eval, steady_state_te, color='red')

    axs[2].scatter(relevant_profiles.hrts_psi + offset, calculate_pressure(relevant_profiles.te*1E-3, relevant_profiles.ne*1E-19), color='grey', edgecolors='black', label=f'Data in {pulse.t1:.4} - {pulse.t2:.4}')
    axs[2].plot(psi_eval, steady_state_pe, color='red', label='JET PDB Fit')

    axs[1].set_ylim(0.0, 1)
    axs[2].set_ylim(0, 10)
    for ax in axs: 
        ax.set_xlim(0.8, 1.05)
        ax.set_xlabel(r'$\psi_N$')
    axs[0].set_ylabel(r'$n_e$ ($10^{19} m^{-3}$)')
    axs[1].set_ylabel(r'$T_e$ (keV)')
    axs[2].set_ylabel(r'$P_e$ (kPa)')

    axs[3].plot(psi_eval, np.gradient(steady_state_ne, psi_eval)/steady_state_ne, color='red')
    axs[3].set_title(r'$\nabla_\psi n_e / n_e$')
    axs[4].plot(psi_eval, np.gradient(steady_state_te, psi_eval)/steady_state_te, color='red')
    axs[4].set_title(r'$\nabla_\psi T_e / T_e$')
    axs[5].plot(psi_eval, np.gradient(steady_state_pe, psi_eval)/steady_state_pe, color='red')
    axs[5].set_title(r'$\nabla_\psi P_e / P_e$')

    
    fig.suptitle(f'JET \#{pulse.shot_num} kinetic profiles and fit')
    

# SETUP

C_CRASH = 2.0
CHI_GB  = 3.0
D_BOHM  = 0.05

chi_intraelm_vals = chi_intraelm(psi_eval, CHI_GB, te_fit_params.w, te_fit_params.p, C_CRASH)
heat_source       = source(psi_eval, 10.0, 0.6, 0.1, 0.1)


D_intraelm_vals   = chi_intraelm(psi_eval, D_BOHM, ne_fit_params.w, ne_fit_params.p, C_CRASH, scaling_factor=0.5)
V_intraelm_vals   = np.ones_like(psi_eval)# *0.0
particle_source   = source(psi_eval, 5.0, 1.05, 0.02, 0.1)

# Solve PDE
t0 = steady_state_te
n0 = steady_state_ne

tau_intraelm = 200e-6 # 200 microseconds
t_interval = [0, tau_intraelm]

solutions_te = solve_pde(psi_eval, t0, t_interval, chi_intraelm_vals, heat_source)
solutions_ne = solve_npde(psi_eval, n0, t_interval, D_intraelm_vals, V_intraelm_vals, particle_source)
solutions_pe = calculate_pressure(solutions_te[:, -1], solutions_ne[:, -1])

# plot same as above but add
if len(sys.argv) > 2 and sys.argv[1] == 'plot' and sys.argv[2] == 'intraelm':
    axs[0].plot(psi_eval, solutions_ne[:, -1], color='blue')
    axs[1].plot(psi_eval, solutions_te[:, -1], color='blue')
    axs[2].plot(psi_eval, solutions_pe, color='blue', label=f'Post ELM crash: C_CRASH={C_CRASH:0.3}')
    

elif sys.argv[1] == 'plot':
    axs[2].legend()

    plt.show()