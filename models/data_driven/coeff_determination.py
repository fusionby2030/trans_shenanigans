from load_data import * 
from helpers import mtanh, calculate_pressure, normal_distribution, TransParams, solve_pde, T_model, n_model, get_chi_inter, get_d_inter, BCS, solve_time_evolution

import matplotlib.pyplot as plt 
import scienceplots 
import os 
import sys 

plt.style.use(['science', 'grid'])

# PULSE_STRUCT_DIR="/home/akadam/EUROfusion/2024/data/jet_83628"
BASE_PULSE_DIR = "/home/akadam/EUROfusion/2024/data"
shot_num = 83628
PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")
JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"

psi_eval = np.linspace(0.8, 1.0, 100)

pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
relevant_profiles = pulse.return_profiles_in_time_windows()

te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)
print(machineparams)
steady_state_te = mtanh(psi_eval, te_fit_params.h1, te_fit_params.h0, te_fit_params.s, te_fit_params.w, te_fit_params.p)
steady_state_ne = mtanh(psi_eval, ne_fit_params.h1, ne_fit_params.h0, ne_fit_params.s, ne_fit_params.w, ne_fit_params.p)
steady_state_pe = mtanh(psi_eval, pe_fit_params.h1, pe_fit_params.h0, pe_fit_params.s, pe_fit_params.w, pe_fit_params.p)

""" Plotting of crash """
if shot_num == 83624:
    offset = 0.04
elif shot_num == 83628:
    offset = 0.052
elif shot_num == 83625:
    offset = 0.035
else: 
    offset = 0.0
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.ravel()

axs[0].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.ne*1E-19, color='grey', edgecolors='black')
axs[0].plot(psi_eval, steady_state_ne, color='red')

axs[1].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.te*1E-3, color='grey', edgecolors='black')
axs[1].plot(psi_eval, steady_state_te, color='red')

axs[2].scatter(relevant_profiles.hrts_psi + offset, calculate_pressure(relevant_profiles.te*1E-3, relevant_profiles.ne*1E-19), color='grey', edgecolors='black', label=f'Data in {pulse.t1:.4} - {pulse.t2:.4}')
axs[2].plot(psi_eval, steady_state_pe, color='red', label='JET PDB Fit')

axs[0].set_xlim(0.0, 7.5)
axs[1].set_ylim(0.0, 1)
axs[2].set_ylim(0, 10)
for ax in axs[:3]: 
    ax.set_xlim(0.8, 1.05)
    ax.set_xlabel(r'$\psi_N$')
axs[0].set_ylabel(r'$n_e$ ($10^{19} m^{-3}$)')
axs[1].set_ylabel(r'$T_e$ (keV)')
axs[2].set_ylabel(r'$P_e$ (kPa)')



C_CRASH    = 2.5
D0_INTRA   = 25.0
CHI0_INTRA = 0.5


intra_params = TransParams(
    CHI=normal_distribution(psi_eval, te_fit_params.p - te_fit_params.w, te_fit_params.w, C_CRASH, CHI0_INTRA), # np.ones_like(psi_eval),
    D=normal_distribution(psi_eval, ne_fit_params.p, ne_fit_params.w / 2.0, C_CRASH, D0_INTRA), # np.ones_like(psi_eval),
    V=np.ones_like(psi_eval),
    S_N=np.zeros_like(psi_eval),
    S_T=np.zeros_like(psi_eval),
    _C = C_CRASH,
    ne_fitparams=ne_fit_params, 
    te_fitparams=te_fit_params, 
    bcs=BCS.INTRA
)

# Solve INTRA-ELM transport
tau_intraelm = 200E-6 # 200 us
t_interval = [0, tau_intraelm]
t_keep     = np.linspace(0, tau_intraelm, 10)

solutions_intraelm_te = solve_pde(psi_eval, steady_state_te, t_interval, intra_params, T_model, t_keep)
solutions_intraelm_ne = solve_pde(psi_eval, steady_state_ne, t_interval, intra_params, n_model, t_keep)
solutions_intraelm_pe = calculate_pressure(solutions_intraelm_te, solutions_intraelm_ne)

post_elm_te = solutions_intraelm_te[:, -1]
post_elm_ne = solutions_intraelm_ne[:, -1]
post_elm_pe = solutions_intraelm_pe[:, -1]

# Solve INTER-ELM Transport 

# Plotting
if sys.argv[1] == 'growth': 
    D0_INTER = 8.0
    CHI0_INTRA = 0.5
    C_INTER = 5.0
    axs[0].plot(psi_eval, post_elm_ne, color='red', label='Post ELM')
    axs[1].plot(psi_eval, post_elm_te, color='red', label='Post ELM')

    for C_INTER in np.linspace(1.0, 5.0, 5): 
        
        INTER_PARAMS = TransParams(
            CHI = np.empty_like(psi_eval),
            D   = np.empty_like(psi_eval),
            V   = psi_eval**5,
            S_N = np.zeros_like(psi_eval),
            S_T = np.zeros_like(psi_eval),
            _C  = C_INTER, 
            ne_fitparams=ne_fit_params, 
            te_fitparams=te_fit_params,
            bcs=BCS.INTER
        )

        INTER_PARAMS.CHI = get_chi_inter(psi_eval, CHI0_INTRA/C_INTER, steady_state_te, INTER_PARAMS)
        INTER_PARAMS.D   = get_d_inter(psi_eval, D0_INTER/C_INTER, steady_state_ne, INTER_PARAMS)

        tau_interelm = 0.04 # 
        t_interval = [0, tau_interelm]
        t_keep    = np.linspace(0, tau_interelm, 10)

        solutions_interelm_te = solve_pde(psi_eval, post_elm_te, t_interval, INTER_PARAMS, T_model, t_keep)
        solutions_interelm_ne = solve_pde(psi_eval, post_elm_ne, t_interval, INTER_PARAMS, n_model, t_keep)
        solutions_interelm_pe = calculate_pressure(solutions_interelm_te, solutions_interelm_ne)


        axs[0].plot(psi_eval, solutions_interelm_ne[:, -1], ls='--', lw=2, label=f'C_INTER={C_INTER}')
        axs[1].plot(psi_eval, solutions_interelm_te[:, -1], ls='--', lw=2, label=f'C_INTER={C_INTER}')
        axs[2].plot(psi_eval, solutions_interelm_pe[:, -1], ls='--', lw=2, label=f'C_INTER={C_INTER:.3}')

        axs[3].plot(psi_eval, INTER_PARAMS.D, label=f'C_INTER={C_INTER}')
        axs[4].plot(psi_eval, INTER_PARAMS.CHI, label=f'C_INTER={C_INTER}')
    axs[5].remove()

if sys.argv[1] == 'crash': 
    for C_CRASH in np.linspace(0.1, 3.0, 10): # [1.0, 2.0, 3.0, 4.0]:

        intra_params = TransParams(
            CHI=normal_distribution(psi_eval, te_fit_params.p, te_fit_params.w*2.0, C_CRASH, CHI0_INTRA), # np.ones_like(psi_eval),
            D=normal_distribution(psi_eval, ne_fit_params.p, ne_fit_params.w*8.0, C_CRASH, D0_INTRA), # np.ones_like(psi_eval),
            V=C_CRASH*np.ones_like(psi_eval),
            S_N=np.zeros_like(psi_eval),
            S_T=np.zeros_like(psi_eval),
            _C = C_CRASH,
            ne_fitparams=ne_fit_params, 
            te_fitparams=te_fit_params,
            bcs=BCS.INTRA
        )

        solutions_intraelm_te = solve_pde(psi_eval, steady_state_te, t_interval, intra_params, T_model, t_keep)
        solutions_intraelm_ne = solve_pde(psi_eval, steady_state_ne, t_interval, intra_params, n_model, t_keep)
        solutions_intraelm_pe = calculate_pressure(solutions_intraelm_te, solutions_intraelm_ne)

        axs[0].plot(psi_eval, solutions_intraelm_ne[:, -1], ls='--', lw=2, label=f'C_CRASH={C_CRASH}')
        axs[1].plot(psi_eval, solutions_intraelm_te[:, -1], ls='--', lw=2, label=f'C_CRASH={C_CRASH}')
        axs[2].plot(psi_eval, solutions_intraelm_pe[:, -1], ls='--', lw=2, label=f'C_CRASH={C_CRASH:.3}')

        axs[3].plot(psi_eval, intra_params.D, label=f'C_CRASH={C_CRASH}')
        axs[4].plot(psi_eval, intra_params.CHI, label=f'C_CRASH={C_CRASH}')
    axs[5].remove()

if sys.argv[-1] == 'full': 
    D0_INTER = 8.0
    CHI0_INTRA = 0.5
    C_INTER = 5.0
    
    INTER_PARAMS = TransParams(
        CHI = np.empty_like(psi_eval),
        D   = np.empty_like(psi_eval),
        V   = psi_eval**5,
        S_N = np.zeros_like(psi_eval),
        S_T = np.zeros_like(psi_eval),
        _C  = C_INTER, 
        ne_fitparams=ne_fit_params, 
        te_fitparams=te_fit_params,
        bcs=BCS.INTER
    )

    INTER_PARAMS.CHI = get_chi_inter(psi_eval, CHI0_INTRA/C_INTER, steady_state_te, INTER_PARAMS)
    INTER_PARAMS.D   = get_d_inter(psi_eval, D0_INTER/C_INTER, steady_state_ne, INTER_PARAMS)

    C_CRASH    = 2.5
    D0_INTRA   = 25.0

    INTRA_PARAMS = TransParams(
        CHI=normal_distribution(psi_eval, te_fit_params.p - te_fit_params.w, te_fit_params.w, C_CRASH, CHI0_INTRA), # np.ones_like(psi_eval),
        D=normal_distribution(psi_eval, ne_fit_params.p, ne_fit_params.w / 2.0, C_CRASH, D0_INTRA), # np.ones_like(psi_eval),
        V=np.ones_like(psi_eval),
        S_N=np.zeros_like(psi_eval),
        S_T=np.zeros_like(psi_eval),
        _C = C_CRASH,
        ne_fitparams=ne_fit_params, 
        te_fitparams=te_fit_params, 
        bcs=BCS.INTRA
    )

    psi_eval = np.linspace(0.8, 1.0, 100)
    PED_IDX  = np.argmin(abs(psi_eval - (te_fit_params.p - te_fit_params.w)))
    time_plot, density_plot, temperature_plot, pressure_plot = solve_time_evolution(steady_state_ne, steady_state_te, INTER_PARAMS, INTRA_PARAMS)
    
    axs[3].plot(time_plot, density_plot[:, PED_IDX])
    axs[4].plot(time_plot, temperature_plot[:, PED_IDX])
    axs[5].plot(time_plot, pressure_plot[:, PED_IDX])
    axs[3].axhline(ne_fit_params.h1, color='red')
    axs[4].axhline(te_fit_params.h1, color='red')
    axs[5].axhline(pe_fit_params.h1, color='red')

    for ax in axs[:3]: 
        ax.axvline(psi_eval[PED_IDX])

    axs[3].set_ylim(0.0, 7.5)
    axs[4].set_ylim(0.0, 1)
    axs[5].set_ylim(0, 10)

    np.save( f'time_{pulse.shot_num}.npy', time_plot,)
    np.save( f'density_{pulse.shot_num}.npy', density_plot,)
    np.save( f'temperature_{pulse.shot_num}.npy', temperature_plot,)
    np.save( f'pressure_{pulse.shot_num}.npy', pressure_plot,)

    
# axs[3].plot(psi_eval, np.gradient(steady_state_ne, psi_eval)/steady_state_ne, color='red')
# axs[3].set_title(r'$\nabla_\psi n_e / n_e$')
# axs[4].plot(psi_eval, np.gradient(steady_state_te, psi_eval)/steady_state_te, color='red')
# axs[4].set_title(r'$\nabla_\psi T_e / T_e$')
# axs[5].plot(psi_eval, np.gradient(steady_state_pe, psi_eval)/steady_state_pe, color='red')
# axs[5].set_title(r'$\nabla_\psi P_e / P_e$')

fig.suptitle(f'JET \#{pulse.shot_num} kinetic profiles and fit\nINTRA-ELM Crash')

axs[2].legend(loc=(0.2, -1.1), frameon=False)
plt.show()