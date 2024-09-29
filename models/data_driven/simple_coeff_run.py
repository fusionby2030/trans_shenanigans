from load_data import * 
from helpers import mtanh, calculate_pressure, normal_distribution, calculate_bohmgyrbohm_particlediffusion, TransParams, solve_pde, T_model, n_model, get_chi_inter, get_d_inter, get_chi_inter_unscaled, get_d_inter_unscaled, BCS, solve_time_evolution, calculate_bohmgrybohm_from_jardin, calculate_bohmdiffusion
from helpers import find_optimal_c_crash, Domain, single_c_growth_run_from_postelm_crash, find_optimal_c_growth
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import scienceplots 
import os 
import sys 
plt.style.use(['science', 'grid'])

from scipy.signal import find_peaks
from tqdm.notebook import tqdm 

PLOTTING = False
if len(sys.argv) > 2: 
    print('plotting')
    PLOTTING = True

BASE_PULSE_DIR = "/home/akadam/EUROfusion/2024/data"
SAVE_FIGURE_DIR = '/home/akadam/EUROfusion/2024/pedestal_transport/models/data_driven/figures'
shot_num = int(sys.argv[1])
# shot_num = 83627 # 83624 # 
PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")
JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"

pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
relevant_profiles = pulse.return_profiles_in_time_windows()

""" Plotting of crash """
if shot_num == 83624:
    offset = 0.046
elif shot_num == 83625:
    offset = 0.041
elif shot_num == 83627:
    offset = 0.043
elif shot_num == 83628:
    offset = 0.052
elif shot_num == 83630: 
    offset = 0.045
elif shot_num == 83631:
    offset = 0.0384
elif shot_num == 83633:
    offset = 0.04
elif shot_num == 83637:
    offset = 0.048
elif shot_num == 83640:
    offset = 0.048
else: 
    print(f'No offset found for {shot_num}')
    offset = 0.0


te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)

# Setup Domain 
N_POINTS    = 1000
R           = np.linspace(machineparams.Rmaj, machineparams.Rmaj + machineparams.Rmin, N_POINTS)
phi         = R**2
phi_norm    = (phi - phi.min()) / (phi.max() - phi.min())
pedestal_region =  phi_norm > 0.65
phi_norm    = phi_norm[pedestal_region]
phi         = phi[pedestal_region]
R           = R[pedestal_region]
dphi_normdr = 2*R / (phi.max() - phi.min())

xdomain = Domain(phi_norm, R, phi, dphi_normdr)


steady_state_te = mtanh(phi_norm, te_fit_params.h1, te_fit_params.h0, te_fit_params.s, te_fit_params.w, te_fit_params.p)
steady_state_ne = mtanh(phi_norm, ne_fit_params.h1, ne_fit_params.h0, ne_fit_params.s, ne_fit_params.w, ne_fit_params.p)
# steady_state_pe = mtanh(phi_norm, pe_fit_params.h1, pe_fit_params.h0, pe_fit_params.s, pe_fit_params.w, pe_fit_params.p)
steady_state_pe = calculate_pressure(steady_state_te, steady_state_ne)

te_fit_params.steady_state_arr = steady_state_te
ne_fit_params.steady_state_arr = steady_state_ne
pe_fit_params.steady_state_arr = steady_state_pe

# Finding ELM timings 
normed_tbeo = (pulse.tbeo.data - pulse.tbeo.data.mean() ) / pulse.tbeo.data.std()
delta_t = pulse.tbeo.time[1] - pulse.tbeo.time[0]
distance_between_elms = int(200E-6 / delta_t)*20
threshold_height = normed_tbeo.std()*2
peaks, _ = find_peaks(normed_tbeo, height=threshold_height, distance=distance_between_elms)
adjusted_peaks = np.empty_like(peaks)
# TODO: SHIFT THE TBEO SIGNAL BY SOME TIME 
for i, pk in enumerate(peaks): 
    for k in range(30): 
        if normed_tbeo[pk - k] < 0: 
            zero_crossing = pk-k 
            break 
        else: 
            zero_crossing = pk 
    adjusted_peaks[i] = zero_crossing


# elm_times_from_expdata = pulse.tbeo.time[peaks]
elm_times_from_expdata = pulse.tbeo.time[adjusted_peaks]
# Get the tau_interelms_from_expdata
tau_interelms_from_expdata = []
for i in range(len(elm_times_from_expdata) - 1): 
    tau_interelms_from_expdata.append(elm_times_from_expdata[i+1] - elm_times_from_expdata[i])


elm_frequency_from_expdata = len(peaks) / (pulse.tbeo.time[-1] - pulse.tbeo.time[0])
t_minus_t_elm_from_expdata = np.empty(len(relevant_profiles.hrts_times))
for i, t in enumerate(relevant_profiles.hrts_times):
    nearest_elm_idx = np.argmin(np.abs(elm_times_from_expdata - t))
    t_nearest_elm = elm_times_from_expdata[nearest_elm_idx]    
    t_minus_t_elm_from_expdata[i] = t - t_nearest_elm

chi_gb, gyroradius = calculate_bohmgrybohm_from_jardin(R, phi, phi_norm, steady_state_te, machineparams)
Dbohm = calculate_bohmdiffusion(steady_state_te, machineparams)


""" 
Find the optimal C_CRASH first
"""


best_c_crash, (scanned_crashes, logliklihoods), data_from_best_crash = find_optimal_c_crash(xdomain, 
                                                                      te_fit_params, ne_fit_params, 
                                                                      chi_gb[0], Dbohm[0],
                                                                      t_minus_t_elm_from_expdata,
                                                                      relevant_profiles, offset)

print(f"Best crash: {best_c_crash}")

_, INTRA_PARAMS, bestcrash_intraelm_te, bestcrash_intraelm_ne, bestcrash_intraelm_pe = data_from_best_crash

post_elm_ne = bestcrash_intraelm_ne[:, -1]
post_elm_te = bestcrash_intraelm_te[:, -1]
post_elm_pe = bestcrash_intraelm_pe[:, -1]
if PLOTTING: 
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].scatter(scanned_crashes, logliklihoods)
    axs[0].axvline(best_c_crash, color='red')
    axs[0].set_yscale('log')
    axs[0].set_yticks([1, 10, 100])
    axs[1].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.ne*1E-19, color='grey', edgecolors='black')
    axs[1].plot(phi_norm, steady_state_ne, color='red')
    axs[1].plot(phi_norm, post_elm_ne, color='blue')

    axs[2].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.te*1E-3, color='grey', edgecolors='black')
    axs[2].plot(phi_norm, steady_state_te, color='red')
    axs[2].plot(phi_norm, post_elm_te, color='blue')

    axs[3].scatter(relevant_profiles.hrts_psi + offset, calculate_pressure(relevant_profiles.te*1E-3, relevant_profiles.ne*1E-19), color='grey', edgecolors='black', label=f'Data in {pulse.t1:.4} - {pulse.t2:.4}')
    axs[3].plot(phi_norm, steady_state_pe, color='red', label='JET PDB Fit')
    axs[3].plot(phi_norm, post_elm_pe, color='blue', label='Post ELM')

    for ax in axs[1:]:
        ax.set_xlabel(r'$\phi$')
        ax.set_xlim(phi_norm.min(), 1.05)

    axs[0].set_xlabel(r'$C_{crash}$')
    axs[0].set_ylabel(r'$\sum_{t, \phi} \log \mathcal{L}$')
    axs[0].set_title('- log liklihood of C-CRASH')
    
    axs[1].set_ylabel(r'$n_e$ ($10^{19} m^{-3}$)')
    axs[2].set_ylabel(r'$T_e$ (keV)')
    axs[3].set_ylabel(r'$P_e$ (kPa)')
    axs[3].legend()
    axs[1].set_ylim(0.0, 7.5)
    axs[2].set_ylim(0.0, 1.25)
    axs[3].set_ylim(0.0, 10.0)

    fig.suptitle(f'JET \# {shot_num}')

    fig.savefig(os.path.join(SAVE_FIGURE_DIR, f'{shot_num}_crash.png'))
    # plt.show()


"""
Find the optimal C_GROWTH 
"""
CHI0_INTER = chi_gb[0]*0.50 # Factor 0.3 USED TO MATCH THE NE GROWTH, MORE OR LESS. 
D0_INTER   = Dbohm[0]*1.35

exp_tau_interelm_mean = np.mean(tau_interelms_from_expdata)
best_c_growth, (scan_growths, tau_interelms_from_sim), evolution = find_optimal_c_growth(exp_tau_interelm_mean, 
                                         xdomain, 
                                         post_elm_ne, post_elm_te, 
                                         te_fit_params, ne_fit_params,  
                                         CHI0_INTER, D0_INTER, machineparams)
print(f'Best growth: {best_c_growth}')

if PLOTTING:
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.ravel()
    axs[0].hist(tau_interelms_from_expdata, bins=20)
    axs[0].set_xlabel(r'$\tau_{interelm, exp}$')
    axs[0].set_title('inter-ELM times from TBEO signal')
    axs[0].set_ylabel('Counts')
    axs[0].axvline(np.mean(tau_interelms_from_expdata), color='red', label=f'Mean {np.mean(tau_interelms_from_expdata):.4}')
    axs[0].axvline(np.median(tau_interelms_from_expdata), color='blue', label=f'Median {np.median(tau_interelms_from_expdata):.4}')
    axs[0].legend()

    axs[1].scatter(scan_growths, tau_interelms_from_sim)
    axs[1].set_xlabel(r'$C_{growth}$')
    axs[1].set_ylabel(r'$\tau_{interelm, sim}$')
    axs[1].set_title('Inter-ELM times from simulation')
    axs[1].axhline(exp_tau_interelm_mean, color='red', ls='--')

    # post elm starting point
    axs[3].plot(xdomain.phi_norm, post_elm_ne, label='Post ELM', color='salmon', zorder=20)
    axs[4].plot(xdomain.phi_norm, post_elm_te, label='Post ELM', color='salmon', zorder=20)
    axs[5].plot(xdomain.phi_norm, post_elm_pe, label='Post ELM', color='salmon', zorder=20)
    # Experimental data 
    axs[3].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.ne*1E-19, color='grey', edgecolors='black')
    axs[4].scatter(relevant_profiles.hrts_psi + offset, relevant_profiles.te*1E-3, color='grey', edgecolors='black')
    axs[5].scatter(relevant_profiles.hrts_psi + offset, calculate_pressure(relevant_profiles.te*1E-3, relevant_profiles.ne*1E-19), color='grey', edgecolors='black', label=f'Data in {pulse.t1:.4} - {pulse.t2:.4}')
    # Steady state
    axs[3].plot(xdomain.phi_norm, steady_state_ne, label='Steady state', color='red', zorder=20)
    axs[4].plot(xdomain.phi_norm, steady_state_te, label='Steady state', color='red', zorder=20)
    axs[5].plot(xdomain.phi_norm, steady_state_pe, label='Steady state', color='red', zorder=20)
    # Growth 
    growth_time, growth_te, growth_ne, growth_pe = evolution
    # Shapes: time(n), te(n, m) ...
    for i in range(len(growth_time)): 
        color = plt.cm.viridis(i/len(growth_time))
        axs[3].plot(xdomain.phi_norm, growth_ne[i], color=color, alpha=0.5, zorder=4)
        axs[4].plot(xdomain.phi_norm, growth_te[i], color=color, alpha=0.5, zorder=4)
        axs[5].plot(xdomain.phi_norm, growth_pe[i], color=color, alpha=0.5, zorder=4)
    
    axs[2].remove()
   
   # x and ylims
    for ax in axs[3:]:
       ax.set_xlim(xdomain.phi_norm.min(), 1.05)
       ax.set_xlabel(r'$\phi$')
    
    axs[3].set_ylabel(r'$n_e$ ($10^{19} m^{-3}$)')
    axs[4].set_ylabel(r'$T_e$ (keV)')
    axs[5].set_ylabel(r'$P_e$ (kPa)')
    axs[3].set_ylim(0.0, 7.5)
    axs[4].set_ylim(0.0, 1.25)
    axs[5].set_ylim(0.0, 10.0)
    fig.suptitle(f'JET \# {shot_num}, C-crash: {best_c_crash:.4}, C-growth: {best_c_growth:.4}')
    fig.savefig(os.path.join(SAVE_FIGURE_DIR, f'{shot_num}_growth.png'))
    
if PLOTTING:     
    plt.show()