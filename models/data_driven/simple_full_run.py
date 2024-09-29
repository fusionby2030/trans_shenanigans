from helpers import Domain, setup_intra_params, setup_inter_params, TransParams, setup_domain
from load_data import get_fit_params_from_pdbshot, load_single_pulse_struct, PulseStruct, KineticProfileStruct, SignalStruct
from helpers import calculate_bohmdiffusion, calculate_bohmgrybohm_from_jardin_dc, get_c_inter_over_time, get_chi_inter, get_d_inter
from helpers import calculate_pressure, mtanh, calculate_maxalpha, solve_pde, T_model
import sys 
import os 
import numpy as np 

C_CRASH  = 1.456 # 2.0755172413793104
C_GROWTH = 2.053 # 1.8421052631578947

JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"
BASE_PULSE_DIR = "/home/akadam/EUROfusion/2024/data"
shot_num = int(sys.argv[1])
if len(sys.argv) > 2:
    SAMPLE_EVERY_CRASH = True 
else:
    SAMPLE_EVERY_CRASH = False
    
PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")

pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
relevant_profiles = pulse.return_profiles_in_time_windows()

te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)
xdomain = setup_domain(machineparams)

steady_state_te = mtanh(xdomain.phi_norm, te_fit_params.h1, te_fit_params.h0, te_fit_params.s, te_fit_params.w, te_fit_params.p)
steady_state_ne = mtanh(xdomain.phi_norm, ne_fit_params.h1, ne_fit_params.h0, ne_fit_params.s, ne_fit_params.w, ne_fit_params.p)
steady_state_pe = calculate_pressure(steady_state_te, steady_state_ne)
te_fit_params.steady_state_arr = steady_state_te
ne_fit_params.steady_state_arr = steady_state_ne
pe_fit_params.steady_state_arr = steady_state_pe

chi_gb, gyroradius = calculate_bohmgrybohm_from_jardin_dc(xdomain, steady_state_te, machineparams)
Dbohm = calculate_bohmdiffusion(steady_state_te, machineparams)

CHI0_INTER = chi_gb[0]*0.50 # Factor 0.3 USED TO MATCH THE NE GROWTH, MORE OR LESS. 
D0_INTER   = Dbohm[0]*1.35

INTRA_PARAMS = setup_intra_params(C_CRASH, xdomain, te_fit_params, ne_fit_params, chi_gb[0], Dbohm[0])
INTER_PARAMS = setup_inter_params(xdomain, te_fit_params, ne_fit_params, machineparams, CHI0_INTER, D0_INTER)

ALPHA_CRIT = calculate_maxalpha(steady_state_pe, xdomain) 

# Writing setings 
dt           = 0.0001 # half of tau_interelm 
tau_interelm = dt
tau_intraelm = 200E-6

total_sim_time = 1.0
num_steps      = int(total_sim_time / dt) 
savetimes      = np.linspace(0.0, total_sim_time, num_steps)

current_time   = 0.0
current_temp   = steady_state_te
current_dens   = steady_state_ne
current_pres   = steady_state_pe

all_times, all_ne, all_te, all_pe = [], [], [], []

while current_time < total_sim_time: 
    cycle_ne, cycle_te, cycle_pe = [], [], []
    # Start with ELM crash 
    _t_interval_internal = [current_time, current_time + tau_intraelm]
    _t_keep_interval     = np.linspace(_t_interval_internal[0], _t_interval_internal[1], 1)
    print(_t_interval_internal, _t_keep_interval, int(tau_interelm / dt))
    solutions_intraelm_te = solve_pde(xdomain.phi_norm, current_temp, _t_interval_internal, INTRA_PARAMS, T_model, _t_keep_interval)
    solutions_intraelm_ne = solve_pde(xdomain.phi_norm, current_dens, _t_interval_internal, INTRA_PARAMS, T_model, _t_keep_interval)

    current_ne = solutions_intraelm_ne[:,  -1]
    current_te = solutions_intraelm_te[:,  -1]
    current_pe = calculate_pressure(current_te, current_ne)
    
    cycle_ne.append(current_ne)
    cycle_te.append(current_te)
    cycle_pe.append(current_pe)

    current_time += tau_interelm
    all_times.append(current_time)

    # Then Grow
    ALPHA_EXP = calculate_maxalpha(current_pe, xdomain)
    t_last_elm = 0.0
    while True: 
        if ALPHA_EXP > ALPHA_CRIT: 
            break
        _t_internal = [current_time, current_time + dt]
        INTER_PARAMS._C = get_c_inter_over_time(t_last_elm, C_GROWTH)
        INTER_PARAMS.CHI = get_chi_inter(xdomain.phi_norm, CHI0_INTER, te_fit_params.steady_state_arr, INTER_PARAMS)
        INTER_PARAMS.D = get_d_inter(xdomain.phi_norm, D0_INTER, ne_fit_params.steady_state_arr, INTER_PARAMS)

        solutions_interelm_te = solve_pde(xdomain.phi_norm, current_te, _t_internal, INTER_PARAMS, T_model)
        solutions_interelm_ne = solve_pde(xdomain.phi_norm, current_ne, _t_internal, INTER_PARAMS, T_model)

        current_te = solutions_interelm_te[:, -1]
        current_ne = solutions_interelm_ne[:, -1]
        current_pe = calculate_pressure(current_te, current_ne)

        cycle_ne.append(current_ne)
        cycle_te.append(current_te)
        cycle_pe.append(current_pe)
        t_last_elm += dt 
        current_time += dt
        all_times.append(current_time)
        ALPHA_EXP = calculate_maxalpha(current_pe, xdomain)
        print(f'Current Time: {current_time:.4}, Alpha Exp: {ALPHA_EXP:.4}, Alpha Crit: {ALPHA_CRIT:.4}, C_INTER: {INTER_PARAMS._C:.4}')
    all_ne.extend(cycle_ne)
    all_te.extend(cycle_te)
    all_pe.extend(cycle_pe)
    print("Cycle completed", all_times[-1])

all_ne = np.array(all_ne)
all_te = np.array(all_te)
all_pe = np.array(all_pe)
# save them 
np.save(f"ne_{shot_num}.npy", all_ne)
np.save(f"te_{shot_num}.npy", all_te)
np.save(f"pe_{shot_num}.npy", all_pe)
np.save(f"times_{shot_num}.npy", all_times)
