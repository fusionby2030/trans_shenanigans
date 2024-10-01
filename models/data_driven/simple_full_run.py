from helpers import Domain, setup_intra_params, setup_inter_params, TransParams, setup_domain
from load_data import get_fit_params_from_pdbshot, load_single_pulse_struct, PulseStruct, KineticProfileStruct, SignalStruct
from helpers import calculate_bohmdiffusion, calculate_bohmgrybohm_from_jardin_dc, get_c_inter_over_time, get_chi_inter, get_d_inter
from helpers import calculate_pressure, mtanh, calculate_maxalpha, solve_pde, T_model, n_model
import sys 
import os 
import numpy as np 
import pickle 
import argparse 

# C_CRASH  = 1.456 # 2.0755172413793104
# C_GROWTH = 2.053 # 1.8421052631578947

# JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"
# BASE_PULSE_DIR = "/home/akadam/EUROfusion/2024/data"
JET_PDB_DIR    = "/scratch/project_2009007/stability_fun/data" 
BASE_PULSE_DIR = JET_PDB_DIR
LOCAL_SAVE_DIR = "/scratch/project_2009007/stability_fun/trans_shenanigans/models/data_driven/outputs"

parser = argparse.ArgumentParser(description='Process input parameters.')
parser.add_argument('-shot_num', type=int, help='Shot number')
parser.add_argument('-ccrash', type=float, default=1.0, help='C_CRASH coefficient')
parser.add_argument('-cgrowth', type=float, default=1.0, help='C_GROWTH coefficient')
parser.add_argument('-probable', type=int, choices=[0, 1, 2, 3], default=0, help='Probabilistic mode (0: off, 1: both, 2: C_GROWTH only, 3: C_CRASH only)')
parser.add_argument('-savedir', type=str, default=LOCAL_SAVE_DIR, help='Where to store outputs')
parser.add_argument('--loc_c_crash', type=float, default=0.1, help='Location parameter for resampling C_CRASH')
parser.add_argument('--loc_c_growth', type=float, default=0.1, help='Location parameter for resampling C_GROWTH')

# Parse arguments
args = parser.parse_args()


shot_num = args.shot_num # int(sys.argv[1])
C_CRASH  = args.ccrash   # float(sys.argv[2])
C_GROWTH = args.cgrowth  # float(sys.argv[3])
probabilistic_mode = args.probable
SAMPLE_EVERY_CRASH = probabilistic_mode != 0
SAVE_DIR = args.savedir

save_dir = os.path.join(os.path.join(SAVE_DIR, f'{shot_num}'))
if SAMPLE_EVERY_CRASH:
    MU_C_GROWTH = C_GROWTH
    MU_C_CRASH  = C_CRASH
    LOC_C_CRASH = args.loc_c_crash
    LOC_C_GROWTH = args.loc_c_growth
    # LOC_C_GROWTH = 0.12
    # LOC_C_CRASH  = 0.075
    seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(seed)
    save_dir = os.path.join(save_dir, f'probabilistic_{seed}')

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


PULSE_STRUCT_DIR = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}")

pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
relevant_profiles = pulse.return_profiles_in_time_windows()

te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)
xdomain = setup_domain(machineparams)

with open(os.path.join(save_dir, 'domain.pickle'), 'wb') as file: 
    pickle.dump(xdomain, file)

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

# Save input parameters to a text file
with open(os.path.join(save_dir, 'input_parameters.txt'), 'w') as f:
    f.write(f"Shot Number: {shot_num}\n")
    f.write(f"C_CRASH: {C_CRASH}\n")
    f.write(f"C_GROWTH: {C_GROWTH}\n")
    f.write(f"Probabilistic Mode: {probabilistic_mode}\n")
    if SAMPLE_EVERY_CRASH:
        f.write(f"MU_C_GROWTH: {MU_C_GROWTH}\n")
        f.write(f"MU_C_CRASH: {MU_C_CRASH}\n")
        f.write(f"LOC_C_GROWTH: {LOC_C_GROWTH}\n")
        f.write(f"LOC_C_CRASH: {LOC_C_CRASH}\n")
    f.write(f"Random Seed: {seed if SAMPLE_EVERY_CRASH else 'N/A'}\n")
    f.write(f'ALPHA_CRIT: {ALPHA_CRIT}\n')


# Writing setings 
dt           = 0.0001 # half of tau_interelm 
tau_interelm = dt
tau_intraelm = 200E-6

total_sim_time = 1.00
num_steps      = int(total_sim_time / dt) 
savetimes      = np.linspace(0.0, total_sim_time, num_steps)

current_time   = 0.0
current_temp   = steady_state_te
current_dens   = steady_state_ne
current_pres   = steady_state_pe

all_times, all_ne, all_te, all_pe = [], [], [], []
all_c_growth, all_c_crash = [], []
ELM_COUNTER = 0
while current_time < total_sim_time: 
    cycle_ne, cycle_te, cycle_pe = [], [], []
    all_c_growth.append(C_GROWTH)
    all_c_crash.append(C_CRASH)

    # Start with ELM crash 
    _t_interval_internal = [current_time, current_time + tau_intraelm]
    solutions_intraelm_te = solve_pde(xdomain.phi_norm, current_temp, _t_interval_internal, INTRA_PARAMS, T_model)
    solutions_intraelm_ne = solve_pde(xdomain.phi_norm, current_dens, _t_interval_internal, INTRA_PARAMS, n_model)

    current_ne = solutions_intraelm_ne[:,  -1]
    current_te = solutions_intraelm_te[:,  -1]
    current_pe = calculate_pressure(current_te, current_ne)
    
    cycle_ne.append(current_ne)
    cycle_te.append(current_te)
    cycle_pe.append(current_pe)

    current_time += tau_interelm
    ELM_COUNTER += 1
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
        solutions_interelm_ne = solve_pde(xdomain.phi_norm, current_ne, _t_internal, INTER_PARAMS, n_model)

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
    print("Cycle completed", all_times[-1], ELM_COUNTER)
    if SAMPLE_EVERY_CRASH:
        # resample parameters
        if probabilistic_mode == 1:
            C_CRASH  = np.random.normal(MU_C_CRASH, LOC_C_CRASH)
            C_GROWTH = np.random.lognormal(np.log(MU_C_GROWTH), LOC_C_GROWTH)
        elif probabilistic_mode == 2:
            C_GROWTH = np.random.lognormal(np.log(MU_C_GROWTH), LOC_C_GROWTH)
        elif probabilistic_mode == 3:
            C_CRASH  = np.random.normal(MU_C_CRASH, LOC_C_CRASH)
        print(f'NEW: C_CRASH {C_CRASH:.4} | C_GROWTH {C_GROWTH:.4}')
        INTRA_PARAMS = setup_intra_params(C_CRASH, xdomain, te_fit_params, ne_fit_params, chi_gb[0], Dbohm[0])


all_ne = np.array(all_ne)
all_te = np.array(all_te)
all_pe = np.array(all_pe)
print(all_ne.shape)
# save them 
np.save(os.path.join(save_dir, f"ne.npy"), all_ne)
np.save(os.path.join(save_dir, f"te.npy"), all_te)
np.save(os.path.join(save_dir, f"pe.npy"), all_pe)
np.save(os.path.join(save_dir, f"times.npy"), all_times)

if SAMPLE_EVERY_CRASH:
    all_c_growth = np.array(all_c_growth)
    all_c_crash  = np.array(all_c_crash)
    all_coeffs   = np.vstack([all_c_growth, all_c_crash])
else: 
    all_coeffs = np.array([C_CRASH, C_GROWTH])

np.save(os.path.join(save_dir, "coeffs.npy"), all_coeffs)
