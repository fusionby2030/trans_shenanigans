# To diagnose the run setup of the data driven model, i.e., by checking scans of the parameters and the data, given a shot number
from load_data import * 
import sys 
from helpers import mtanh , calculate_pressure, calculate_bohmdiffusion, calculate_bohmgrybohm_from_jardin

import scipy 

BASE_PULSE_DIR = "/home/akadam/EUROfusion/2024/data"
JET_PDB_DIR    = "/home/akadam/EUROfusion/2024/data"


shot_num = int(sys.argv[1])
print("Shot number: ", shot_num)

""" 
Loading experimental data
- JET pedestal database 
- relevant HRTS profiles and machine parameters
"""

pulse = load_single_pulse_struct(os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}"))
relevant_profiles = pulse.return_profiles_in_time_windows()
te_fit_params, ne_fit_params, pe_fit_params, machineparams  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)

print(machineparams)

""" 
Setup Radii

"""

N_POINTS    = 1000
R           = np.linspace(machineparams.Rmaj, machineparams.Rmaj + machineparams.Rmin, N_POINTS)
phi         = R**2
phi_norm    = (phi - phi.min()) / (phi.max() - phi.min())
pedestal_region =  phi_norm > 0.8
phi_norm    = phi_norm[pedestal_region]
phi         = phi[pedestal_region]
R           = R[pedestal_region]
dphi_normdr = 2*R / (phi.max() - phi.min())

steady_state_te = mtanh(phi_norm, te_fit_params.h1, te_fit_params.h0, te_fit_params.s, te_fit_params.w, te_fit_params.p)
steady_state_ne = mtanh(phi_norm, ne_fit_params.h1, ne_fit_params.h0, ne_fit_params.s, ne_fit_params.w, ne_fit_params.p)
steady_state_pe = mtanh(phi_norm, pe_fit_params.h1, pe_fit_params.h0, pe_fit_params.s, pe_fit_params.w, pe_fit_params.p)


""" 
Get ELM timings 
"""

normed_tbeo = (pulse.tbeo.data - pulse.tbeo.data.mean() ) / pulse.tbeo.data.std()
delta_t = pulse.tbeo.time[1] - pulse.tbeo.time[0]
distance_between_elms = int(200E-6 / delta_t)*20
threshold_height = normed_tbeo.std()*2
peaks, _ = scipy.signal.find_peaks(normed_tbeo, height=threshold_height, distance=distance_between_elms)
elm_times = pulse.tbeo.time[peaks]
elm_frequency = len(peaks) / (pulse.tbeo.time[-1] - pulse.tbeo.time[0])
t_minus_t_elm = np.empty(len(relevant_profiles.hrts_times))
for i, t in enumerate(relevant_profiles.hrts_times):
    nearest_elm_idx = np.argmin(np.abs(elm_times - t))
    t_nearest_elm = elm_times[nearest_elm_idx]    
    t_minus_t_elm[i] = t - t_nearest_elm


chi_gb, gyroradius = calculate_bohmgrybohm_from_jardin(R, phi, phi_norm, steady_state_te, machineparams)
Dbohm = calculate_bohmdiffusion(steady_state_te, machineparams)


""" 
Heat & particle flux in is given by what the "steady state" would require to be du/dt = 0
"""

heatfluxin     = phi_norm[0]*(chi_gb[0])*np.gradient(steady_state_te, phi_norm)[0]
particlefluxin = (Dbohm[0])*np.gradient(steady_state_ne, phi_norm)[0]*dphi_normdr[0] + intra_params.V[0]*steady_state_ne[0]