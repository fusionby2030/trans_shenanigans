import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid', 'retro'])
import os
import numpy as np
import re
from dataclasses import dataclass, field
from typing import List
import pickle
import sys 

@dataclass
class SimulationState:
    nx: int
    nghosts: int
    dx: float
    pedestal_loc: float
    power_input: float
    c_etb: float
    c_crash: float
    pressure_grad_threshold: float
    chi_0: float
    psin: List[float] | np.ndarray
    time_steps: List[float] | np.ndarray = field(default_factory=list)
    mode_flags: List[bool] | np.ndarray = field(default_factory=list)
    temperature: List[np.ndarray] | np.ndarray = field(default_factory=list)
    density: List[np.ndarray] | np.ndarray = field(default_factory=list)
    pressure: List[np.ndarray] | np.ndarray = field(default_factory=list)
    alpha: List[np.ndarray] | np.ndarray = field(default_factory=list)
    trans_chi: List[np.ndarray] | np.ndarray = field(default_factory=list)
    trans_D: List[np.ndarray] | np.ndarray = field(default_factory=list)
    trans_V: List[np.ndarray] | np.ndarray = field(default_factory=list)

def parse_fortran_file(file_path: str) -> SimulationState:
    """
    Parse a Fortran ASCII file and extract simulation data into a SimulationState dataclass.

    Parameters:
        file_path (str): Path to the Fortran ASCII file.

    Returns:
        SimulationState: An instance of the dataclass containing parsed simulation data.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract header parameters
    nx = int(re.search(r'nx\s+(\d+)', lines[0]).group(1))
    nghosts = int(re.search(r'nghost\s+(\d+)', lines[1]).group(1))
    dx = float(re.search(r'dx\s+([\d.]+)', lines[2]).group(1))
    psin = [float(val) for val in re.search(r'psin\s+([\d.\s-]+)', lines[3]).group(1).split()]
    power_input = float(re.search(r'power_input\s+([\d.]+)', lines[4]).group(1))
    pedestal_loc = float(re.search(r'pedestal_loc\s+([\d.]+)', lines[5]).group(1))
    c_etb = float(re.search(r'c_etb\s+([\d.]+)', lines[6]).group(1))
    c_crash = float(re.search(r'c_crash\s+([\d.]+)', lines[7]).group(1))
    pressure_grad_threshold = float(re.search(r'pressure_grad_threshold\s+([\d.]+)', lines[8]).group(1))
    chi_0 = float(re.search(r'chi_0\s+([\d.]+)', lines[9]).group(1))

    # Variables to hold time steps and data
    time_steps = []
    mode_flags = []
    temperature_data = []
    density_data = []
    pressure_data = []
    alpha_data = []
    trans_chi_data = []
    trans_D_data = []
    trans_V_data = []

    # Define regular expressions to match different variables
    time_pattern = re.compile(r'tout\s+([\d.-]+)')
    mode_pattern = re.compile(r'mode\s+(\w)')
    T_pattern = re.compile(r'temperature\s+([\d.\s-]+)')
    n_pattern = re.compile(r'density\s+([\d.\s-]+)')
    p_pattern = re.compile(r'pressure\s+([\d.\s-]+)')
    alpha_pattern = re.compile(r'alpha\s+([\d.\s-]+)')
    chi_pattern = re.compile(r'trans_chi\s+([\d.\s-]+)')
    D_pattern = re.compile(r'trans_D\s+([\d.\s-]+)')
    V_pattern = re.compile(r'trans_V\s+([\d.\s-]+)')

    # Iterate over the lines and extract data
    for line in lines[4:]:
        time_match = time_pattern.search(line)
        if time_match:
            time_steps.append(float(time_match.group(1)))

        mode_match = mode_pattern.search(line)
        if mode_match:
            mode_flags.append(mode_match.group(1) == 'T')

        # Extract data for temperature, density, pressure, alpha, chi, D, V
        T_match = T_pattern.search(line)
        if T_match:
            temperature_data.append([float(val) for val in T_match.group(1).split()])

        n_match = n_pattern.search(line)
        if n_match:
            density_data.append([float(val) for val in n_match.group(1).split()])

        p_match = p_pattern.search(line)
        if p_match:
            pressure_data.append([float(val) for val in p_match.group(1).split()])

        alpha_match = alpha_pattern.search(line)
        if alpha_match:
            alpha_data.append([float(val) for val in alpha_match.group(1).split()])

        chi_match = chi_pattern.search(line)
        if chi_match:
            trans_chi_data.append([float(val) for val in chi_match.group(1).split()])

        D_match = D_pattern.search(line)
        if D_match:
            trans_D_data.append([float(val) for val in D_match.group(1).split()])

        V_match = V_pattern.search(line)
        if V_match:
            trans_V_data.append([float(val) for val in V_match.group(1).split()])

    # Convert lists to NumPy arrays for easier handling
    temperature_array = np.array(temperature_data)
    density_array = np.array(density_data)
    pressure_array = np.array(pressure_data)
    alpha_array = np.array(alpha_data)
    chi_array = np.array(trans_chi_data)
    D_array = np.array(trans_D_data)
    V_array = np.array(trans_V_data)
    psin = np.array(psin)
    time_steps = np.array(time_steps)

    # Create and return the SimulationState dataclass
    return SimulationState(
        nx=nx,
        nghosts=nghosts,
        power_input=power_input,
        pedestal_loc=pedestal_loc,
        c_etb=c_etb,
        c_crash=c_crash,
        pressure_grad_threshold=pressure_grad_threshold,
        chi_0=chi_0,
        dx=dx,
        psin=psin,
        time_steps=time_steps,
        mode_flags=mode_flags,
        temperature=temperature_array,
        density=density_array,
        pressure=pressure_array,
        alpha=alpha_array,
        trans_chi=chi_array,
        trans_D=D_array,
        trans_V=V_array
    )

if not os.path.exists('simulation_state.pkl'):
    simulation_state = parse_fortran_file('./output.txt')
    with open('simulation_state.pkl', 'wb') as f:
        pickle.dump(simulation_state, f)
else:
    with open('simulation_state.pkl', 'rb') as f:
        simulation_state = pickle.load(f)

ped_idx = np.argmin(np.abs(simulation_state.psin - simulation_state.pedestal_loc))

from scipy.signal import find_peaks

N = len(simulation_state.mode_flags)

dt = simulation_state.time_steps[1] - simulation_state.time_steps[0]
num_time_steps_intra_elm =  int(0.0002 / dt)
# Only look at the last half of the simulation
# Find peaks in temperature signal signal
max_alpha = np.max(simulation_state.alpha, axis=1)
maxmax_alpha = min(np.max(max_alpha[N//2:]), simulation_state.pressure_grad_threshold)
# peaks, _ = find_peaks(max_alpha, height=maxmax_alpha-0.75) # simulation_state.pressure_grad_threshold-0.5
ped_temperature = simulation_state.temperature[:, ped_idx]
ped_density     = simulation_state.density[:, ped_idx]
ped_pressure    = simulation_state.pressure[:, ped_idx]

peaks, _ = find_peaks(ped_temperature) # , height=maxmax_alpha-0.75) # simulation_state.pressure_grad_threshold-0.5

elm_freq = sum(peaks >= N//2) / (simulation_state.time_steps[-1] - simulation_state.time_steps[N//2]) #  - simulation_state.time_steps[N//2])
# print(peaks, simulation_state.time_steps[peaks], elm_freq)
# print(len(peaks), elm_freq)

avg_pre_elm_teped = np.mean(ped_temperature[peaks[peaks >= N//2]])
std_pre_elm_teped = np.std(ped_temperature[peaks[peaks >= N//2]])

avg_pre_elm_neped = np.mean(ped_density[peaks[peaks >= N//2]])
std_pre_elm_neped = np.std(ped_density[peaks[peaks >= N//2]])

avg_pre_elm_peped = np.mean(ped_pressure[peaks[peaks >= N//2]])
std_pre_elm_peped = np.std(ped_pressure[peaks[peaks >= N//2]])

if len(peaks) == 0:
    peaks = [1, N//2, N//2 +N//4, N//2 +N//4, N//2 +N//4]
# print(N//2, simulation_state.time_steps[N//2])

with open('./postprocessed_elmcount', 'w') as file:
    file.write(f'c_etb={simulation_state.c_etb}\n')
    file.write(f'c_crash={simulation_state.c_crash}\n')
    file.write(f'elm_freq={elm_freq}\n')
    file.write(f'power_input={simulation_state.power_input}\n')
    file.write(f'teped={avg_pre_elm_teped}\n')
    file.write(f'teped_std={std_pre_elm_teped}\n')
    file.write(f'neped={avg_pre_elm_neped}\n')
    file.write(f'neped_std={std_pre_elm_neped}\n')
    file.write(f'peped={avg_pre_elm_peped}\n')
    file.write(f'peped_std={std_pre_elm_peped}\n')
    

print(f'c_etb={simulation_state.c_etb:.4}, c_crash={simulation_state.c_crash:.4}')
print(f'ELM frequency: {elm_freq:.4}, total elms encountered: {len(peaks)}')
print(f'Teped = {avg_pre_elm_teped:.4}')

if sys.argv[-1] == 'p': 
    dash, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs = axs.ravel()
    # AX 0:  Plot Temperature & Density at pedestal location and 0.8, 0.9, 1.0
    dens_ax = axs[0].twinx()
    for rho in [0.8, 1.0, simulation_state.pedestal_loc]:
        idx = np.argmin(np.abs(simulation_state.psin - rho))
        axs[0].plot(simulation_state.time_steps, simulation_state.temperature[:, idx], label=f'rho = {rho}')
        dens_ax.plot(simulation_state.time_steps, simulation_state.density[:, idx], ls='--')
    axs[0].legend(loc=(-0.5, 0.0), frameon=False)
    # AX 1: Pressure
    for rho in [0.8, 1.0, simulation_state.pedestal_loc]:
        axs[1].plot(simulation_state.time_steps, simulation_state.pressure[:, np.argmin(np.abs(simulation_state.psin - rho))], label=f'rho = {rho}')
    # AX 2: Alpha
    axs[2].plot(simulation_state.time_steps, max_alpha) #simulation_state.alpha[:, np.argmin(np.abs(simulation_state.psin - rho))])
    axs[2].axhline(simulation_state.pressure_grad_threshold, color='black', label='Threshold')
    axs[2].plot(simulation_state.time_steps[peaks], max_alpha[peaks], 'x', color='red', label='ELM')
    axs[2].set_title(f'ELM Frequency = {elm_freq:.2f} Hz')
    dash.suptitle(f'c_etb = {simulation_state.c_etb}, c_crash = {simulation_state.c_crash}, chi_0 = {simulation_state.chi_0}')
    axs[2].legend(frameon=False)
    # AX 3: Transport coefficients
    if len(peaks) > 2: 
        tstart_idx = peaks[-2] # N//2 # peaks[3]
        tend_idx   = peaks[-1] -1 # N//2 + N//4 # peaks[5] # + 5
    else: 
        tstart_idx = N//2 
        tend_idx = N//2 + N//4
    for tidx in range(tstart_idx, tend_idx):
        color = plt.cm.viridis((tidx - tstart_idx) / (tend_idx - tstart_idx))
        axs[3].plot(simulation_state.psin, simulation_state.trans_chi[tidx], color='grey')
        axs[4].plot(simulation_state.psin, simulation_state.temperature[tidx], color=color)
        axs[5].plot(simulation_state.psin, simulation_state.alpha[tidx], color=color)
    axs[3].set_title(f't = {simulation_state.time_steps[tstart_idx]:.2f} - {simulation_state.time_steps[tend_idx]:.2f}')
    axs[3].axhline(simulation_state.chi_0, color='black', label='chi_0')
    axs[3].axvline(simulation_state.pedestal_loc + 0.04383409, color='black', ls='--', label='Pedestal')
    plt.show()
