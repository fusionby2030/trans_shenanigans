import numpy as np 
from interpretive import mtanh, source, pinch_term, ode_system, p_chi, r_chi, p_d, r_d, chi_intraelm, calculate_pressure
from scipy.integrate import solve_ivp
from load_data import * 
import matplotlib.pyplot as plt
import scienceplots 
plt.style.use(['grid', 'science'])


def T_model(x, T, chi, S_T): 
    dTdx = np.gradient(T, x)
    flux = np.gradient(x*chi*dTdx, x)
    dTdt = flux + S_T
    # Boundary conditions 
    dTdt[0]  = 0.0 
    dTdt[-1] = 0.0
    return dTdt 

def n_model(x, n, D, V, S_N): 
    dndx = np.gradient(n, x)
    flux = np.gradient(D*dndx + V*n, x)
    dndt = flux + S_N

    # Left boundary (x = 0), assume the loss is proportional to the density
    dndt[0] = D[0] * dndx[0]  + V[0] * n[0]
    # An influx would have a negative sign. 
    # print(f"Left Boundary: dndx[0] = {dndx[0]}, D[0] = {D[0]}, dndt[0] = {dndt[0]}")
    
    # Right boundary (x = L), similar condition
    dndt[-1] = D[-1] * dndx[-1]  + V[-1]*n[-1] # Same scaling or different for the right side
    return dndt


def solve_pde(x, u0, tinterval: list[float], chi, S_T, tkeep=None): 
    ode_to_handle = lambda t, y: T_model(x, y, chi, S_T)
    sol = solve_ivp(ode_to_handle, tinterval, u0, t_eval=tkeep)
    return sol.y

def solve_npde(x, u0, tinerval: list[float], D, V, S_N, tkeep=None): 
    ode_to_handle = lambda t, y: n_model(x, y, D, V, S_N)
    sol = solve_ivp(ode_to_handle, tinerval, u0, t_eval=tkeep)
    return sol.y

if __name__ == '__main__': 

    PULSE_STRUCT_DIR="/home/akadam/EUROfusion/2024/data/jet_83628"
    JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"



    pulse = load_single_pulse_struct(PULSE_STRUCT_DIR)
    relevant_profiles = pulse.return_profiles_in_time_windows()

    te_fit_params, ne_fit_params, pe_fit_params  = get_fit_params_from_pdbshot(JET_PDB_DIR, pulse.shot_num)

    # Domain for x
    x_start = 0.8
    x_end = 1.0
    x_eval = np.linspace(x_start, x_end, 200)

    """ CHI DETERMINATION """

    # Parameter values, Consumed from Data
    h1Value = te_fit_params.h1
    h0Value = te_fit_params.h0
    sValue =   te_fit_params.s
    wValue =  te_fit_params.w
    pValue = te_fit_params.p
    PED_IDX = np.argmin(abs(x_eval - (pValue  - wValue)))

    steady_state_te = mtanh(x_eval, h1Value, h0Value, sValue, wValue, pValue)

    AValue = 0.0
    muValue = 0.6
    sigmaValue = 0.1
    scalefactorValue = 0.1

    heat_source    = source(x_eval, AValue, muValue, sigmaValue, scalefactorValue)

    C_ETB = 0.5
    CHI_GB  = 3.0 # THIS CONTROLS THE ELM FREQUENCY 
    C_INTRA = 5.00
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
    chi_interelm_vals = chi_solution.y[0]
    chi_intraelm_vals = chi_intraelm(x_eval, CHI_GB, wValue, pValue, C_INTRA)

    """ D DETERMINATION """

    h1Value = ne_fit_params.h1
    h0Value = ne_fit_params.h0
    sValue  = ne_fit_params.s
    wValue  = ne_fit_params.w
    pValue  = ne_fit_params.p

    steady_state_ne = mtanh(x_eval, h1Value, h0Value, sValue, wValue, pValue)
    steady_state_pressure = calculate_pressure(steady_state_te, steady_state_ne)

    AValue = 0.0
    muValue = 1.05
    sigmaValue = 0.02
    scalefactorValue = 0.1

    particle_source    = np.zeros_like(x_eval) # source(x_eval, AValue, muValue, sigmaValue, scalefactorValue)
    # particle_source = np.zeros_like(x_eval)

    D_BOHM = 50.5

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

    D_interelm_vals = D_solution.y[0]
    D_intraelm_vals = chi_intraelm(x_eval, D_BOHM, wValue, pValue, C_INTRA, scaling_factor=1.0)
    V_intraelm_vals = -np.ones_like(x_eval)
    V_interelm_vals = pinch_term(x_eval)



    """ First show ELM evolution with C_INTRA"""

    t0 = steady_state_te
    n0 = steady_state_ne
    # intra elm time is 200 microseconds
    tau_intraelm = 200e-6
    t_interval = [0, tau_intraelm]

    solutions = solve_pde(x_eval, t0, t_interval, chi_intraelm_vals, heat_source)
    solutions_ne = solve_npde(x_eval, n0, t_interval, D_intraelm_vals, V_intraelm_vals, particle_source)
    print(solutions_ne.shape, solutions_ne[0, :])
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(x_eval, t0,  color='black', label='initial')
    axs[0].plot(x_eval, solutions[:, -1], label=f't={t_interval[1]:.3}', color='blue')

    axs[1].plot(x_eval, n0, color='black', label='initial')
    axs[1].plot(x_eval, solutions_ne[:, -1], label=f't={t_interval[1]:.3}', color='blue')
    axs[1].set_title('Density')
    axs[0].set_title('Temperature')
    axs[0].set_ylim(0.0, 1)
    axs[1].set_ylim(0.0, 5.5)
    for ax in axs: 
        ax.legend()
        ax.axvline(x_eval[PED_IDX])
    plt.show()

    """ Then Evolve from there"""


    t0 = solutions[:, -1]
    n0 = solutions_ne[:, -1]

    # Solve the PDE
    t_interval = [0, 0.1]
    t_eval = np.linspace(t_interval[0], t_interval[1], 100)

    solutions = solve_pde(x_eval, t0, t_interval, chi_vals, heat_source, tkeep=t_eval)
    solutions_ne = solve_npde(x_eval, n0, t_interval, D_interelm_vals, V_interelm_vals, particle_source, tkeep=t_eval)

    fig, axs = plt.subplots(1, 3, figsize=(8, 8))
    # ax.plot(x_eval, t0,  color='black', label='initial')
    axs[0].plot(x_eval, t0, label=f'initial', color='black')
    axs[0].plot(x_eval, solutions[:, -1], label=f't={t_interval[1]:.3}', color='blue', ls='--')
    axs[1].plot(x_eval, n0, color='black', label='initial')
    axs[1].plot(x_eval, solutions_ne[:, -1], label=f't={t_interval[1]:.3}', color='blue', ls='--')

    for i in [0, 1, 2, 3, 4]: 
        idx = i*((len(t_eval)-1)//4)

        axs[0].plot(x_eval, solutions[:, idx], label=f't={t_eval[idx]:.3}')
        axs[1].plot(x_eval, solutions_ne[:, idx], label=f't={t_eval[idx]:.3}')
        pressure_time = calculate_pressure(solutions[:, idx], solutions_ne[:, idx])
        
        axs[2].plot(x_eval, abs(np.gradient(pressure_time, x_eval)))


    axs[0].plot(x_eval, steady_state_te, label='desired', color='red', ls='--')
    axs[1].plot(x_eval, steady_state_ne, label='desired', color='red', ls='--')


    axs[2].plot(x_eval, abs(np.gradient(steady_state_pressure, x_eval)), ls='--', color='black')
    for ax in axs: 
        ax.axvline(x_eval[PED_IDX])

    axs[0].legend()


    plt.show()


    """ 
    Evolve completely, starting from steady state te
    Step at increments of tau_interlm / 2.0
    """

    D_vals = D_interelm_vals
    V_vals = V_interelm_vals
    chi_vals = chi_interelm_vals

    initial_temperature = steady_state_te
    initial_density = steady_state_ne

    ALPHA_CRIT = max(abs(np.gradient(calculate_pressure(initial_temperature, initial_density), x_eval)))
    print(ALPHA_CRIT)
    INTRA_PHASE = False
    T_LAST_ELM = 0.0
    t_beg, t_fin = 0.0, 0.5
    num_time_steps = int(t_fin / (tau_intraelm / 2.0))

    t_eval = np.linspace(t_beg, t_fin, num_time_steps)

    temperature_profiles_all = np.empty((num_time_steps, len(initial_temperature)))
    density_profiles_all = np.empty((num_time_steps, len(steady_state_ne)))
    temperature_profiles_all[0] = initial_temperature
    density_profiles_all[0] = initial_density

    for i in range(num_time_steps-1): 
        t_init, t_end = t_eval[i], t_eval[i+1]
        t_interval = [t_init, t_end]

        solutions = solve_pde(x_eval, initial_temperature, t_interval, chi_vals, heat_source)
        solutions_ne = solve_npde(x_eval, initial_density, t_interval, D_vals, V_vals, particle_source)

        temperature_profiles_all[i+1] = solutions[:, -1]
        initial_temperature = solutions[:, -1]
        density_profiles_all[i+1] = solutions_ne[:, -1]
        initial_density = solutions_ne[:, -1]


        pressure_slice = calculate_pressure(initial_temperature, initial_density)

        if (max(abs(np.gradient(pressure_slice, x_eval))) > ALPHA_CRIT) and INTRA_PHASE is False: 
            chi_vals = chi_intraelm_vals
            D_vals = D_intraelm_vals
            V_vals = V_intraelm_vals
            INTRA_PHASE = True 

        if INTRA_PHASE is True and T_LAST_ELM >= tau_intraelm: 
            T_LAST_ELM = 0.0
            INTRA_PHASE = False 
            chi_vals = chi_interelm_vals
            D_vals = D_interelm_vals
            V_vals = V_interelm_vals

        
        T_LAST_ELM += (t_end - t_init)
        print(t_end, INTRA_PHASE, T_LAST_ELM, max(abs(np.gradient(pressure_slice, x_eval))))


    pressure_profiles_all = calculate_pressure(temperature_profiles_all, density_profiles_all)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    axs[0].plot(t_eval, temperature_profiles_all[:, PED_IDX])
    axs[0].axhline(y=steady_state_te[PED_IDX], color='red', ls='--')

    axs[1].plot(t_eval, density_profiles_all[:, PED_IDX])
    axs[1].axhline(y=steady_state_ne[PED_IDX], color='red', ls='--')
    axs[2].plot(t_eval, pressure_profiles_all[:, PED_IDX])
    axs[2].axhline(y=steady_state_pressure[PED_IDX], color='red', ls='--')

    # for i in range(num_time_steps): 
    #     plt.plot(x_eval, temperature_profiles_all[i])

    plt.show()


