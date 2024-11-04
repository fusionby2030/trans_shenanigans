import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scienceplots 
plt.style.use('science')

def plot_density_evolution(simulation_file):

    times = []
    densities = []
    sources   = []
    resevoir  = []
    pressure  = []
    x = None
    
    # Read data from         
    with h5py.File(simulation_file, 'r') as f:
        time_keys = [_key for _key in list(f.keys()) if 'time' in _key]
        times     = [float(time.split('_')[1]) for time in time_keys]
        # time = list(f.keys())[0]
        # times.append(float(time.split('_')[1]))
        print(times)
        for time in time_keys:
            densities.append(np.array(f[time]['n']))
            sources.append(np.array(f[time]['S']))
            if x is None:
                x = np.array(f[time]['X'])
            resevoir.append(np.array(f[time]['resevoir']))
            pressure.append(np.array(f[time]['p']))

        ne_steady_state = np.array(f['setup']['ne_steady_state'])
        particle_flux_out = np.array(f['setup']['particle_flux_out'])
        C_CRASH = np.array(f[time_keys[0]]['C_ETB'])
        C_ETB   = np.array(f[time_keys[0]]['C_CRASH'])
        nx = np.array(f['setup']['nx'])
        ne_p   = np.array(f['setup']['ne_p'])
        ne_w   = np.array(f['setup']['ne_w'])

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(231, projection='3d')
    
    T, X = np.meshgrid(times, x)
    Z = np.array(densities).T
    
    ax.plot_surface(X, T, Z, cmap='viridis')
    
    ax.set_ylabel('Time')
    ax.set_xlabel('Spatial Coordinate X')
    ax.set_zlabel('Density n')
    ax.set_title('Density Evolution in Time')
    
    ax2 = fig.add_subplot(232, projection='3d')
    Z = np.array(sources).T
    ax2.plot_surface(X, T, Z, cmap='viridis')

    pres_ax = fig.add_subplot(233, projection='3d')
    Z = np.array(pressure).T
    pres_ax.plot_surface(X, T, Z, cmap='viridis')

    ax3 = fig.add_subplot(234)
    for _t in range(len(times)):
        color = plt.cm.viridis(_t / len(times))
        ax3.plot(x, densities[_t], label=f'time = {times[_t]:.2f}', color=color)

    ax3.plot(x, ne_steady_state, label='Steady State', color='black', ls='--')
    ax3.set_xlabel('Spatial Coordinate X')
    ax3.set_ylabel('Density n')
    ax4 = fig.add_subplot(235)
    for _t in range(len(times)):
        color = plt.cm.viridis(_t / len(times))
        ax4.plot(x, sources[_t], label=f'time = {times[_t]:.2f}', color=color)

    ax4.set_title('Source (m$^{-3}$ / second)')
    ax4.set_yscale('log')
    ax4.set_ylim(1, 500)
    ax4.set_yticks([1, 10, 100], ["$10^{19}$", "$10^{20}$", "$10^{21}$"])
    pres2_ax = fig.add_subplot(236)
    alpha_ax = pres2_ax.twinx()

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    axs[0].scatter(times, resevoir)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Resevoir ($10^{19}$ particles)')
    axs[0].set_title('Resevoir Evolution in Time')

    ne_ped_loc = np.argmin(abs(x - (ne_p - ne_w)))

    # max pressure grad val 
    for _t in range(len(times)): 
        color = plt.cm.viridis(_t / len(times))
        color2 = plt.cm.gnuplot(_t / len(times))
        ped_bool = x > 0.9
        max_pres_grad = np.max(abs(np.gradient(pressure[_t], x))[ped_bool])
        axs[1].scatter(times[_t], max_pres_grad, color=color)
        pres2_ax.plot(x, pressure[_t],  color=color)
        alpha_ax.plot(x, abs(np.gradient(pressure[_t], x)), color=color)
        
        axs[2].scatter(times[_t], densities[_t][ne_ped_loc], color=color)
    axs[1].set_title('Max pressure gradient')
    axs[2].set_title('Density')
    for ax in axs: 
        ax.set_xlabel('time (s)')

    fig.suptitle('C_CRASH = {:>5.4} | C_ETB = {:>5.4}'.format(C_CRASH, C_ETB))
    plt.show()

# Example usage
results_file = "./results/simulation_output.h5"
plot_density_evolution(results_file)
