import os
from helpers import SimGrid, gather_necessary_data_from_equilibrium, get_max_alpha_value
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt 
import scienceplots 
plt.style.use('science')



@dataclass 
class SimState: 
    grid: SimGrid
    Chi: np.ndarray # Heat diffusion
    D: np.ndarray   # Density diffusion
    V: np.ndarray   # Ware pinch velocity
    SN: np.ndarray  # Density sources
    ST: np.ndarray  # Temperature sources
    time: float = 0.0

    def push(self, dt: float): 
        """
        Push the system forward in time.
        Chain rule: d/dt n = d/dr D d/dr n + D d^2/dr^2 n + d/dr V n + V d/dr n + SN
        ! Chain rule: d/dt T =  Chi d/dA T + A (d/dA Chi d/dA T + Chi d^2/dA^2 T) + ST
        """
        
        
        dndr = np.gradient(self.grid.density, self.grid.Rmaj)
        dTdA = np.gradient(self.grid.temperature, self.grid.Area)
        dChidA = np.gradient(self.Chi, self.grid.Area)
        dVdr = np.gradient(self.V, self.grid.Rmaj)
        dDdr = np.gradient(self.D, self.grid.Rmaj)
        d2ndr2 = np.gradient(dndr, self.grid.Rmaj)
        d2TdA2 = np.gradient(dTdA, self.grid.Area)

        n_new = self.grid.density + dt * (dDdr * dndr + self.D * d2ndr2 + dVdr * self.grid.density + self.V * dndr + self.SN)
        T_new = self.grid.temperature + dt * (self.Chi * dTdA + self.grid.Area * (dChidA * dTdA + self.Chi * d2TdA2) + self.ST)

        """ 
        <--- Boundary conditions --->
        """
        n_new[0] = self.grid.density[0]
        n_new[-1] = self.grid.density[-1]
        T_new[0] = self.grid.temperature[0]
        T_new[-1] = self.grid.temperature[-1]

        self.grid.density     = n_new
        self.grid.temperature = T_new

        self.grid.update_pressure()
        self.grid.update_bootstrap()
        self.grid.update_alpha()

        self.time += dt

def plot_simstate(sim: SimState, figaxs=None):
    if figaxs is None: 
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.ravel() 
        axs[0].set_title(r'$n_e$ [$10^{19}$ m$^{-3}$]')
        axs[1].set_title(r'$T_e$ [keV]')
        axs[2].set_title(r'$p_e$ [kPa]')
        axs[3].set_title(r'$\chi$ [m$^2$/s]')
        axs[4].set_title(r'$D$ [m$^2$/s], $V$ [m/s]')
        axs[5].set_title(r'$S_N$ [1/m$^3$/s], $S_T$ [MW/m$^3$]')
        axs[6].set_title(r'Normalized $j_B$ [-]')
        axs[7].set_title(r'$\alpha$ [-]')
        axs[8].set_title('Ideal MHD Stability Boundary')
        for ax in axs[:5]: 
            ax.set_xlabel(r'$\Psi_N$')
        fig.suptitle(f"Time: {sim.time:.2f} s")

    else: 
        fig, axs = figaxs

    axs[0].plot(sim.grid.psin, sim.grid.density)

    axs[1].plot(sim.grid.psin, sim.grid.temperature)

    axs[2].plot(sim.grid.psin, sim.grid.pressure / 1000.0)

    axs[3].plot(sim.grid.psin, sim.Chi)
    

    axs[4].plot(sim.grid.psin, sim.D)
    axs[4].plot(sim.grid.psin, sim.V)
    
    # axs[4].legend(['D', 'V'])

    axs[5].plot(sim.grid.psin, sim.SN)
    axs[5].plot(sim.grid.psin, sim.ST, ls='--')
    
    axs[5].legend(['SN', 'ST'])

    axs[6].plot(sim.grid.psin, sim.grid.get_total_current())
    
    
    axs[7].plot(sim.grid.psin, sim.grid.alpha) 
    

    axs[8].plot(sim.grid.alpha_bndry, sim.grid.jb_bndry, label='Ideal MHD Stability Boundary', color='black')
    amax, jbmax = sim.grid.get_alpha_jb_max()
    axs[8].scatter(amax, jbmax, label='Current State')
    
    
    axs[0].set_ylim(0, 5)
    axs[1].set_ylim(0, 1.5)
    axs[2].set_ylim(0, 10)

    axs[6].set_ylim(0.0, 1.4)
    axs[7].set_ylim(0, 8)
    axs[8].set_xlim(0, 8)
    axs[8].set_ylim(0, 1.4)

    
    plt.tight_layout()
    return fig, axs


BASE_PULSE_DIR = JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"

shot_num = 83625
NX = 300
eq_fpath = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}/base_helena")
stab_cont_fname = os.path.join(BASE_PULSE_DIR, f'jet_{shot_num}/boundary_values_{shot_num}.txt')

simgrid: SimGrid = gather_necessary_data_from_equilibrium(eq_fpath, stab_cont_fname, NX=NX)


chi0 = np.ones_like(simgrid.psin) * 1e-3
d0 = np.zeros_like(simgrid.psin) * 1e-3
v0 = np.zeros_like(simgrid.psin) * 1e-3
sn0 = (1 / simgrid.density )*10.0
st0 = np.ones_like(simgrid.psin) * 1e-3

sim = SimState(simgrid, chi0, d0, v0, sn0, st0, time=0.0)


figaxs = plot_simstate(sim)

dt         = 1E-2
TOTAL_TIME = dt*10 

while sim.time < TOTAL_TIME:
    sim.push(dt)
    

plot_simstate(sim, figaxs)

plt.show()