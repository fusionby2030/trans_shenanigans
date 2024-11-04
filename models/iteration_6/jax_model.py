import jax.numpy as jnp
import jax
import h5py
import numpy as np
import os
from dataclasses import dataclass
from helpers import * # TransParams, setup_base_transparams
from scipy.integrate import trapezoid
import enum 

# ENUM for the phases, 0 for inter-elm, 1 for intra-elm
class Phase(enum.Enum):
    INTER = 0
    INTRA = 1

@dataclass
class Simulation:
    trans_params: TransParams
    nx: int = 100
    particle_flux_out: float = 0.5
    time: float = 0.0
    elm_phase: Phase = Phase.INTER
    resevoir: float = 0.0
    def __post_init__(self, ):
        self.x, self.dx = jnp.linspace(0.7, 1.0, self.nx, retstep=True)
        
        self.n = self.trans_params.mtanh(self.x, self.trans_params.n_fitparams)
        self.t = self.trans_params.mtanh(self.x, self.trans_params.t_fitparams)
        self.p = self.calculate_pressure(self.t, self.n)
        self.resevoir = 0.0 # trapezoid(self.trans_params.S_N, self.x)            
        # pressure gradient 
        self.dp_dx = abs(jnp.gradient(self.p, self.x))
        self.max_alpha_exp = jnp.max(self.dp_dx)
        print("Initial Resevoir: ", self.resevoir)
    def calculate_pressure(self, te, ne): 
        # kPa
        return (2.0*(te*11604.5)*(ne / 10.0)*1.36064852) / 10000.0
        
def calculate_cfl(sim: Simulation):
    # Compute time step based on CFL condition
    max_velocity = jnp.max(jnp.abs(sim.trans_params.V))
    max_diffusion = jnp.max(sim.trans_params.D)
    dt = 0.5 * min(sim.dx**2 / (2 * max_diffusion), sim.dx / max_velocity)
    return dt

# @jax.jit
# def calculate_source(density, c_etb):
#     # Calculate source term S = 1 / (n ^ C_etb)
#     # Scaling factor, as ne is in the order of 1e19
#     # and S is on order of 1E21
#     scaling_factor = 1e2
#     S = 1.0 / (density ** c_etb)
#     return S * scaling_factor
# 

def push_sim(sim: Simulation, dt):
    # Compute spatial derivatives using central differences
    if sim.elm_phase == Phase.INTER: 
        current_d = scale_d_inter(sim.x, sim.trans_params.D, sim.trans_params)
    else: 
        current_d = sim.trans_params.D
    dn_dx = (jnp.roll(sim.n, -1) - jnp.roll(sim.n, 1)) / (2 * sim.dx)
    d2n_dx2 = (jnp.roll(sim.n, -1) - 2 * sim.n + jnp.roll(sim.n, 1)) / (sim.dx**2)
    dD_dx = (jnp.roll(current_d, -1) - jnp.roll(current_d, 1)) / (2 * sim.dx)
    dV_dx = (jnp.roll(sim.trans_params.V, -1) - jnp.roll(sim.trans_params.V, 1)) / (2 * sim.dx)

    # Calculate source term
    if sim.elm_phase == Phase.INTER and sim.resevoir >= 0.0:
        S = sim.trans_params._source(sim.x, sim.trans_params.n_fitparams, sim.n)*10.0
    else: 
        S = jnp.zeros_like(sim.n)

    # Update 'n' using expanded partial derivative
    n_new = sim.n + dt * (current_d * d2n_dx2 + sim.trans_params.V * dn_dx + dD_dx * dn_dx + dn_dx * dV_dx + S)
    # Flux out is everything inside the last d_dx at last cell, 
    # i.e., Gamma = D * dn_dx + V * n
    flux_out = (current_d*dn_dx + sim.trans_params.V*sim.n)[-1]

    if sim.elm_phase == Phase.INTRA: 
        # Fill up the resevoir as particls are being ejected
        # i.e., add the change in density to the resevoir 
        # During the ELM phase, the density is decreasing, so we add the negative of the delta density
        sim.resevoir += max(0.0, trapezoid(n_new - sim.n, dx=sim.dx)) # * dt
    else: 
        # during the INTER-ELM phase, the density is being fueled by the resevoir 
        # We can not use the n_new - sim.n as this will have core source as well, 
        # where we are just interested in the resevoir contribution from the edge (S)
        sim.resevoir -= trapezoid(S, dx=sim.dx) * dt
        # print(flux_out*dt, trapezoid(S, dx=sim.dx) * dt)
        sim.resevoir += dt*flux_out # dt*(current_d[-1] * d2n_dx2[-1] + sim.trans_params.V[-1] * dn_dx[-1] + dD_dx[-1] * dn_dx[-1] + dn_dx[-1] * dV_dx[-1])
    # Apply boundary conditions
    n_new = n_new.at[0].set(sim.n[0])  # Fixed value at inner boundary
    n_new = n_new.at[-1].set(sim.n[-1])  # Fixed value at outer boundary
    # current_flux = sim.trans_params.D[-1] * d2n_dx2[-1] + sim.trans_params.V[-1] * dn_dx[-1]
    # n_new = n_new.at[-1].set(sim.n[-1] + dt * ((sim.particle_flux_out / dt - current_flux)) )  # Flux out condition at outer boundary

    # print(sim.particle_flux_out / dt, current_flux, sim.n[-1], S[-1], n_new[-1], )
    # print("{:>8.2e} | {:>5.2e} | {:>5.2e} | {:>5.2e} | {:>5.2e}".format(sim.particle_flux_out / dt, current_flux, sim.n[-1], S[-1], n_new[-1]))
    #print('{:>8} | {:>8} | {:>8} | {:>8} | {:>8}'.format('Flux_out', 'Current', 'n[-1]', 'S[-1]', 'n_new[-1]'))
#
    #print("{:>8.2e} | {:>8.2e} | {:>8.2e} | {:>8.2e} | {:>8.2e}".format(sim.particle_flux_out / dt, current_flux, sim.n[-1], S[-1], n_new[-1]))

    # Update simulation state
    sim.n = n_new
    sim.time += dt
    sim.trans_params.S_N = S  # Update source term in simulation
    sim.p = sim.calculate_pressure(sim.t, sim.n)

    return sim



def save_to_hdf5(sim: Simulation, filename="simulation_output.h5"):
    with h5py.File(filename, "a") as f:
        grp = f.create_group(f"time_{sim.time:.6f}")
        grp.create_dataset("D", data=np.array(sim.trans_params.D))
        grp.create_dataset("V", data=np.array(sim.trans_params.V))
        grp.create_dataset("X", data=np.array(sim.x))
        grp.create_dataset("C_ETB", data=sim.trans_params.C_ETB)
        grp.create_dataset("C_CRASH", data=sim.trans_params.C_CRASH)
        grp.create_dataset("n", data=np.array(sim.n))
        grp.create_dataset("S", data=np.array(sim.trans_params.S_N))
        grp.create_dataset("resevoir", data=sim.resevoir)
        grp.create_dataset("p", data=np.array(sim.p))
        grp.create_dataset("t", data=np.array(sim.t))

        if sim.time <= 0.0: 
            setup_grp = f.create_group("setup")
            setup_grp.create_dataset("nx", data=sim.nx)
            setup_grp.create_dataset("particle_flux_out", data=sim.particle_flux_out)
            setup_grp.create_dataset("ne_steady_state", data=sim.trans_params._mtanh(sim.x, sim.trans_params.n_fitparams))
            setup_grp.create_dataset('ne_p', data=sim.trans_params.n_fitparams.p)
            setup_grp.create_dataset('ne_w', data=sim.trans_params.n_fitparams.w)
        # grp.create_dataset("time", data=sim.time)


simfolder = '/home/akadam/EUROfusion/2024/pedestal_transport/models/iteration_6/results'
fname = os.path.join(simfolder, f'simulation_output.h5')
if os.path.exists(fname): 
    os.remove(fname)

# intra_trans_params = setup_base_intertransparams(83624)
c_crash = 0.3
c_etb = 0.5
intra_trans_params = setup_base_intratransparams(83624, c_crash, c_etb)
sim = Simulation(
    trans_params=intra_trans_params, 
    nx=100,
    particle_flux_out=0.1,
)


# intra-elm timescale = 200 microseconds 
tau_intraelm = 200e-6

total_time = 0.3
simtime    = 0.0
wstep      = tau_intraelm / 3.0
wout       = 0.0

t_last_elm = 0.0
# Save initial state 
save_to_hdf5(sim, fname)

sim.elm_phase = Phase.INTRA

while sim.time < total_time:
    dt = calculate_cfl(sim)
    sim = push_sim(sim, dt)
    simtime += dt
    sim.time = simtime
    wout    += dt
    t_last_elm += dt

    print_statement = "Time: {:>5.4f} | dt: {:>5.2e}".format(sim.time, dt)
    # print("Time: {:>5.4f} | dt: {:>5.2e}".format(sim.time, dt))

    if wout >= wstep:
        save_to_hdf5(sim, fname)
        wout = 0.0
    print_statement += " | Resevoir: {:>5.4e}".format(sim.resevoir)
    print_statement += " | Phase: {}".format(sim.elm_phase)
    # Check to see if we have reached the end of the intra-elm phase
    if sim.elm_phase == Phase.INTRA and t_last_elm < tau_intraelm:
        pass 
    elif sim.elm_phase == Phase.INTRA and t_last_elm >= tau_intraelm:
        sim.elm_phase = Phase.INTER
        inter_trans_params = setup_base_intertransparams(83624, c_etb, c_crash)
        sim.trans_params = inter_trans_params
        t_last_elm = 0.0
        sim.trans_params._CURRENT_C_ETB = get_c_inter_over_time(t_last_elm, sim.trans_params.C_ETB)
    else: 
        
        sim.trans_params._CURRENT_C_ETB = get_c_inter_over_time(t_last_elm, sim.trans_params.C_ETB)

        # check for MHD stability 
        pressure = sim.p
        ped_x_bool = sim.x > 0.90
        dp_dx = abs(jnp.gradient(pressure, sim.x))

        # Max pressure grad in edge 
        max_dp_dx = jnp.max(dp_dx[ped_x_bool])
        print_statement += " | Max Pressure Gradient: {:>5.4e}".format(max_dp_dx / sim.max_alpha_exp)
        if max_dp_dx >= sim.max_alpha_exp:
            sim.elm_phase = Phase.INTRA
            t_last_elm = 0.0
            sim.trans_params = intra_trans_params
    # add phase to print statement
    print(print_statement)
    if sim.time >= 100*tau_intraelm: # 0.02 seconds
        break 
    # break 