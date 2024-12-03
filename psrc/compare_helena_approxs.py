"""
Main file for the pedestal transport code.
"""



import numpy as np


""" 
Setup Simulation

1a. Read grid from ELITE input file (eliteinp)
    -> R (Rmaj), A (Area), x_norm (PsiNorm)
1b. Read Bootstrap current / alpha stuff from helena output (fort.20)
    -> initial_total_current
    -> Fcirc, Vol, bootstrap_current
1c. Interpolate domain on to evenly spaced grid on psi=0.7-1.0
1d. Read initial conditions from helena input file (fort.10)
    -> n, T : mtanh profiles using ade, bde, cde, dde, ede; ate, bte, cte, dte, ete
2. Calculate initial alpha, jb profiles
    -> alpha, jb : bootstrap is always boosted (added) by initial_current
3. Set stability boundary from MISHKA run (stability_contour.txt)
    -> stability_contour : (alpha_max, jb_max) a line in the alpha-jb plane
4. Check if initial alpha, jb are within stability boundary
    -> if so, move to B 
    -> if not move to A
A. Start with ELM trigger [Set transport coefficients (Chi, D, V, SN, ST)]
    -> Chi, D, V, SN, ST :
    -> evolve for 200 microseconds (200E-6 seconds) 
B. Do ETB phase until stability boundary crossed 
    -> Chi, D, V, SN, ST : 
    -> evolve until stability boundary crossed
6. Repeat A-B until 1 second is reached
"""

plotting = False

""" 
1a. Read grid from ELITE input file (eliteinp)
    -> R (Rmaj), A (Area), x_norm (PsiNorm)
"""

import os 
from helpers import read_eliteinput, calculate_area
BASE_PULSE_DIR = JET_PDB_DIR = "/home/akadam/EUROfusion/2024/data"

shot_num = 83625
eq_fpath = os.path.join(BASE_PULSE_DIR, f"jet_{shot_num}/base_helena")

elite_fname = os.path.join(eq_fpath, 'eliteinp')
elite_data = read_eliteinput(elite_fname)

PsiN_elite     = elite_data['PsiN'][1:-1]
qprofile_elite = elite_data['q'][1:-1]
Rmaj_elite = np.empty(PsiN_elite.shape)
Rmin_elite = np.empty(PsiN_elite.shape)
Area_elite = np.empty(PsiN_elite.shape)
for idx, psi in enumerate(PsiN_elite):
    Rmaj_elite[idx] = elite_data['R'][:, idx].max()
    Rmin_elite[idx] = Rmaj_elite[idx] - Rmaj_elite[0]
    Area_elite[idx] = calculate_area(elite_data['R'][:, idx], elite_data['z'][:, idx])


"""
1.b Read Bootstrap current / alpha stuff from helena output (fort.20)
"""
from helpers import get_data_from_f20
f20_fname = os.path.join(eq_fpath, 'fort.20')
_, vol_f20, volp_f20, j_f20, _, pe_f20, ne_f20, te_f20, _, _, alpha_f20, _, fcirc_f20, jb_f20 = get_data_from_f20(f20_fname)


"""
1.c Interpolate domain on to evenly spaced grid on psi=0.7-1.0
"""

NX = 300
PsiN_interp = np.linspace(0.7, np.max(PsiN_elite), NX)

from helpers import interpolate_data_with_spline

Rmin_interp = interpolate_data_with_spline(PsiN_elite, Rmin_elite, PsiN_interp)
Rmaj_interp = interpolate_data_with_spline(PsiN_elite, Rmaj_elite, PsiN_interp)
Area_interp = interpolate_data_with_spline(PsiN_elite, Area_elite, PsiN_interp)
volp_interp  = interpolate_data_with_spline(PsiN_elite, volp_f20, PsiN_interp)
vol_interp  = interpolate_data_with_spline(PsiN_elite, vol_f20, PsiN_interp)
q_interp    = interpolate_data_with_spline(PsiN_elite, qprofile_elite, PsiN_interp)
fcirc_interp = interpolate_data_with_spline(PsiN_elite[:-1], fcirc_f20[:-1], PsiN_interp)
j_interp = interpolate_data_with_spline(PsiN_elite, j_f20, PsiN_interp)
alpha_interp = interpolate_data_with_spline(PsiN_elite, alpha_f20, PsiN_interp)
eps_interp   = Rmin_interp/Rmaj_interp

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
# Plot comparisons of interpolated data with original 

if plotting == True: 
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs = axs.ravel() 
    axs[0].plot(PsiN_elite, Rmaj_elite, 'o', label='Original')
    axs[0].plot(PsiN_interp, Rmaj_interp, '-', label='Interpolated')
    axs[0].plot(PsiN_elite, Rmin_elite, 'o', label='Original')
    axs[0].plot(PsiN_interp, Rmin_interp, '-', label='Interpolated')
    axs[0].set_title('Rmaj (m) | Rmin (m)')

    axs[1].plot(PsiN_elite, Area_elite, 'o', label='Original')
    axs[1].plot(PsiN_interp, Area_interp, '-', label='Interpolated')
    axs[1].set_title('Area (m$^2$)')

    axs[2].plot(PsiN_elite, vol_f20, 'o', label='Original, VOL')
    axs[2].plot(PsiN_interp, vol_interp, '-', label='Interpolated, VOL')

    axs[2].plot(PsiN_elite, volp_f20, 'o', label='Original, VOLP')
    axs[2].plot(PsiN_interp, volp_interp, '-', label='Interpolated, VOLP')
    axs[2].set_title('Volume (m$^3$)')
    axs[2].legend()

    axs[3].plot(PsiN_elite, qprofile_elite, 'o', label='Original')
    axs[3].plot(PsiN_interp, q_interp, '-', label='Interpolated')
    axs[3].set_title('q')

    axs[4].plot(PsiN_elite, fcirc_f20, 'o', label='Original')
    axs[4].plot(PsiN_interp, fcirc_interp, '-', label='Interpolated')
    axs[4].set_title('Frac Circulating particles')

    axs[5].plot(PsiN_elite, j_f20, 'o', label='Original')
    axs[5].plot(PsiN_interp, j_interp, '-', label='Interpolated')
    axs[5].set_title('Bootstrap current')

    for ax in axs: 
        ax.set_xlim(0.7, 1.0)

"""
1d. Read initial conditions from helena input file (fort.10)
    -> n, T : mtanh profiles using ade, bde, cde, dde, ede; ate, bte, cte, dte, ete
"""

from helpers import read_namelist, mtanh, eped_profile_helena, calculate_pressure

fort10_fname = os.path.join(eq_fpath, 'fort.10')
helnml = read_namelist(fort10_fname)

if helnml['phys']['idete'] == 5: 
    temperature = mtanh(PsiN_interp, helnml['phys']['ate'], helnml['phys']['bte'], helnml['phys']['cte'], helnml['phys']['ddte'], helnml['phys']['ete'])
    density = mtanh(PsiN_interp, helnml['phys']['ade'], helnml['phys']['bde'], helnml['phys']['cde'], helnml['phys']['dde'], helnml['phys']['ede'])
else:
    temperature = eped_profile_helena(PsiN_interp, helnml['phys']['ate'], helnml['phys']['bte'], helnml['phys']['cte'], helnml['phys']['ddte'], helnml['phys']['ete'], helnml['phys']['fte'], helnml['phys']['gte'], helnml['phys']['hte'])
    density = eped_profile_helena(PsiN_interp, helnml['phys']['ade'], helnml['phys']['bde'], helnml['phys']['cde'], helnml['phys']['dde'], helnml['phys']['ede'], helnml['phys']['fde'], helnml['phys']['gde'], helnml['phys']['hde'])

""" FROM REAL WORLD OUTPUT SECTION OF FORT:20"""
RPE = 0.5
CPSURF = 0.5988

pressure = calculate_pressure(temperature, density)

BT = helnml['phys']['bvac']
Zeff = helnml['phys']['zeff']

if plotting == True:
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].plot(PsiN_elite, ne_f20*1E-19, 'o', label='Original')
    axs[0].plot(PsiN_interp, density, label='Density')
    axs[1].plot(PsiN_elite, (te_f20) / 1000.0, 'o', label='Original')
    axs[1].plot(PsiN_interp,temperature, label='Temperature')
    axs[2].plot(PsiN_elite, pe_f20 / 1E3, 'o', label='Original')
    axs[2].plot(PsiN_interp, pressure / 1E3, label='Pressure')
    axs[0].set_title('Density 10$^{19}$ m$^{-3}$')
    axs[1].set_title('Temperature (keV)')
    axs[0].set_ylim(0, 7.5)
    axs[1].set_ylim(0, 1.5)
    axs[0].legend()

"""
2. Calculate initial alpha, jb profiles
    -> alpha (calculate_alpha)
        | PsiN, Vol, Pressure, Rmaj
    -> jb (calculate_zjbt)
        | Fcirc, density, density_gradient, temperature, temperature_gradient, q, Rmaj, Inverse aspect ratio (Rmin / Rmaj)
        | Zeff, Zmain

    -> alpha, jb : bootstrap is always boosted (added) by initial_current
"""

from helpers import calculate_alpha, calculate_zjbt, calculate_pressure, mu_0



bootstrap_approx = calculate_zjbt(
    fcirc_interp, 
    density, density, np.gradient(density, PsiN_interp), np.gradient(density, PsiN_interp),
    temperature, temperature, np.gradient(temperature, PsiN_interp),np.gradient(temperature, PsiN_interp),
    q_interp, Rmaj_interp, eps_interp, Zeff, BT, 
    # COLLMULT=0.1
)

# SPLINE(NPSI,CS,P0,DP0,DPE,1,P1,P2,P3,P4)
# SPLINE(N,X,Y,ALFA,BETA,TYP,A,B,C,D)
alpha_approx = calculate_alpha(PsiN_interp, volp_interp, pressure, Rmaj_interp)
xiab = (mu_0 * Rmaj_interp / BT**2)
baseline_current = j_interp - bootstrap_approx

""" STUFF FOR HELENA ALPHA APPROX """
rho_interp = np.sqrt(vol_interp / vol_interp[-1])
cs_interp  = np.sqrt(PsiN_interp)
drhods_interp = cs_interp / (rho_interp*vol_interp[-1])*volp_interp
pscale =  (BT**2 * eps_interp) / (mu_0 * pressure[0]**2)
normed_pressure = (pressure / pscale) / 0.75E8

P2 = np.gradient(normed_pressure, PsiN_interp) 

alpha_approx   = ((rho_interp / cs_interp**2)) * drhods_interp * (-P2*2) * eps_interp**3 / (CPSURF**2) * vol_interp[-1]*np.sqrt(vol_interp[-1])

# CS Var is np.sqrt(PsiN_elite))
# 
if plotting == True:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(PsiN_interp, alpha_interp, 'o', label='Original')
    axs[0].plot(PsiN_interp, alpha_approx, label='Approx')
    psin_maxalpha = PsiN_interp[np.argmax(alpha_approx)]
    axs[0].axvline(psin_maxalpha, color='red', linestyle='--', label='Max Alpha')
    psin_maxalpha_orig = PsiN_interp[np.argmax(alpha_interp)]
    axs[0].axvline(psin_maxalpha_orig, color='green', linestyle='--', label='Max Alpha Original')

    axs[1].plot(PsiN_interp, j_interp , 'o', label='Original')
    axs[1].plot(PsiN_interp, bootstrap_approx, label='Approx')
    axs[1].plot(PsiN_interp, baseline_current, label='New Baseline')
    axs[1].plot(PsiN_interp, baseline_current + bootstrap_approx, label='Total Current Approx')
    
    axs[0].set_title('Alpha')
    axs[1].set_title('Bootstrap')
    axs[0].legend()
    axs[1].legend()

""" 
3. Stability Boundary
"""
from helpers import get_max_alpha_value

stab_cont_fname = os.path.join(BASE_PULSE_DIR, f'jet_{shot_num}/boundary_values_{shot_num}.txt')
stability_contour = np.loadtxt(stab_cont_fname, skiprows=1)

max_alpha_approx, max_alpha_idx_approx = get_max_alpha_value(alpha_approx, PsiN_interp)
max_alpha_original, max_alpha_idx_original = get_max_alpha_value(alpha_interp, PsiN_interp)

max_jb_approx = (baseline_current+  bootstrap_approx)[max_alpha_idx_approx]
max_jb_elite = j_interp[max_alpha_idx_original]

max_alpha_elite = alpha_interp[max_alpha_idx_original]
correcting_factor = max_alpha_elite - max_alpha_approx

if plotting == True:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(stability_contour[:, 0], stability_contour[:, 1])
    ax.scatter(max_alpha_approx, max_jb_approx, label='Approx')
    ax.scatter(max_alpha_approx + correcting_factor, max_jb_approx, label='Corrected')
    ax.scatter(max_alpha_elite, max_jb_elite, label='Original', marker='x')
    ax.set_title('Stability Contour')
    ax.legend()

""" 
Check if initial alpha, jb are within stability boundary

stability_boundary is a line in the alpha-jb plane
need to check if the initial alpha, jb are within the stability boundary
-> algorithm: 
"""
from helpers import is_point_left_of_boundary


is_stable = is_point_left_of_boundary(max_alpha_approx, max_jb_approx, stability_contour)
# print(f"Is left of boundary: {is_left}")

if plotting: 
    # test with grid of points 
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(stability_contour[:, 0], stability_contour[:, 1])
    color = 'red' if is_stable else 'blue'
    ax.scatter(max_alpha_approx, max_jb_approx, color=color, label='Approx')
    for alpha, jb in zip(np.linspace(3, 10, 10), np.linspace(0, 2, 10)): 
        is_left = is_point_left_of_boundary(alpha, jb, stability_contour)
        color = 'red' if is_left else 'blue'
        ax.scatter(alpha, jb, color=color)


""" 
Plot the initial sim conditions vs equilibrium output

1. n, t, p
2. jb, alpha, stability diagram 
"""

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

COLOR_ORIGINAL = 'blue'
COLOR_SETUP    = 'red'

axs = axs.ravel() 

axs[0].scatter(PsiN_elite, ne_f20*1E-19, label='Original', color=COLOR_ORIGINAL)
axs[0].plot(PsiN_interp, density, label='Density', color=COLOR_SETUP)

axs[1].scatter(PsiN_elite, (te_f20) / 1000.0, label='Original', color=COLOR_ORIGINAL)
axs[1].plot(PsiN_interp,temperature, label='Temperature', color=COLOR_SETUP)

axs[2].scatter(PsiN_elite, pe_f20 / 1E3, label='Original', color=COLOR_ORIGINAL)
axs[2].plot(PsiN_interp, pressure / 1E3, label='Pressure', color=COLOR_SETUP)

axs[3].scatter(PsiN_elite, j_f20, label='Original', color=COLOR_ORIGINAL)
axs[3].plot(PsiN_interp, baseline_current + bootstrap_approx, label='Approx', color=COLOR_SETUP)

axs[4].scatter(PsiN_elite, alpha_f20, label='Original', color=COLOR_ORIGINAL)
axs[4].plot(PsiN_interp, alpha_approx, label='Approx', color=COLOR_SETUP)

axs[5].plot(stability_contour[:, 0], stability_contour[:, 1], label='Stability Boundary', color='black')
axs[5].scatter(max_alpha_approx, max_jb_approx, label='Approx', color=COLOR_SETUP)
axs[5].scatter(max_alpha_elite, max_jb_elite, label='Original', color=COLOR_ORIGINAL)


for ax in axs[:5]: 
    ax.set_xlim(PsiN_interp.min(), PsiN_interp.max())
plt.show()