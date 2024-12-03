import os
import subprocess
import numpy as np
import f90nml

def read_namelist(namelist_file: str) -> f90nml.namelist.Namelist:
    return f90nml.read(namelist_file)

def calculate_area(x, z):
    # Gauss-shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
    # could likely use the determinant approach
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n  # roll over at n
        area += x[i] * z[j]
        area -= z[i] * x[j]
    area = abs(area) / 2.0
    return area 

def read_fortran_ascii(file_path, keywords, N)-> dict[str, np.ndarray]:
    """
    Reads data from a Fortran ASCII file and extracts arrays following specified keywords.

    Parameters:
        file_path (str): Path to the ASCII file.
        keywords (list of str): List of strings to search for in the file.
        N (int): Number of floats to extract for each keyword.

    Returns:
        dict: A dictionary where keys are keywords and values are 1D NumPy arrays of floats.
    """
    extracted_data = {key: None for key in keywords}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    for keyword in keywords:
        # Find the keyword in the lines
        for i, line in enumerate(lines):
            if line.strip().startswith(keyword):
                # Collect the floats following the keyword
                floats = []
                j = i + 1  # Start reading the lines after the keyword
                while j < len(lines) and len(floats) < N:
                    floats.extend(map(float, lines[j].split()))
                    j += 1
                # Truncate or reshape if more/less than expected N values are found
                extracted_data[keyword] = np.array(floats[:N])
                break  # Stop looking for this keyword
    
    return extracted_data

def read_fortran_repeated_arrays(file_path, keywords, N, M) -> dict[str, np.ndarray]:
    """
    Reads data from a Fortran ASCII file and extracts repeated arrays of length N following a keyword.

    Parameters:
        file_path (str): Path to the ASCII file.
        keyword (str): String to search for in the file to start reading data.
        N (int): Number of floats in each array.
        M (int): Number of arrays of length N to extract.

    Returns:
        np.ndarray: A 2D NumPy array of shape (M, N) containing the extracted data.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    extracted_data = {key: None for key in keywords}
    
    for keyword in keywords: 
        data = []
        reading = False  # Flag to start reading after the keyword
        floats_collected = 0  # Track the number of floats collected
        for i, line in enumerate(lines):
            if not reading:
                if line.strip().startswith(keyword):  # Find the keyword
                    reading = True
                    continue
            if reading:
                # Extract floats from the line
                floats = list(map(float, line.split()))
                data.extend(floats)
                floats_collected += len(floats)
                # Stop reading if we've collected M * N floats
                if floats_collected >= M * N:
                    break

        # Reshape the collected data into a 2D array of shape (M, N)
        data = np.array(data[:M * N]).reshape(M, N)
        extracted_data[keyword] = data
    return extracted_data

def read_eliteinput(filepath) -> dict[str, np.ndarray]: 
    """ 
    Reads the following parameters: 
    1D: Psi, dp/dpsi, d2p/dpsi, fpol, ffp, dffp, q, ne, dne/dpsi, Te, dTe/dpsi, Ti, dTi/dpsi, nMainIon, nZ
    2D: R, z, Bp
    Calculates PsiN from Psi
    returns dictionary with above keys
    """
    with open(filepath, 'r') as file: 
        file.readline()
        N, M = file.readline().split()
    N, M = int(N), int(M)

    keywords_Nshape = ['Psi', 'dp/dpsi', 'd2p/dpsi', 'fpol', 'ffp', 'dffp', 'q', 'ne','dne/dpsi', 'Te', 'dTe/dpsi', 'Ti', 'dTi/dpsi', 'nMainIon', 'nZ' ]
    keywords_Mshape = ['R', 'z', 'Bp']

    data_N = read_fortran_ascii(filepath, keywords_Nshape, N)
    data_M = read_fortran_repeated_arrays(filepath, keywords_Mshape, N, M)

    data_N['PsiN'] = (data_N['Psi'] - data_N['Psi'].min()) / (data_N['Psi'].max() - data_N['Psi'].min())
    return {**data_M, **data_N}


def get_data_from_f20(fort_20_file): 
    with open(fort_20_file, 'r') as file: 
        for line in file: 
            if 'P [Pa],' in line: 
                break 
        next_line = file.readline()
        pressures, psis = [], []
        temperature_e, temperature_i, density = [], [], []
        bootstrap = []
        while True: 
            data_stream = file.readline().split()
            if len(data_stream) == 0: 
                break 
            psis.append(float(data_stream[0]))
            pressures.append(float(data_stream[1]))
            density.append(float(data_stream[2]))
            temperature_e.append(float(data_stream[3]))
            temperature_i.append(float(data_stream[4]))
            bootstrap.append(float(data_stream[5]))

        psi_pressure  = np.array(psis)**2
        pressure      = np.array(pressures)
        density       = np.array(density)*1E19
        temperature_e = np.array(temperature_e)
        temperature_i = np.array(temperature_i)
        bootstrap_current = np.array(bootstrap)

    with open(fort_20_file, 'r') as file: 
        for line in file: 
            if 'VOLP' in line: 
                break 
        next_line = file.readline()
        data_stream = file.readline().split()
        array_size = int(data_stream[0])
        psi, s, j, vol, volp, area = np.empty(array_size), np.empty(array_size), np.empty(array_size), np.empty(array_size), np.empty(array_size), np.empty(array_size)
        psi[0] = float(data_stream[1])
        j[0]   = float(data_stream[3])
        vol[0] = float(data_stream[7])
        volp[0] = float(data_stream[8])

        for i in range(1, array_size):
            data_stream = file.readline().split()

            psi[i] = float(data_stream[1])
            j[i]   = float(data_stream[3])
            vol[i] = float(data_stream[7])
            volp[i] = float(data_stream[8])

    
    with open(fort_20_file, 'r') as file: 
        for line in file: 
            if 'FMARG' in line:
                break 
        next_line = file.readline() 
        data_stream = file.readline().split()
        rho, alpha = np.empty(array_size), np.empty(array_size)
        # rho: 3
        # alpha 6
        # alpha1 7
        rho[0]    = float(data_stream[2])
        alpha[0]  = float(data_stream[6])
        for i in range(1, array_size): 
            data_stream = file.readline().split() 
            rho[i]    = float(data_stream[2])
            alpha[i]  = float(data_stream[6])


    # with open(fort_20_file, 'r') as file:
    #     for line in file: 
    #         if "* I,     X,          PSI,          P,         Q   *" in line: 
    #             break 
    #     print(line)
    #     file.readline()
    #     for line in file: 
    #         data_stream = line.split()
    #         index = int(data_stream[0]) - 1
    #         q_val = float(data_stream[-1])
    #         if index >= len(q_profile):
    #             break 
    #         q_profile[-index] = q_val
    # 
    q_profile = np.empty_like(psi)
    Fcirc = np.empty_like(psi)
    with open(fort_20_file, 'r') as file: 
        for line in file: 
            if "Fcirc" in line: 
                break 
        file.readline()
        index = 0
        for line in file: 
            data_stream = line.split() 
            # print(data_stream)
            Fcirc[index] = float(data_stream[2])
            q_profile[index] = float(data_stream[1])
            index += 1
            if index >= len(q_profile) -1: 
                break
    # q_profile_values = [
    # with open(fort_20_file, 'r') as file:
    #     for line in file:
    #         if 'Q PROFILE' in line:
    #             print(line)
    #             break
    #     # Start reading floats after "Q PROFILE :"
    #     file.readline()
    #     for line in file:
    #         print(line)
    #         if '(' not in line:  # Stop if the line does not contain data
    #             break
    #         # Extract the floats from each line
    #         data_stream = line.split()
    #         print(data_stream)
    #         data_stream = [entry for entry in data_stream if '(' in entry]
    #         for entry in data_stream:
    #             value = entry.split('(')[0]  # Extract only the float part
    #             q_profile_values.append(float(value))
    #             if len(q_profile_values) >= 300:  # Stop if we reach 300 values
    #                 break
    #         if len(q_profile_values) >= 300:
    #             break
    # q_profile = np.array(q_profile_values)

    return psi[1:], vol[1:], volp[1:], j[1:], psi_pressure, pressure, density, temperature_e, temperature_i, rho, alpha[1:], q_profile[1:], Fcirc[1:], bootstrap_current

from scipy.interpolate import CubicSpline
def interpolate_data_with_spline(xdata, ydata, xnew): 
    cs = CubicSpline(xdata, ydata)
    return cs(xnew)


def mtanh(x: float | np.ndarray, h1, h0, s, p, w) -> float | np.ndarray:
    # ate, bte, cte, dte, ete (order of arguments)    
    r = (p - x) / (w * 2)
    return (h1 - h0) / 2 * ((1 + s * r) * (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r)) + 1) + h0

def eped_profile_helena(x, a_0, y_sep, a_1, x_mid, delta, x_ped, alpha_1, alpha_2):
    """
    x_mid, x_ped, a_0, a_1, y_sep, alpha_1, alpha_2, delta
    - $a_{T0}$ = ate                                         
    - $T_{sep}$ = bte                                        
    - $a_{T1}$ = cte 
    - $\psi_{mid}$ = ddte                                     
    - $\Delta$ = ete                                       
    - $\psi_{ped}$ = fte                                      
    - $\alpha_{T1}$ = gte                                     
    - $\alpha_{T2}$ = hte 
    """
    # print(f"x_mid={x_mid}, x_ped={x_ped}, a_0={a_0}, "
    #     f"a_1={a_1}, y_sep={y_sep}, alpha_1={alpha_1}, alpha_2={alpha_2}, delta={delta}"
    # )
    profile = np.zeros_like(x)
    for _idx, _x in enumerate(x):
        term1 = (np.tanh(2 / delta * (1 - x_mid)) - np.tanh(2 / delta * (_x - x_mid)))
        if (_x > x_ped):
            term2 = 0.0
        else:
            term2 = a_1 * (1 - (_x / x_ped) ** alpha_1) ** alpha_2
        profile[_idx] = y_sep + a_0 * term1 + term2
    return profile

mu_0 = 4*np.pi*1e-7
def calculate_alpha(psi, volume, pressure, R):
    grad_vol = np.gradient(volume, psi)
    grad_p   = np.gradient(pressure, psi)
    # NOTE: 4 pi factor I assume is coming from the total pressure derivative
    return -2 * grad_vol / (2*np.pi)**2 * (volume / (2*np.pi**2 * R))**0.5 * 4*np.pi * mu_0 * grad_p

def calculate_alpha_helena(psin, pressure, psin_bndry, vol, volp, eps, BT):
    rho = np.sqrt(vol / vol[-1])
    cs  = np.sqrt(psin)
    drhods = cs / (rho*vol[-1]) *volp
    pscale = (BT**2 * eps) / (mu_0 * pressure[0]**2)
    fudge_factor = 0.75E8
    normed_pressure = (pressure / pscale ) / fudge_factor
    pgrad = np.gradient(normed_pressure, psin)

    alpha_approx   = ((rho / cs**2)) * drhods * (-pgrad*2) * eps**3 / (psin_bndry**2) * vol[-1]*np.sqrt(vol[-1])
    return alpha_approx

def calculate_zjbt(
    FCIRC,  # Fraction of circulating particles
    ZZNE,  # Electron density in 1e19 m^-3
    ZZNI,  # Ion density in 1e19 m^-3
    ZDNE,  # Electron density derivative in 1e19 m^-3
    ZDNI,  # Ion density derivative in 1e19 m^-3
    TE,  # Electron temperature in keV
    TI,  # Ion temperature in keV
    DTE,  # Derivative of electron temperature in keV
    DTI,  # Derivative of ion temperature in keV
    q,  # Safety factor
    R,  # Major radius in meters
    EPS,  # Inverse aspect ratio
    ZEFF,  # Effective charge
    BT, # Toroidal field strength in Tesla
    ZMAIN=1.0,  # Hydrogenic charge number
    COLLMULT=1.0,  # Collisional multiplier
    A_I=2.0,  # Ion mass number
):
    # Constants
    E = 1.6021773e-19  # Elementary charge (C)
    ME = 9.1093897e-31  # Electron mass (kg)
    M_I = A_I * 1.6726e-27  # Ion mass (kg)
    
    EPS = EPS + 1e-6
    # Derived quantities
    FT = 1. - FCIRC
    ZNE = ZZNE * 1e19
    ZNI = ZZNI * 1e19
    DNE = ZDNE * 1e19
    DNI = ZDNI * 1e19

    TI = TI * 1000.0
    TE = TE * 1000.0
    DTE = DTE * 1000.0
    DTI = DTI * 1000.0
    
    RPE = ZNE * TE / (ZNI * TI + ZNE * TE)
    ZALPHA = ZMAIN
    ZAVE = ZNE / ZNI
    ZAVE2 = (ZALPHA**2 * ZAVE * ZEFF)**0.25
    
    ZLE = 31.3 - np.log(np.sqrt(ZNE) / TE)
    ZLI = 30.0 - np.log(ZMAIN**3 * np.sqrt(ZNI) / TI**1.5)
    
    ZNUE = 6.921e-18 * q * R * ZNE * ZEFF * ZLE / (TE**2 * EPS**1.5) * COLLMULT
    ZNUI = 4.90e-18 * q * R * ZNI * ZMAIN**4 * ZLI / (TI**2 * EPS**1.5) * COLLMULT
    
    F33TEF = FT / (
        1.0 + 0.25 * (1.0 - 0.7 * FT) * np.sqrt(ZNUE)
        * (1.0 + 0.45 * (ZEFF - 1.0)**0.5)
        + 0.61 * (1.0 - 0.41 * FT) * ZNUE / ZEFF**0.5
    )
    
    ZNZ = 0.58 + 0.74 / (0.76 + ZEFF)
    SIGSPITZ = 1.9012e4 * TE**1.5 / (ZEFF * ZNZ * ZLE)
    X = F33TEF
    
    F31TEF = FT / (
        1.0 + (0.67 * (1.0 - 0.7 * FT)) * np.sqrt(ZNUE)
        / (0.56 + 0.44 * ZEFF)
        + (0.52 + 0.086 * np.sqrt(ZNUE)) * (1.0 + 0.87 * FT) * ZNUE
        / (1 + 1.13 * np.sqrt(ZEFF - 1.0))
    )
    
    ZL31 = (
        (1.0 + 0.15 / (ZEFF**1.2 - 0.71)) * X
        - 0.22 / (ZEFF**1.2 - 0.71) * X**2
        + 0.01 / (ZEFF**1.2 - 0.71) * X**3
        + 0.06 / (ZEFF**1.2 - 0.71) * X**4
    )
    
    F32TEFE = FT / (
        1.0 + 0.23 * (1.0 - 0.96 * FT) * np.sqrt(ZNUE) / ZEFF**0.5
        + 0.13 * (1.0 - 0.38 * FT) * ZNUE / ZEFF**2
        * (np.sqrt(1.0 + 2 * np.sqrt(ZEFF - 1.0))
        + FT**2 * np.sqrt((0.075 + 0.25 * (ZEFF - 1.0)**2) * ZNUE))
    )
    
    ZL32 = F32TEFE
    
    F34TEF = FT / (
        1.0 + (1.0 - 0.1 * FT) * np.sqrt(ZNUE) 
        + 0.5 * (1.0 - 0.5 * FT) * ZNUE / ZEFF
    )
    
    A0 = - (
        (0.62 + 0.055 * (ZEFF - 1.0)) 
        / (0.53 + 0.17 * (ZEFF - 1.0))
        * (1.0 - FT) 
        / (1.0 - (0.31 - 0.065 * (ZEFF - 1.0)) * FT - 0.25 * FT**2)
    )
    
    ANU = (
        (A0 + 0.7 * ZEFF * FT**0.5 * np.sqrt(ZNUI))
        / (1.0 + 0.18 * np.sqrt(ZNUI)) 
        - 0.002 * ZNUI**2 * FT**6
    ) / (1.0 + 0.004 * ZNUI**2 * FT**6)
    
    ZL34 = ZL31
    
    P = (ZNI * TI + ZNE * TE) * E
    DP = (ZNI * DTI + DNI * TI + ZNE * DTE + DNE * TE) * E
    PE = (ZNE * TE) * E
    
    ZJBT = -P * (
        ZL31 * DNE / ZNE 
        + RPE * (ZL31 + ZL32) * DTE / TE 
        + (1.0 - RPE) * (1.0 + ZL34 / ZL31 * ANU) * ZL31 * DTI / TI
    )
    
    return ZJBT * (mu_0 * R / BT**2)

def calculate_pressure(te, ne): 
    # te in keV, ne in 10^19 m^-3
    # returns pressure in kPa
    return (1.60217 * (ne) * (te*1000) / 0.5)
    # return (2.0*(te*11604.5)*(ne / 10.0)*1.36064852) / 10000.0


def get_max_alpha_value(alpha, psiN): 
    # relevant_region = psiN > 0.9

    # second_derivative = np.gradient(np.gradient(alpha[relevant_region], psiN[relevant_region]), psiN[relevant_region])
    max_index = np.argmax(alpha[:-2])
    return alpha[max_index], max_index



def is_point_left_of_boundary(x, y, contour):
    """
    Determines if a point (x, y) is to the left of a boundary defined by (x_bndry, y_bndry).
    
    Args:
        x, y: Coordinates of the point to test.
        x_bndry, y_bndry: Coordinates of the boundary as arrays.
    
    Returns:
        True if the point is to the left of the boundary, False otherwise.
    """
    # Ensure boundary is closed
    x_bndry, y_bndry = contour[:, 0], contour[:, 1]
    
    # if x_bndry[0] != x_bndry[-1] or y_bndry[0] != y_bndry[-1]:
    #     x_bndry = np.append(x_bndry, x_bndry[0])
    #     y_bndry = np.append(y_bndry, y_bndry[0])
    # Define a cross-product function to test left/right
    winding_number = 0
    
    # Loop over boundary segments
    for i in range(len(x_bndry) - 1):
        x1, y1 = x_bndry[i], y_bndry[i]
        x2, y2 = x_bndry[i + 1], y_bndry[i + 1]
        
        if y1 <= y < y2:  # Upward crossing
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) > 0:
                winding_number += 1
        elif y2 <= y < y1:  # Downward crossing
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) < 0:
                winding_number -= 1
    
    return winding_number != 0

from dataclasses import dataclass 

@dataclass
class SimGrid: 
    psin: np.ndarray
    fcirc: np.ndarray
    q: np.ndarray
    Area: np.ndarray
    Rmaj: np.ndarray
    eps: np.ndarray
    vol: np.ndarray
    volp: np.ndarray
    baseline_current: np.ndarray
    density: np.ndarray 
    temperature: np.ndarray 
    pressure: np.ndarray 
    bootstrap_current: np.ndarray 
    alpha:np.ndarray
    alpha_bndry: np.ndarray 
    jb_bndry   : np.ndarray
    Zeff: float
    BT: float
    psi_bndry: float

    def get_total_current(self) -> np.ndarray: 
        return self.baseline_current + self.bootstrap_current

    def get_alpha_jb_max(self): 
        max_alpha, max_alpha_idx = get_max_alpha_value(self.alpha, self.psin)
        max_jb,    max_jb_idx    = get_max_alpha_value(self.get_total_current(), self.psin)
        return max_alpha, max_jb # self.get_total_current()[max_alpha_idx]
    
    def update_pressure(self): 
        self.pressure = calculate_pressure(self.temperature, self.density)

    def update_bootstrap(self): 
        self.bootstrap_approx = calculate_zjbt(
                                            self.fcirc, 
                                            self.density, self.density, np.gradient(self.density, self.psin), np.gradient(self.density, self.psin),
                                            self.temperature, self.temperature, np.gradient(self.temperature, self.psin),np.gradient(self.temperature, self.psin),
                                            self.q, self.Rmaj, self.eps, self.Zeff, self.BT, 
                                            # COLLMULT=0.1
                                        )

    def update_alpha(self): 
        self.alpha = calculate_alpha_helena(self.psin, self.pressure, self.psi_bndry, self.vol, self.volp, self.eps, self.BT)


def gather_necessary_data_from_equilibrium(eq_fpath: str, bndry_fpath: str, NX: int):
    """ 
    Returns all the necessary data from the equilibrium and boundary files. 

    Bootstrap Current: 
        - Diff between approx and baseline , which is added to bootstrap approx
            - baseline_current
        - relevant arrays for bootstrap approx
            - fcirc, q, Rmaj, eps
        - relevant floats for bootstrap approx
            - Zeff, BT
    Alpha: 
        - relevant arrays for alpha
            - vol, volp, eps
        - relevant floats for alpha
            - psin_bndry, BT 

    PDEs: 
        - relevant arrays for PDEs
            - density, temperature
            - area, rmaj 
    """
    elite_fname = os.path.join(eq_fpath, 'eliteinp')
    f20_fname = os.path.join(eq_fpath, 'fort.20')
    stab_cont_fname = os.path.join(bndry_fpath)
    fort10_fname = os.path.join(eq_fpath, 'fort.10')
    
    helnml = read_namelist(fort10_fname)
    elite_data = read_eliteinput(elite_fname)
    _, vol_f20, volp_f20, j_f20, _, pe_f20, ne_f20, te_f20, _, _, alpha_f20, _, fcirc_f20, jb_f20 = get_data_from_f20(f20_fname)
    stability_contour = np.loadtxt(stab_cont_fname, skiprows=1)




    PsiN_elite     = elite_data['PsiN'][1:-1]
    qprofile_elite = elite_data['q'][1:-1]
    Rmaj_elite = np.empty(PsiN_elite.shape)
    Rmin_elite = np.empty(PsiN_elite.shape)
    Area_elite = np.empty(PsiN_elite.shape)
    for idx, psi in enumerate(PsiN_elite):
        Rmaj_elite[idx] = elite_data['R'][:, idx].max()
        Rmin_elite[idx] = Rmaj_elite[idx] - Rmaj_elite[0]
        Area_elite[idx] = calculate_area(elite_data['R'][:, idx], elite_data['z'][:, idx])


    PsiN_elite     = elite_data['PsiN'][1:-1]

    PsiN_interp = np.linspace(0.7, np.max(PsiN_elite), NX)

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


    """ FROM REAL WORLD OUTPUT SECTION OF FORT:20"""
    RPE = 0.5
    psi_bndry = 0.5988

    BT = helnml['phys']['bvac']
    Zeff = helnml['phys']['zeff']


    if helnml['phys']['idete'] == 5: 
        temperature = mtanh(PsiN_interp, helnml['phys']['ate'], helnml['phys']['bte'], helnml['phys']['cte'], helnml['phys']['ddte'], helnml['phys']['ete'])
        density = mtanh(PsiN_interp, helnml['phys']['ade'], helnml['phys']['bde'], helnml['phys']['cde'], helnml['phys']['dde'], helnml['phys']['ede'])
    else:
        temperature = eped_profile_helena(PsiN_interp, helnml['phys']['ate'], helnml['phys']['bte'], helnml['phys']['cte'], helnml['phys']['ddte'], helnml['phys']['ete'], helnml['phys']['fte'], helnml['phys']['gte'], helnml['phys']['hte'])
        density = eped_profile_helena(PsiN_interp, helnml['phys']['ade'], helnml['phys']['bde'], helnml['phys']['cde'], helnml['phys']['dde'], helnml['phys']['ede'], helnml['phys']['fde'], helnml['phys']['gde'], helnml['phys']['hde'])

    pressure = calculate_pressure(temperature, density)

    alpha_bndry = stability_contour[:, 0]
    jb_bndry = stability_contour[:, 1]

    # TODO: this is some bullshit to get things on the right axis
    PsiN_interp += 0.01
    bool_mask = PsiN_interp < 1.0
    PsiN_interp = PsiN_interp[bool_mask]
    fcirc_interp = fcirc_interp[bool_mask]
    q_interp = q_interp[bool_mask]
    Area_interp = Area_interp[bool_mask]
    Rmaj_interp = Rmaj_interp[bool_mask]
    eps_interp = eps_interp[bool_mask]
    vol_interp = vol_interp[bool_mask]
    volp_interp = volp_interp[bool_mask]
    j_interp = j_interp[bool_mask]
    density = density[bool_mask]
    temperature = temperature[bool_mask]
    pressure = pressure[bool_mask]
    # alpha_interp = alpha_interp[bool_mask]
    # baseline_current = baseline_current[bool_mask]


    alpha_approx = calculate_alpha_helena(PsiN_interp, pressure, psi_bndry, vol_interp, volp_interp, eps_interp, BT)
    bootstrap_approx = calculate_zjbt(
                                            fcirc_interp, 
                                            density, density, np.gradient(density, PsiN_interp), np.gradient(density, PsiN_interp),
                                            temperature, temperature, np.gradient(temperature, PsiN_interp),np.gradient(temperature, PsiN_interp),
                                            q_interp, Rmaj_interp, eps_interp, Zeff, BT, 
                                            # COLLMULT=0.1
                                        )
    baseline_current = j_interp - bootstrap_approx
    
    return SimGrid(PsiN_interp, fcirc_interp, q_interp, Area_interp, Rmaj_interp, eps_interp, vol_interp, volp_interp, baseline_current, density, temperature, pressure, bootstrap_approx, alpha_approx, alpha_bndry, jb_bndry, Zeff, BT, psi_bndry)
