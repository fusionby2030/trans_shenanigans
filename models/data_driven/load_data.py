import numpy as np 
from dataclasses import dataclass
import pickle 
import os 
import pandas as pd 

RAW_JET_DIR = "/scratch/project_2005083/aarojarvinen/ped_ssm/shot_dirs/JET/"
LOCAL_STORAGE_LOC = "/scratch/project_2007159/profile_uncertainty/raw_jet_arrays"



@dataclass
class SignalStruct:
    """Simple struct for holding signal data and time"""

    data: np.ndarray
    time: np.ndarray

    def reduce_time_window(self, t1: float, t2: float, buffer: float = 0.1):
        window = np.logical_and(self.time >= t1 - buffer, self.time <= t2 + buffer)
        self.data = self.data[window]
        self.time = self.time[window]


@dataclass
class KineticProfileStruct:
    """Struct for holding kinetic profile information"""

    ne: np.ndarray
    te: np.ndarray
    ne_unc: np.ndarray
    te_unc: np.ndarray
    hrts_times: np.ndarray
    hrts_psi: np.ndarray
    ne_fit_params: dict[str, float] = None
    te_fit_params: dict[str, float] = None
    final_ne_fit_params: dict[str, float] = None
    final_te_fit_params: dict[str, float] = None
    fit_type: str = None

    def filter_upto_psi_1(self):
        ne, te, ne_unc, te_unc, hrts_psi = [], [], [], [], []
        for t_idx, _ in enumerate(self.hrts_times):
            for n_idx, psi in enumerate(self.hrts_psi[t_idx]):
                if 0.0 < psi < 1.0:
                    ne.append(self.ne[t_idx, n_idx])
                    te.append(self.te[t_idx, n_idx])
                    ne_unc.append(self.ne_unc[t_idx, n_idx])
                    te_unc.append(self.te_unc[t_idx, n_idx])
                    hrts_psi.append(self.hrts_psi[t_idx, n_idx])

        self.ne = np.array(ne)
        self.te = np.array(te)
        self.ne_unc = np.array(ne_unc)
        self.te_unc = np.array(te_unc)
        self.hrts_psi = np.array(hrts_psi)


@dataclass
class PulseStruct:
    """holds the necessary information for doing the worfklow"""

    shot_num: int
    t1: float = None
    t2: float = None
    profiles: KineticProfileStruct = None
    tbeo: SignalStruct = None
    elm_times: np.ndarray = None
    relevant_hrts_measurements: np.ndarray = None
    elm_fractions: np.ndarray = None
    failure: bool = False
    reason: str = ""
    base_path: str = ""

    def reduce_structs_to_time_window(self):
        buffer = 0.1
        self.tbeo.reduce_time_window(self.t1, self.t2, buffer)

    def load_extra_data(self, extra_path: str, shot_num: int, dda: str, datatype: str) -> SignalStruct:
        time_fname = f"{shot_num}_{dda}_{datatype}_time.npy"
        data_fname = f"{shot_num}_{dda}_{datatype}_data.npy"
        time_fname = os.path.join(extra_path, time_fname)
        data_fname = os.path.join(extra_path, data_fname)
        data, time = np.load(data_fname), np.load(time_fname)
        return SignalStruct(data=data, time=time)

    def load_profile_data(self, data_loc: str, shot_num: int) -> KineticProfileStruct:
        fname = os.path.join(data_loc, str(shot_num))
        with open(fname, "rb") as pulse_file:
            pulse_dict = pickle.load(pulse_file)
        hrts_times, ne, ne_unc, te_unc, te, hrts_psi = [
            pulse_dict["profiles"][name] for name in ["time", "ne", "ne_unc", "Te_unc", "Te", "radius"]
        ]
        return KineticProfileStruct(ne, te, ne_unc, te_unc, hrts_times, hrts_psi)

    def return_profiles_from_a_bool_array(self, relevant_bool_arr: np.ndarray) -> KineticProfileStruct:
        reduced_ne = self.profiles.ne[relevant_bool_arr]
        reduced_ne_unc = self.profiles.ne_unc[relevant_bool_arr]
        reduced_te_unc = self.profiles.te_unc[relevant_bool_arr]
        reduced_te = self.profiles.te[relevant_bool_arr]
        reduced_hrts_times = self.profiles.hrts_times[relevant_bool_arr]
        reduced_hrts_psi = self.profiles.hrts_psi[relevant_bool_arr]

        return KineticProfileStruct(
            reduced_ne, reduced_te, reduced_ne_unc, reduced_te_unc, reduced_hrts_times, reduced_hrts_psi
        )

    def return_fit_relevant_arrays(self) -> KineticProfileStruct:
        relevant_bool_arr = self.relevant_hrts_measurements
        return self.return_profiles_from_a_bool_array(relevant_bool_arr)

    def return_profiles_in_time_windows(
        self,
    ) -> KineticProfileStruct:
        relevant_bool_array = np.logical_and(self.profiles.hrts_times > self.t1, self.profiles.hrts_times < self.t2)
        return self.return_profiles_from_a_bool_array(relevant_bool_array)

    def __post_init__(self):
        """Load data"""
        self.profiles = self.load_profile_data(RAW_JET_DIR, self.shot_num)
        self.tbeo = self.load_extra_data(LOCAL_STORAGE_LOC, self.shot_num, "edg8", "tbeo")

@dataclass
class FitParams:
    """
    offset, pedestal height, slope, position, width
    """

    h0: float
    h1: float
    s: float
    p: float
    w: float
    steady_state_arr: np.ndarray = None

@dataclass 
class MachineParameters: 
    BT: float   # toroidal field strength [Tesla] 
    Rmaj: float # major radius [meters]
    Rmin: float # minor radius [meters]
    Ip: float   # plasma current [Amps]
    Zeff: float # effective charge state of impurities

    def __str__(self): 
        return f"BT: {self.BT}, Rmaj: {self.Rmaj}, Rmin: {self.Rmin}, Ip: {self.Ip}, Zeff: {self.Zeff}"


def load_single_pulse_struct(_dir: str) -> PulseStruct:
    fname = os.path.join(_dir, "pulse_struct.pickle")
    with open(fname, "rb") as file:
        pulse_struct = pickle.load(file)

    return pulse_struct


def load_jetpdb(_dir: str) -> pd.DataFrame: 
    fpath = os.path.join(_dir, 'jet-all-full.csv')
    return pd.read_csv(fpath)

def get_fit_params_from_pdbshot(_dir: str, shot_num: int) -> tuple[FitParams, FitParams, FitParams, MachineParameters]:
    df = load_jetpdb(_dir)

    shotrow = df[df["shot"] == shot_num]
    if len(shotrow) > 1: 
        print("MORE THAN ONE INSTANCE OF GIVEN SHOT NUMBER")
        print(shotrow)
    shotrow = shotrow.iloc[0]
    
    return FitParams(
            h1=shotrow["Tepedheight(keV)"],
            h0=0.0,
            s=shotrow["Teinnerslope"],
            p=shotrow["Teposition(psiN)"],
            w=shotrow["Tepedestalwidth(psiN%)"],
        ), FitParams(
            h1=shotrow["nepedheight10^19(m^-3)"],
            h0=0.0,
            s=shotrow["neinnerslope"],
            p=shotrow["neposition(psiN)"],
            w=shotrow["Nepedestalwidth(psiN%)"],
        ), FitParams(
            h1=shotrow["pepedheight(kPa)"],
            h0=0.0,
            s=shotrow["peinnerslope"],
            p=shotrow["peposition(psiN)"],
            w=shotrow["pepedestalwidth(psiN%)"],
        ), MachineParameters(
            BT=shotrow["B(T)"], 
            Rmaj=shotrow["R(m)"],
            Rmin=shotrow["a(m)"], 
            Ip=shotrow["Ip(MA)"], 
            Zeff=shotrow["Zeff"],
        )
