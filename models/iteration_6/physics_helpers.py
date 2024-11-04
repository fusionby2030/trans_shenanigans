from load_data import MachineParameters

def calculate_bohmdiffusion(T0, mps: MachineParameters):
    # D_BOHM = (1.0/16.0) * (k_b *T) / (e * B)
    kev2j    = 1.60218e-17    # J
    e        = 1.60218e-19    # C
    D_BOHM = (1.0/16.0) * (kev2j*T0) / (e * mps.BT)
    return D_BOHM
