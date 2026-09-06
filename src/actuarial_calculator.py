import pandas as pd
from commutation import calculate_commutative_numbers
ALPHA = 0.02
GAMMA = 0.04

Dx, Nx, Mx = calculate_commutative_numbers()

OIS_ammount = [1000, 2000, 3000, 4000, 5000]
OID_ammount= [1000, 2000, 3000, 4000, 5000]


def check_values(x, n, t, OIS, OID):
    if not (216 <= x <= 780):
        raise ValueError(f"Age months value (x): {x} isn't supported. Must satisfy 216 <= x <= 780")
    if not (24 <= n <= 420):
        raise ValueError(f"Duration months value (n): {n} isn't supported. Must satisfy 24 <= n <= 420")
    if x + n > 900:
        raise ValueError(f"x+n = {x + n} isn't supported. Must satisfy x+n <= 900")
    if not (0 <= t <= n):
        raise ValueError(f"Elapsed months value (t): {t} isn't supported. Must satisfy 0 <= t <= n (n={n} months)")
    if OIS not in OIS_ammount:
        raise ValueError(f"OIS_ammout {OIS} isn't supported, must be one of the following: {OIS_ammount}")
    if OID not in OID_ammount:
        raise ValueError(f"OID_ammount {OID} isn't supported, must be one of the following: {OID_ammount}")
    return True

def calculate_k(OIS_ammount, OID_ammount):
    return OIS_ammount/OID_ammount

def calculate_delta(k):
    if k == 1/5 or k==1/4 or k==1/3 or k==1/2 or k == 1 :
        return 0.0
    elif k == 2:
        return 0.25
    elif k == 3:
        return 0.50
    elif k== 4:
        return 0.75
    elif k == 5:
        return 1.00
    else:
        raise ValueError(f"Unsupported value of k={k}. K value needs to be one of the following: 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5")

def get_shifted_commutation_values(x, n, t):
    return {
        "Dx_t": Dx[x + t],
        "Nx_n": Nx[x + n],
        "Nx_t": Nx[x + t],
        "Mx_n": Mx[x + n],
        "Mx_t": Mx[x + t],
        "Dx_n": Dx[x + n],
    }

def calculate_premium_parts(x, n, OIS, OID):
    check_values(x, n, 0, OIS, OID)
    k = calculate_k(OIS, OID)
    delta = calculate_delta(k)
    shifted = get_shifted_commutation_values(x, n, 0)

    Ps = OIS * (1 + delta) * (Mx[x] - shifted["Mx_n"]) / (Nx[x] - shifted["Nx_n"])
    Pd = OID * shifted["Dx_n"] / (Nx[x] - shifted["Nx_n"])
    return Ps, Pd

def net_premium(x, n, OIS, OID):
    Ps, Pd = calculate_premium_parts(x, n, OIS, OID)
    return Ps+Pd

def gross_premium(x, n, OIS, OID, alpha=ALPHA, gamma=GAMMA):
    net_ps, net_pd = calculate_premium_parts(x, n, OIS, OID)
    bp_s = net_ps / (1 - alpha - gamma)
    bp_d = net_pd / (1 - alpha - gamma)
    return bp_s+bp_d

def mathematical_reserve(x, n, t, OIS, OID):
    check_values(x, n, t, OIS, OID)
    k = calculate_k(OIS, OID)
    delta = calculate_delta(k)
    shifted = get_shifted_commutation_values(x, n, t)
    P_xn = net_premium(x, n, OIS, OID)
    death_part = OIS * (1 + delta) * (shifted["Mx_t"] - shifted["Mx_n"]) / shifted["Dx_t"]
    survival_part = OID * shifted["Dx_n"] / shifted["Dx_t"]
    premium_part = P_xn * (shifted["Nx_t"] - shifted["Nx_n"]) / shifted["Dx_t"]
    reserve = death_part + survival_part - premium_part
    return reserve

def surrender_value(age_months, duration_months, elapsed_months, OIS, OID):
    check_values(age_months, duration_months, elapsed_months, OIS, OID)
    reserve = mathematical_reserve(age_months, duration_months, elapsed_months, OIS, OID)
    if reserve < 0:
        print(f"Warning: negative reserve ({reserve}) for x={age_months}, n={duration_months}, t={elapsed_months}")
    value = min(OIS, 0.97 * reserve)
    if value < 0:
        print(f"Warning: negative surrender value ({value}) for x={age_months}, n={duration_months}, t={elapsed_months}")

    return value


