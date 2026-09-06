import random
import pandas as pd
from actuarial_calculator import surrender_value, calculate_k, calculate_delta

ALLOWED_PAIRS = [(1000,5000),(1000,4000),(1000,3000),(1000,2000),(1000,1000),
                 (2000,1000),(3000,1000),(4000,1000),(5000,1000)]

def generate_random_policy():
    while True:
        x = random.randint(216, 780)
        n = random.randint(24, 420)
        if x + n <= 900:
            return x, n

N_POLICIES = 10000
rows = []

for policy_id in range(N_POLICIES):
    x, n = generate_random_policy()
    OIS, OID = random.choice(ALLOWED_PAIRS)
    k = calculate_k(OIS, OID)
    delta = calculate_delta(k)

    t_values = sorted(set([int(0.10*n), int(0.25*n), int(0.50*n), int(0.75*n), n]))

    for t in t_values:
        sv = surrender_value(x, n, t, OIS, OID)
        rows.append({
            "policy_id": policy_id, "x": x, "n": n, "t": t,
            "OIS": float(OIS), "OID": float(OID), "k": k, "delta": delta,
            "surrender_value": sv,
        })

df = pd.DataFrame(rows)
df.to_csv("data/ml_scenarios.csv", index=False)
print(f"Generated {len(df)} rows for {N_POLICIES} policies")