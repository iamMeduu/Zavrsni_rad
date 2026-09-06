import pandas as pd

data= pd.read_csv("data/mortality_2010_2012.csv")
lx=data["lx"]

rows = []
for a in range(100):
    for r in range(12):
        l_val = (1 - r/12) * lx[a] + (r/12) * lx[a+1]
        rows.append({"age_months": a*12+r, "age_years": round(a+r/12, 4), "lx": round(l_val)})

rows.append({"age_months": 1200, "age_years": 100.0000, "lx": lx[100]})

data_age_months = pd.DataFrame(rows)
data_age_months["lx"] = data_age_months["lx"].astype(int)
data_age_months = data_age_months.sort_values(["age_years", "age_months"]).reset_index(drop=True)
data_age_months.to_csv("data/mortality_2010_2012_monthly.csv", index=False)