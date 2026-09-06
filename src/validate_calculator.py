import pandas as pd
from actuarial_calculator import net_premium, gross_premium

val = pd.read_csv("data/validation_premiums.csv")

val["net_python"] = val.apply(
    lambda r: round(net_premium(r.age_years*12, r.duration_years*12, r.OIS, r.OID), 2), axis=1
)
val["gross_python"] = val.apply(
    lambda r: round(gross_premium(r.age_years*12, r.duration_years*12, r.OIS, r.OID), 2), axis=1
)

val["abs_error_net"] = (val["net_doc"] - val["net_python"]).abs()
val["abs_error_gross"] = (val["gross_doc"] - val["gross_python"]).abs()

print(val[["age_years","duration_years","OIS","OID","net_doc","net_python","abs_error_net",
           "gross_doc","gross_python","abs_error_gross"]].to_string(index=False))
print("Average error (net):", val["abs_error_net"].mean())
print("Average error (gross):", val["abs_error_gross"].mean())