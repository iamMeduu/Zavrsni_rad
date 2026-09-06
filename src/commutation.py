import pandas as pd

def calculate_commutative_numbers(csv_path="data/mortality_2010_2012_monthly.csv", i=0.0025):
    df = pd.read_csv(csv_path)
    x, lx = df["age_months"], df["lx"]

    im = (1 + i)**(1/12) - 1
    vm = 1 / (1 + im)

    Dx = (vm**x) * lx
    Dx.index = x

    dx = lx - lx.shift(-1)
    dx.iloc[-1] = lx.iloc[-1]

    Cx = (vm**(x+1)) * dx
    Cx.index = x

    Mx = Cx[::-1].cumsum()[::-1]
    Nx = Dx[::-1].cumsum()[::-1]

    return Dx, Nx, Mx