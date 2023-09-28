from pymgrid.modules import (
    BatteryModule,
    LoadModule,
    RenewableModule,
    GridModule,
    GensetModule)

import pandas as pd
import numpy as np

battery = BatteryModule(
    min_capacity=15,
    max_capacity=285,
    max_charge=2.5,
    max_discharge=2.5,
    efficiency=0.99,
    battery_cost_cycle=0.95,
    init_soc=0.0
)
print(battery)

gas_turbine_generator = GensetModule(
    running_min_production=0,
    running_max_production=600,
    genset_cost=0.55
)

print(gas_turbine_generator)

data = pd.read_csv("data/EnergyGenerationRenewable.csv")
time_solar = data["Solar Generation"].values
time_wind = data["Wind Generation"].values

solar_pv = RenewableModule(
    time_series=time_solar
)

print(solar_pv)

wind_turbine = RenewableModule(
    time_series=time_wind
)

print(wind_turbine)

buy_price = pd.read_csv("data/rate_consumption_charge.csv")["Grid Elecricity Price锛?/kWh锛?"].values * 10000
sell_price = np.ones(len(buy_price)) * 0.2 * 10000
co2 = np.zeros(len(buy_price))

time_grid = np.concatenate([buy_price[:, None], sell_price[:, None], co2[:, None]], axis=1)

grid = GridModule(
    time_series=time_grid,
    max_export=10000,
    max_import=10000
)

time_load = pd.read_csv("data/Load25Households.csv")["load"].values

load = LoadModule(
    time_series=time_load
)

modules = [
    battery,
    gas_turbine_generator,
    ("solar_pv", solar_pv),
    ("wind_turbine", wind_turbine),
    grid,
    load
]