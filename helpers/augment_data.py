import pandas as pd
import numpy as np
from geopy.distance import geodesic

df = pd.read_csv("../data/chicago_taxi_trips.csv")

df = df.sort_values(["taxi_id", "timestamp"])

battery_capacity = 60  # kWh
consumption_rate = 0.20  # kWh/km

def simulate_ev_data(group):
    soc = np.random.uniform(0.6, 1.0)
    soc_list = []
    charge_events = []
    for i in range(1, len(group)):
        prev = group.iloc[i-1]
        curr = group.iloc[i]
        dist_km = geodesic((prev['lat'], prev['lon']), (curr['lat'], curr['lon'])).km
        energy_used = dist_km * consumption_rate
        soc -= energy_used / battery_capacity
        if soc <= 0.2:
            charge_time_hr = (1 - soc) * battery_capacity / 11.0
            charge_events.append({
                "vehicle_id": prev["taxi_id"],
                "start_time": curr["timestamp"],
                "lat": curr["lat"],
                "lon": curr["lon"],
                "charge_time_hr": charge_time_hr
            })
            soc = 1.0
        soc_list.append(soc)
    group = group.iloc[1:].copy()
    group["soc"] = soc_list
    return group, pd.DataFrame(charge_events)

all_ev_data = []
all_charges = []
for vid, g in df.groupby("taxi_id"):
    d, c = simulate_ev_data(g)
    all_ev_data.append(d)
    all_charges.append(c)

df_ev = pd.concat(all_ev_data)
charges = pd.concat(all_charges)

df_ev.to_csv("../data/ev_mobility_simulated.csv", index=False)
charges.to_csv("../data/ev_charging_events.csv", index=False)

