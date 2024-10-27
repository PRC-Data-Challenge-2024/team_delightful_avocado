import os

import pandas as pd
import numpy as np


p_0 = 101325  # Pa
T_0 = 288.15  # K
L = 0.0065  # K/m
g = 9.80665  # m/s^2
M = 0.0289652  # kg/mol
R = 8.31447  # J/(mol K)


def pressure(height):
    # convert height to meters
    height = height * 0.3048
    exponent = g * M / (R * L)
    return p_0 * (1 - L * height / T_0) ** exponent


def speed_of_sound(temp):
    return np.sqrt(1.4 * R * temp / M) * 1.94384  # kt


def qfe(qnh, alt, temp):
    # convert altitude to meters
    alt = alt * 0.3048
    # convert temperature to Kelvin
    temp = temp + 273.15
    return qnh * np.exp(-g * M * alt / (R * temp))


def cas(tas, density, temp):
    a_0 = 661.4786  # kt
    p_0 = 101325  # Pa
    kappa = 1.4
    q = density * (
        (((kappa - 1) / 2) * (tas / speed_of_sound(temp)) ** 2 + 1)
        ** (kappa / (kappa - 1))
        - 1
    )
    return a_0 * np.sqrt(2 / (kappa - 1) * ((1 + q / p_0) ** ((kappa - 1) / kappa) - 1))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    return (
        2
        * R
        * np.arcsin(
            np.sqrt(
                np.sin(d_phi / 2) ** 2
                + np.cos(np.radians(lat1))
                * np.cos(np.radians(lat2))
                * np.sin(d_lambda / 2) ** 2
            )
        )
    )


current_dir = os.path.dirname(os.path.abspath(__file__))

masses = pd.read_csv(
    os.path.join(current_dir, "data/masses.csv"), header=0, delimiter=","
)

airport_data = pd.read_csv(
    os.path.join(current_dir, "data/airport-data.csv"), header=0, delimiter=","
)

flights_per_day = pd.read_csv(
    os.path.join(current_dir, "data/total_flights_per_day.csv"), header=0, delimiter=","
)

mean_performance = pd.read_csv(
    os.path.join(current_dir, "data/mean_data.csv"), header=0, delimiter=","
)


def get_data(path: str) -> pd.DataFrame:
    X = pd.read_csv(f"data/{path}.csv")
    parquet_data = pd.read_csv(f"data/{path}_parquet.csv", header=0, delimiter=",")
    # Drop duplicates by flight_id
    parquet_data = parquet_data.drop_duplicates(subset="flight_id")

    X["date"] = pd.to_datetime(X["date"])
    X["airport_pair"] = X["adep"] + "_" + X["ades"]
    X["elevation_adep"] = X["adep"].map(airport_data.set_index("airport")["elevation"])
    X["elevation_ades"] = X["ades"].map(airport_data.set_index("airport")["elevation"])
    X["time_zone_adep"] = X["adep"].map(airport_data.set_index("airport")["timezone"])
    X["time_zone_ades"] = X["ades"].map(airport_data.set_index("airport")["timezone"])
    X["lat_adep"] = X["adep"].map(airport_data.set_index("airport")["latitude"])
    X["lon_adep"] = X["adep"].map(airport_data.set_index("airport")["longitude"])
    X["lat_ades"] = X["ades"].map(airport_data.set_index("airport")["latitude"])
    X["lon_ades"] = X["ades"].map(airport_data.set_index("airport")["longitude"])
    X["haversine"] = haversine(
        X["lat_adep"], X["lon_adep"], X["lat_ades"], X["lon_ades"]
    )

    X["oew"] = X["aircraft_type"].map(masses.set_index("aircraft_type")["oew"])
    X["mtom"] = X["aircraft_type"].map(masses.set_index("aircraft_type")["mtom"])

    X["avg_tas"] = X["flight_id"].map(parquet_data.set_index("flight_id")["avg_tas"])
    X["climb_rate_mean"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["climb_rate_mean"]
    )
    X["avg_wind_u"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["avg_wind_u"]
    )
    X["avg_wind_v"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["avg_wind_v"]
    )
    X["climb_rate_max"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["climb_rate_max"]
    )
    X["descent_rate_mean"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["descent_rate_mean"]
    )
    X["descent_rate_min"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["descent_rate_min"]
    )
    X["max_crz_alt"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["max_crz_alt"]
    )
    X["mean_crz_alt"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["mean_crz_alt"]
    )
    X["crz_tas"] = X["flight_id"].map(parquet_data.set_index("flight_id")["crz_tas"])
    X["crz_wind_u"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["crz_wind_u"]
    )
    X["crz_wind_v"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["crz_wind_v"]
    )
    X["crz_wind_tot"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["crz_wind_tot"]
    )
    X["crz_gnd_speed"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["crz_gnd_speed"]
    )
    X["adsb_inflight_time"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["adsb_inflight_time"]
    )
    X["landing_tas"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["landing_tas"]
    )
    X["landing_temp"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["landing_temp"]
    )
    X["landing_pressure"] = X["elevation_ades"].apply(pressure)
    X["landing_density"] = (
        0.0289652  # kg/mol
        * X["landing_pressure"]
        / X["landing_temp"]
        / 8.31446  # Nm/(mol K)
    )
    X["landing_cas"] = cas(X["landing_tas"], X["landing_pressure"], X["landing_temp"])
    X["landing_cas_squared"] = X["landing_tas"] ** 2
    X["landing_mass"] = X["landing_density"] * X["landing_cas_squared"]

    X["dep_tas"] = X["flight_id"].map(parquet_data.set_index("flight_id")["dep_tas"])
    X["dep_temp"] = X["flight_id"].map(parquet_data.set_index("flight_id")["dep_temp"])
    X["dep_pressure"] = X["elevation_adep"].apply(pressure)
    X["dep_density"] = (
        0.0289652 * X["dep_pressure"] / X["dep_temp"] / 8.31446  # kg/mol  # Nm/(mol K)
    )
    X["dep_cas"] = cas(X["dep_tas"], X["dep_pressure"], X["dep_temp"])
    X["dep_cas_squared"] = X["dep_tas"] ** 2
    X["dep_mass"] = X["dep_density"] * X["dep_cas_squared"]

    X["init_climb_rate"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["init_climb_rate"]
    )
    X["time_to_10k"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["time_to_10k"]
    )
    X["dep_sc"] = X["flight_id"].map(parquet_data.set_index("flight_id")["dep_sc"])
    X["landing_sc"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["landing_sc"]
    )
    X["climb_efficiency"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["climb_efficiency"]
    )
    X["fuel_efficiency_proxy"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["fuel_efficiency_proxy"]
    )
    X["climb_in_phases"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["climb_in_phases"]
    )
    X["cruise_in_phases"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["cruise_in_phases"]
    )
    X["descent_in_phases"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["descent_in_phases"]
    )
    X["ground_in_phases"] = X["flight_id"].map(
        parquet_data.set_index("flight_id")["ground_in_phases"]
    )

    X["flights_per_day"] = X["date"].dt.date.map(
        flights_per_day.set_index("day")["flights"]
    )

    # Use the mean values for the aircraft type to scale the data
    new_columns = {}
    for col in list(mean_performance):
        if col == "aircraft_type":
            continue
        # Map and scale values as before
        acft_col = X["aircraft_type"].map(
            mean_performance.set_index("aircraft_type")[col]
        )
        new_columns[f"acft-{col}"] = acft_col
        new_columns[f"scaled-{col}"] = X[col] / acft_col

    # Add all new columns to X at once to avoid fragmentation
    X = pd.concat([X, pd.DataFrame(new_columns)], axis=1)

    return X
