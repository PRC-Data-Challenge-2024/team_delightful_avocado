import os
from concurrent.futures import as_completed, ProcessPoolExecutor

import pandas as pd
from datetime import datetime

from traffic.core import Traffic, Flight

target_set = "data/challenge_set.csv"
output_path = "data/challenge_set/"
parquet_path = "data/parquet/"


def load_flights(date: datetime.date):
    """
    Load the flight(-ids) for a given date from the given set
    :param date:
    :return: dataframe
    """
    df = pd.read_csv(target_set)
    df = df[df["actual_offblock_time"].str.startswith(date.strftime("%Y-%m-%d"))]
    return df


def process_one_flight(flight: Flight):
    """
    Processes a single flight
    :param flight:
    :return:
    """
    flight_id = flight.flight_id
    df = flight.data.copy()
    df.rename(
        columns={
            "u_component_of_wind": "wind_u",
            "v_component_of_wind": "wind_v",
        },
        inplace=True,
    )
    df.drop_duplicates(inplace=True)
    df.sort_values(by="timestamp", inplace=True)

    # Create a Flight object from the cleaned dataframe
    flt = Flight(df)

    # Rename columns using pandas rename function
    try:
        flt = flt.compute_TAS()
    except Exception as e:
        print("Error computing TAS for flight " + str(flight_id), e)

    # Compute the phases of the flight
    flt = flt.phases()
    unique_phases = flt.data["phase"].unique()

    # Remove NA and GROUND phases
    flt.data = flt.data[flt.data["phase"] != "NA"]
    flt.data = flt.data[flt.data["phase"] != "GROUND"]

    # Resample the data to 10s intervals
    flt = flt.resample("10s")

    # Compute the time to climb 10,000 feet
    altitude_gain = (flt.data["vertical_rate"] / 6).cumsum()
    reached_10k = altitude_gain >= 10000
    if reached_10k.any():
        time_to_10k = (
            flt.data[reached_10k]["timestamp"].iloc[0] - flt.data["timestamp"].iloc[0]
        ).total_seconds() / 60  # time in minutes
    else:
        # If the flight didn't climb 10,000 feet, handle accordingly
        time_to_10k = pd.NA

    # Filter the data based on the phases
    flt_phase_climb = flt.data["phase"] == "CLIMB"
    flt_phase_descent = flt.data["phase"] == "DESCENT"
    flt_phase_cruise = flt.data["phase"] == "CRUISE"

    result = {
        "flight_id": int(flt.flight_id),
        "avg_tas": flt.data["TAS"].mean(),
        "avg_wind_u": flt.data["wind_u"].mean(),
        "avg_wind_v": flt.data["wind_v"].mean(),
        "climb_rate_mean": flt.data[flt_phase_climb]["vertical_rate"].mean(),
        "climb_rate_max": flt.data[flt_phase_climb]["vertical_rate"].max(),
        "descent_rate_mean": flt.data[flt_phase_descent]["vertical_rate"].mean(),
        "descent_rate_min": flt.data[flt_phase_descent]["vertical_rate"].min(),
        "max_crz_alt": flt.data["altitude"].max() / 100,
        "adsb_inflight_time": (
            flt.data["timestamp"].max() - flt.data["timestamp"].min()
        ).total_seconds()
        / 3600,
        "landing_tas": flt.data["TAS"].iloc[-1],
        "landing_temp": flt.data["temperature"].iloc[-1],
        "dep_temp": flt.data["temperature"].iloc[0],
        "dep_tas": flt.data["TAS"].iloc[flt.data["TAS"].first_valid_index()],
        "init_climb_rate": flt.data["vertical_rate"]
        .iloc[
            flt.data["vertical_rate"]
            .first_valid_index() : flt.data["vertical_rate"]
            .first_valid_index()
            + 3
        ]
        .mean(),
        "time_to_10k": time_to_10k,
        "dep_sc": flt.data["specific_humidity"].iloc[0],
        "landing_sc": flt.data["specific_humidity"].iloc[-1],
        "climb_efficiency": flt.data[flt.data["phase"] == "CLIMB"][
            "vertical_rate"
        ].mean()
        / flt.data[flt.data["phase"] == "CLIMB"]["groundspeed"].mean(),
        "climb_in_phases": int("CLIMB" in unique_phases),
        "cruise_in_phases": int("CRUISE" in unique_phases),
        "descent_in_phases": int("DESCENT" in unique_phases),
        "ground_in_phases": int("GROUND" in unique_phases),
    }

    # Iff we have a cruise phase, compute the cruise metrics

    cruise_part = flt.data[flt_phase_cruise]

    if cruise_part.empty:
        cruise_part = flt.data[flt.data["phase"] == "LEVEL"]

    if cruise_part.empty:
        print("No cruise or level phase found for flight " + str(flight_id))
    else:
        cruise_res = {
            "mean_crz_alt": cruise_part["altitude"].mean() / 100,
            "descent_rate_min": flt.data[flt.data["phase"] == "DESCENT"][
                "vertical_rate"
            ].min(),
            "crz_tas": cruise_part["TAS"].mean(),
            "crz_wind_u": cruise_part["wind_u"].mean(),
            "crz_wind_v": cruise_part["wind_v"].mean(),
            "crz_wind_tot": (
                cruise_part["wind_u"] ** 2 + cruise_part["wind_v"] ** 2
            ).mean()
            ** 0.5,
            "crz_gnd_speed": cruise_part["groundspeed"].mean(),
            "fuel_efficiency_proxy": cruise_part["TAS"].mean()
            - cruise_part["groundspeed"].mean(),
        }
        result.update(cruise_res)
    return result


def process_daily_file(date: datetime.date):
    print("Starting with date " + str(date))
    parquet_date = date
    date_str = parquet_date.strftime("%Y-%m-%d")

    flights = load_flights(parquet_date)

    # Read the daily flight data
    parquet = Traffic.from_file(parquet_path + date_str + ".parquet")

    # Initialize computed dataframe
    computed = pd.DataFrame()

    # Process each flight
    for index, row in flights.iterrows():
        actual_offblock_time = row["actual_offblock_time"]
        arrival_time = row["arrival_time"]
        flight_id = row["flight_id"]

        # Filter records based on flight IDs
        flight_array = parquet.query(query_str="flight_id == " + str(flight_id) + "")

        if flight_array is None:
            print("Flight " + str(flight_id) + " not found in " + date_str)
            res = {"flight_id": flight_id}
            computed = pd.concat([computed, pd.DataFrame([res])]).reset_index(drop=True)
            continue

        flight = flight_array[0]

        res = process_one_flight(flight)
        computed = pd.concat([computed, pd.DataFrame([res])]).reset_index(drop=True)

    return computed


def process_all_files(max_workers=None):
    # Generate a list of dates for the year 2022
    dates = pd.date_range(start="2022-01-01 00:00Z", end="2022-12-31 00:00Z").tolist()

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process the files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each date as a separate task
        future_to_date = {
            executor.submit(process_daily_file, date): date for date in dates
        }

        # Collect results as they complete
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                result = future.result()
                date_str = date.strftime("%Y-%m-%d")
                result.to_csv(
                    output_path + date_str + ".csv", mode="w", header=True, index=False
                )
                print(f"Completed processing for {date}")
            except Exception as exc:
                print(f"Processing for {date} generated an exception: {exc}")


if __name__ == "__main__":
    process_all_files()
