\usepackage{ulem}## Data preparation
The data preparation takes place in the `data_augmentation.py` and `create_Xy.py`files.
There, we define the transformation pipeline and calculate features from the sources below.
The provided, extracted and external downloaded data can be found in the `data/` folder.
We use the following data sources for training: 
- [`data/challenge_set.csv`](data/challenge_set.csv): the main data source provided by the challenge, containing the flight ids
- `data/challenge_set_parquet.csv` [^1]: the extracted data from parquet files corresponding to the flight ids in `challenge_set.csv`
- [`data/airport-data.csv`](data/airport-data.csv): the data from OurAirports, containing information about the aerodromes
- [`data/masses.csv`](data/masses.csv): weights specific to the aircraft types


### Features from the csv file `challenge_set.csv`
We use the following features from the csv file:
- `actual_offblock_time`: extracting the following features. Importantly, we convert all times to local time of the departure aerodrome
    - `month`
    - `day`
    - `hour`
    - `minute`
    - `day_of_week`: encoded day of the week, i.e. Monday=0, Sunday=6
    - `day_of_year`: day of the year
    - `is_month_start`: boolean if the day is the first day of the month
    - `is_month_end`: boolean if the day is the last day of the month
    - `is_weekend`: boolean if the day is a weekend
    - `quarter`: quarter of the year
    - `iata_schedule_flag`: indicating whether flight is in summer or winter schedule
    - `is_holiday`: boolean if the day is a holiday. We only use a very basic set of holidays including Christmas and New Year.
- `arrival_time`: we use the same features as above, again converting to local time of the arrival aerodrome
- `adep`: departure aerodrome, OneHotEncoded with a minimum frequency of 50
- `ades`: arrival aerodrome, OneHotEncoded with a minimum frequency of 50
- `airport_pair`: combination of departure and arrival aerodrome, OneHotEncoded with a minimum frequency of 450
- `aircraft_type`: OneHotEncoded
- `airline`: OneHotEncoded
- `country_code_adep`: OneHotEncoded
- `country_code_ades`: OneHotEncoded
- `wtc`: OrdinalEncoded
- `flight_duration`
- `taxiout_time`
- `flown_distance`

Additionally, we calculate `flights_per_day` as the total number of flights on a given day over the whole challenge set.
This allows us to get a sense of how busy the day is and how many flights are on a given day.


### Features from the parquet files `challenge_set_parquet.csv`
We use the following features from the parquet files:

Of the whole flight envelope (climb, cruise, descent), we extract the following features:
- `avg_tas`: average true airspeed
- `avg_wind_u`: average wind component u
- `avg_wind_v`: average wind component v
- `max_crz_alt`: max reached altitude
- `adsb_inflight_time`: time (in sec) between first and last data entry
- `ground_in_phases`, `climb_in_phases`, `cruise_in_phases`, `descent_in_phases`: boolean flags whether the phase is present in the flight

From the flight phases, we extract the following features:
- `climb_rate_mean`: mean climb rate in climb phase of flight
- `climb_rate_max`: max climb rate in climb phase of flight
- `climb_efficiency`: mean climb rate divided by mean ground speed over the whole climb phase
- `mean_crz_alt`: mean altitude in cruise phase of flight
- `crz_tas`: mean true airspeed in cruise phase of flight
- `crz_wind_u`: mean wind component u in cruise phase of flight
- `crz_wind_v`: mean wind component v in cruise phase of flight
- `crz_wind_tot`: mean total wind in cruise phase of flight
- `crz_gnd_speed`: mean ground speed in cruise phase of flight
- `fuel_efficiency_proxy`: mean ground speed subtracted by mean true airspeed over the cruise phase
- `descent_rate_mean`: mean descent rate in descent phase of flight
- `descent_rate_min`: min descent rate in descent phase of flight

At specific points in time, we extract the following features:
- `landing_tas`: true airspeed at landing
- `landing_temp`: temperature at landing
- `landing_sc`: specific_humidity at landing
- `dep_tas`: true airspeed at departure
- `dep_temp`: temperature at departure
- `dep_sc`: specific_humidity at departure
- `init_climb_rate`: mean climb rate at the first 30 seconds of the flight
- `time_to_10k`: time to reach 10k feet altitude

From the above features, we calculate the following additional features:
- `landing_pressure`: pressure at landing, calculated as $p = p_0  (1 - L h / T_0)^{g M / (R L)}$ using the elevation of the arrival aerodrome
- `dep_pressure`: pressure at departure, calculated as above
- `landing_density`: density at landing, calculated as $\rho = p / (R T)$ using the landing pressure and temperature
- `dep_density`: density at departure, calculated as above
- `landing_cas`: calibrated airspeed at landing, calculated using landing TAS, pressure and temperature according to [^2]
- `dep_cas`: calibrated airspeed at departure, calculated as above
- `landing_cas_squared`: square of the calibrated airspeed at landing
- `dep_cas_squared`: square of the calibrated airspeed at departure
- `landing_mass`: proxy for landing mass calculated from quantity proportional to lift, $F_L \propto \rho v_{\textrm{CAS}}^2$
- `dep_mass`: proxy for departure mass calculated as above

Finally, we calculate the average values of the features calculated using the parquet files for a given aircraft type to scale the features.


### OurAirports [`airport-data.csv`](data/airport-data.csv)
We use OurAirports[^3] to extract the following features:
- `elevation_adep`: elevation of the departure aerodrome
- `elevation_ades`: elevation of the arrival aerodrome
- `timezone_adep`: timezone of the departure aerodrome
- `timezone_ades`: timezone of the arrival aerodrome
- `haversine_distance`: haversine distance between the departure and arrival aerodrome, using the coordinates of the departure and arrival aerodromes


### Airplane Model specific masses [`masses.csv`](data/masses.csv)
Since the challenge data does not contain the specific aircraft variant, obtaining the operating empty weights and maximum takeoff mass can be challenging.
Individual aircraft type oew and mtom data was sourced from Wikipedia contributors, licensed under CC BY-SA 4.0 (compatible with the GPLv3.0 license).
Some of these were augmented, taking into account the challenge data set takeoff weights.
The following features are stored in the masses data file:
- `aircraft_type`: aircraft type (grouped the same as in the challenge data)
- `oew`: operating empty weight of the aircraft
- `mtom`: maximum takeoff weight of the aircraft



[^1]: This file is not provided in the repository due to its size. It can be generated from the separate daily files using the `combine_parquet.py` script.
[^2]: http://walter.bislins.ch/blog/index.asp?page=Fluggeschwindigkeiten%2C+IAS%2C+TAS%2C+EAS%2C+CAS%2C+Mach
[^3]: https://ourairports.com (the data is provided in the repository under unlicense license)
