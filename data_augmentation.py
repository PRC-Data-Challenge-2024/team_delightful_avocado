from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import pandas as pd
import numpy as np

from dates.tabular import create_time_features_tabular


class Augmentation(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model,
        one_hot_features,
        num_features,
        cat_features=None,
        ordinal_features=None,
    ):
        self.one_hot_features = one_hot_features
        self.num_features = num_features
        self.ordinal_features = ordinal_features
        self.cat_features = cat_features
        self.model = model
        self.adep_encoder = OneHotEncoder(
            min_frequency=50,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )
        self.ades_encoder = OneHotEncoder(
            min_frequency=50,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )
        self.pair_encoder = OneHotEncoder(
            min_frequency=450,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )

    def fit(self, X, y=None):
        self.adep_encoder.fit(X[["adep"]])
        self.ades_encoder.fit(X[["ades"]])
        self.pair_encoder.fit(X[["airport_pair"]])
        return self

    def transform(self, X, y=None):
        X["date_col"] = pd.to_datetime(X["actual_offblock_time"])
        date_col = X.apply(
            lambda x: x["date_col"].tz_convert(x["time_zone_adep"]).tz_localize(None),
            axis=1,
        )
        date_col = pd.to_datetime(date_col)

        df = create_time_features_tabular(date_col)

        X["date_col"] = pd.to_datetime(X["arrival_time"])
        date_col = X.apply(
            lambda x: x["date_col"].tz_convert(x["time_zone_adep"]).tz_localize(None),
            axis=1,
        )
        date_col = pd.to_datetime(date_col)
        df2 = create_time_features_tabular(date_col, "_arr")
        df = pd.concat([df, df2], axis=1)

        df["mtom"] = X["mtom"]
        df["oew"] = X["oew"]
        if self.model == "tabular":
            df["adep"] = self.adep_encoder.inverse_transform(
                self.adep_encoder.transform(X[["adep"]])
            ).reshape(-1)
            df["ades"] = self.ades_encoder.inverse_transform(
                self.ades_encoder.transform(X[["ades"]])
            ).reshape(-1)
            df["airport_pair"] = self.pair_encoder.inverse_transform(
                self.pair_encoder.transform(X[["airport_pair"]])
            ).reshape(-1)
        elif self.model == "shap":
            df["adep"] = np.zeros(X.shape[0])
            df["ades"] = np.zeros(X.shape[0])
            df["airport_pair"] = np.zeros(X.shape[0])

        df["adep"] = df["adep"].astype("category")
        df["ades"] = df["ades"].astype("category")
        df["airport_pair"] = df["airport_pair"].astype("category")
        for col in self.cat_features:
            df[col] = X[col].astype("category")
        for col in self.num_features:
            df[col] = X[col]
        return df


def get_constructor(model: str):
    # what about in phases?
    parquet_features = [
        "avg_tas",
        "climb_rate_mean",
        "avg_wind_u",
        "avg_wind_v",
        "climb_rate_max",
        "descent_rate_mean",
        "descent_rate_min",
        "max_crz_alt",
        "mean_crz_alt",
        "crz_tas",
        "crz_wind_u",
        "crz_wind_v",
        "crz_wind_tot",
        "crz_gnd_speed",
        "adsb_inflight_time",
        "landing_tas",
        "landing_temp",
        "landing_cas",
        "landing_cas_squared",
        "landing_density",
        "landing_mass",
        "landing_pressure",
        "dep_tas",
        "dep_temp",
        "dep_pressure",
        "dep_density",
        "dep_cas",
        "dep_cas_squared",
        "dep_mass",
        "init_climb_rate",
        "time_to_10k",
        "dep_sc",
        "landing_sc",
        "climb_efficiency",
        "fuel_efficiency_proxy",
        "acft-avg_tas",
        "acft-avg_wind_u",
        "acft-avg_wind_v",
        "acft-climb_rate_mean",
        "acft-climb_rate_max",
        "acft-descent_rate_mean",
        "acft-descent_rate_min",
        "acft-max_crz_alt",
        "acft-adsb_inflight_time",
        "acft-landing_tas",
        "acft-landing_temp",
        "acft-dep_temp",
        "acft-dep_tas",
        "acft-init_climb_rate",
        "acft-time_to_10k",
        "acft-dep_sc",
        "acft-landing_sc",
        "acft-climb_efficiency",
        "acft-climb_in_phases",
        "acft-cruise_in_phases",
        "acft-descent_in_phases",
        "acft-ground_in_phases",
        "acft-mean_crz_alt",
        "acft-crz_tas",
        "acft-crz_wind_u",
        "acft-crz_wind_v",
        "acft-crz_wind_tot",
        "acft-crz_gnd_speed",
        "acft-fuel_efficiency_proxy",
        "acft-flight_duration",
        "acft-taxiout_time",
        "acft-flown_distance",
    ]

    cat_columns = [
        "aircraft_type",
        "airline",
        "country_code_adep",
        "country_code_ades",
        "wtc",
    ]
    num_columnns = [
        "flight_duration",
        "taxiout_time",
        "flown_distance",
        "mtom",
        "oew",
        "elevation_adep",
        "elevation_ades",
        "haversine",
        "flights_per_day",
    ]
    constructor = Augmentation(
        model=model,
        one_hot_features=None,
        cat_features=cat_columns,
        num_features=num_columnns + parquet_features,
    )
    return constructor
