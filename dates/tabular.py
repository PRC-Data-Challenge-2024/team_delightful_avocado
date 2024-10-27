import pandas as pd

from dates.base import iata_schedule_flag, holidays


def create_time_features_tabular(date_col, arr=""):
    df = pd.DataFrame(
        {
            f"month{arr}": date_col.dt.month,
            f"day{arr}": date_col.dt.day,
            f"hour{arr}": date_col.dt.hour,
            f"minute{arr}": date_col.dt.minute,
            f"dayofweek{arr}": date_col.dt.dayofweek,
            f"dayofyear{arr}": date_col.dt.dayofyear,
            f"is_month_start{arr}": date_col.dt.is_month_start.astype(int),
            f"is_month_end{arr}": date_col.dt.is_month_end.astype(int),
            f"is_weekend{arr}": (date_col.dt.dayofweek >= 5).astype(int),
            f"quarter{arr}": date_col.dt.quarter,
            f"iata_schedule_flag{arr}": date_col.apply(iata_schedule_flag),
        }
    )
    df[f"is_holiday{arr}"] = date_col.apply(
        lambda x: int(x.date() in [h.date() for h in holidays])
    )
    return df
