from datetime import datetime

holidays = [
    datetime(2022, 12, 25),
    datetime(2022, 1, 1),
]


def iata_schedule_flag(date):
    # Define IATA schedule start and end dates for 2022
    iata_summer_start = datetime(2022, 3, 27)  # Last Sunday in March
    iata_summer_end = datetime(2022, 10, 29)  # Last Saturday in October
    # Check if date is within the summer schedule
    if iata_summer_start <= date.replace(tzinfo=None) < iata_summer_end:
        return 0
    else:
        return 1
