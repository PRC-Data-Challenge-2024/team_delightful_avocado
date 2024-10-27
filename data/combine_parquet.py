import pandas as pd


def combine_all_csv_from_path(path: str):
    import os

    all_files = os.listdir(path)
    all_files = [f for f in all_files if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(path, f)) for f in all_files])
    return df


if __name__ == "__main__":
    path = "challenge_set"
    df = combine_all_csv_from_path(path)
    df.to_csv(f"{path}_parquet.csv", index=False)
