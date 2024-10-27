import pickle

import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from create_Xy import get_data
from data_augmentation import get_constructor

# Parameters found using Bayesian optimization with cross-validation, slightly modified
booster = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    max_bins=255,
    learning_rate=0.0227322,
    max_iter=10000,
    max_leaf_nodes=500,
    min_samples_leaf=500,
    l2_regularization=300.0,
    max_depth=1000,
    random_state=42,
)

constructor = get_constructor("tabular")
pipeline = Pipeline(
    [
        ("feature_union", constructor),
        ("model", booster),
    ]
)
data = get_data("challenge_set")
X = data.drop("tow", axis=1)
y = data["tow"]

pipeline.fit(X, y)

sub_set = get_data("final_submission_set")
output = pipeline.predict(sub_set)
df = pd.DataFrame()
df["tow"] = output
df["flight_id"] = sub_set["flight_id"]
df.to_csv("submission.csv", index=False)

with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
