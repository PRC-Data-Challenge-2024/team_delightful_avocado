import numpy as np

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from create_Xy import get_data
from data_augmentation import get_constructor


search_spaces = {
    "model__learning_rate": Real(0.01, 0.1),
    "model__max_iter": Integer(500, 15000),
    "model__min_samples_leaf": Integer(500, 1500),
    "model__l2_regularization": Real(4.0, 500.0),
    "model__max_leaf_nodes": Integer(250, 1500),
    "model__max_depth": Integer(20, 1500),
}


booster = HistGradientBoostingRegressor(
    random_state=42, categorical_features="from_dtype", max_bins=255
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

random_search = BayesSearchCV(
    estimator=pipeline,  # The pipeline to tune
    search_spaces=search_spaces,  # The hyperparameter search space
    n_iter=50,  # Number of iterations
    cv=10,  # 10-fold cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=2,  # Verbosity level
    random_state=42,  # Set random seed for reproducibility
)
np.int = int  # Fix for skopt bug
random_search.fit(X, y)

print(f"Best score: {random_search.best_score_}")
print(f"Best parameters: {random_search.best_params_}")
