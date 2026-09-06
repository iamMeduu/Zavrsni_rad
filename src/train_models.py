import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from training_data_split import split_data, FEATURE_COLS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error

train_df, val_df, test_df = split_data()

print(f"Train: {len(train_df)} rows, {train_df.policy_id.nunique()} policies")
print(f"Val:   {len(val_df)} rows, {val_df.policy_id.nunique()} policies")
print(f"Test:  {len(test_df)} rows, {test_df.policy_id.nunique()} policies")

X_train, y_train = train_df[FEATURE_COLS], train_df["surrender_value"]
X_val, y_val     = val_df[FEATURE_COLS],   val_df["surrender_value"]
X_test, y_test   = test_df[FEATURE_COLS],  test_df["surrender_value"]

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_or_train_linear_regression(X_train, y_train):
    path = os.path.join(MODELS_DIR, "linear_regression.joblib")
    if os.path.exists(path):
        print("Loading already trained model: linear_regression")
        return joblib.load(path)
    print("Training new model: linear_regression")
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model

def get_or_train_decision_tree(X_train, y_train, max_depth=None):
    path = os.path.join(MODELS_DIR, "decision_tree.joblib")
    if os.path.exists(path):
        print("Loading already trained model: decision_tree")
        return joblib.load(path)
    print(f"Training new model: decision_tree (max_depth={max_depth})")
    model = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model

best_depth, best_val_mae = None, float("inf")
for depth in [3, 5, 7, 10, 15, 20, 30, None]:
    candidate = DecisionTreeRegressor(random_state=42, max_depth=depth)
    candidate.fit(X_train, y_train)
    val_mae = mean_absolute_error(y_val, candidate.predict(X_val))
    print(f"max_depth={depth}: val MAE={val_mae:.4f}")
    if val_mae < best_val_mae:
        best_val_mae, best_depth = val_mae, depth

print(f"\nBest max_depth according to the validation set: {best_depth} (val MAE={best_val_mae:.4f})\n")

linreg = get_or_train_linear_regression(X_train, y_train)
tree = get_or_train_decision_tree(X_train, y_train, max_depth=best_depth)

models = {
    "Linear regression": linreg,
    "Decision tree": tree,
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "Max error": max_error(y_test, y_pred),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))