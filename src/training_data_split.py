import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLS = ["x", "n", "t", "OIS", "OID", "k", "delta"]

def split_data(csv_path="data/ml_scenarios.csv"):
    df = pd.read_csv(csv_path)

    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss_train.split(df, groups=df["policy_id"]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss_valtest = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_valtest.split(temp_df, groups=temp_df["policy_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    assert set(train_df.policy_id) & set(val_df.policy_id) == set()
    assert set(train_df.policy_id) & set(test_df.policy_id) == set()
    assert set(val_df.policy_id) & set(test_df.policy_id) == set()

    return train_df, val_df, test_df
