from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# ---------- Data Preprocessing ----------
def process_census(path="./adult.data"):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'martial_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target'
    ]

    df = pd.read_csv(
        path, names=column_names, na_values="?",
        sep=r'\s*,\s*', engine='python'
    ).loc[lambda d: d['race'].isin(['White', 'Black'])]

    # Binary sensitive attrs
    df['race'] = (df['race'] == 'White').astype(int)
    df['sex'] = (df['sex'] == 'Male').astype(int)

    y = (df['target'] == '>50K').astype(int)
    X = df.drop(columns=['target', 'race', 'sex', 'fnlwgt']).fillna('Unknown')
    X = pd.get_dummies(X, drop_first=True)

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
