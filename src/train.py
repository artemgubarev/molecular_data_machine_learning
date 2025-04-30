import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from feature_builder import build_features

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

# directories and file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, args.data_path)
output_dir = os.path.join(base_dir, args.output_dir)

os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.pkl")
features_path = os.path.join(output_dir, "model_features.pkl")

# load dataset
data = pd.read_csv(data_path)
y = data['T80']
y_log = np.log1p(y)

# build features and drop invalid rows
X, _, _ = build_features(data, drop_invalid_rows=True)
X = X.dropna()
y_log = y_log[X.index]

# feature selection using Lasso
def select_features_lasso(X, y):
    lasso = Lasso(max_iter=10000)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(lasso, threshold='mean')),
        ('regressor', Lasso(max_iter=10000))
    ])
    param_grid = {
        'feature_selection__estimator__alpha': [0.01, 0.1, 1, 10],
        'regressor__alpha': [0.01, 0.1, 1, 10]
    }
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_log_error', n_jobs=-1)
    grid_search.fit(X, y)
    best_pipeline = grid_search.best_estimator_
    selected_mask = best_pipeline.named_steps['feature_selection'].get_support()
    selected_features = X.columns[selected_mask]
    return selected_features

selected_features = select_features_lasso(X, y_log)
X = X[selected_features]

# train SVR model
svr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', SVR(kernel='linear', C=1, epsilon=0.01, gamma='scale'))
])
svr_model.fit(X, y_log)

# save model and selected feature list
joblib.dump(svr_model, model_path)
joblib.dump(list(selected_features), features_path)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Selected features saved to: {features_path}")
