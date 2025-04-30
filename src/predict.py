import argparse
import os
import joblib
import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from feature_builder import build_features

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

# paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, args.data_path)
model_path = os.path.join(base_dir, args.model_path)
output_path = os.path.join(base_dir, args.output_path)
feature_path = os.path.join(os.path.dirname(model_path), "model_features.pkl")

# load model and selected features
model = joblib.load(model_path)
selected_features = joblib.load(feature_path)

# load input data
data = pd.read_csv(data_path)
batch_ids = data['Batch_ID']

# build features, keep all rows
X_full, valid_mask, _ = build_features(data, drop_invalid_rows=False)

# select only the features used during training
X_full = X_full[selected_features]

# generate predictions where valid
y_pred = []
for i, valid in enumerate(valid_mask):
    if valid:
        x_row = X_full.iloc[[i]]
        y_log = model.predict(x_row)[0]
        y_value = np.expm1(y_log)
        y_pred.append(y_value)
    else:
        y_pred.append(np.nan)

# save result with Batch_ID and predicted T80
result_df = pd.DataFrame({
    "Batch_ID": batch_ids,
    "T80": y_pred
})
result_df.to_csv(output_path, index=False)

print(f"✅ Predictions saved to: {output_path}")