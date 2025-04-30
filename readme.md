# Molecular Property Prediction

This project uses molecular features and SMILES chemical descriptors to train a machine learning model that predicts molecular stability (`T80`). It includes scripts for training the model and generating predictions.

---

## Requirements

Install the required Python packages with exact versions:

```bash
pip install numpy==1.24.2 pandas==2.0.0 rdkit==2024.3.5 joblib==1.4.2 scikit-learn==1.3.2
```

---

## Training the Model

To train the model, run:

```bash
python src/train.py --data_path ../data/train.csv --output_dir ../models
```

### Input

A CSV file with the following required columns:

- `Batch_ID`
- `Smiles` — valid SMILES strings for molecules
- `T80` — target value to predict
- One or more numeric feature columns

### Output

The following files will be saved in the `--output_dir`:

- `model.pkl` — trained SVR regression model
- `model_features.pkl` — list of features selected using Lasso

---

## Making Predictions

To make predictions on new data, run:

```bash
python src/predict.py --data_path ../data/test.csv --model_path ../models/model.pkl --output_path ../results/predictions.csv
```

### Input

A CSV file with the same structure and column names as the training file. The column `T80` can be left empty or filled with placeholder values.

### Output

A CSV file `predictions.csv` with the following columns:

- `Batch_ID` — taken from input
- `T80` — predicted values from the model

Invalid or unprocessable SMILES strings are handled gracefully, and `NaN` is returned for those rows in the output.

---

## Project Structure

```
project/
├── src/
│   ├── train.py              # training script
│   ├── predict.py            # prediction script
│   ├── smiles_parser.py      # SMILES to descriptors
│   ├── feature_builder.py    # unified feature construction
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── model.pkl
│   └── model_features.pkl
├── results/
│   └── predictions.csv
├── README.md
```

---

## Version Info

Verified library versions:

```
numpy==1.24.2  
pandas==2.0.0  
rdkit==2024.3.5  
joblib==1.4.2  
scikit-learn==1.3.2
```

Make sure to use these versions to ensure compatibility and reproducibility.

---

## Notes

- Molecular descriptors are computed using RDKit.
- Feature selection is performed using Lasso regression.
- The model is trained on log-transformed `T80` values and outputs predictions in original scale.
- Rows with invalid SMILES are excluded from training and marked with `NaN` in prediction results.

---