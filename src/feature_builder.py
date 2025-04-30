import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from smiles_parser import get_all_smiles_features

descriptor_names = [desc[0] for desc in Descriptors.descList]

def build_features(data: pd.DataFrame, drop_invalid_rows: bool):
    """
    Returns:
        - X_full: pd.DataFrame of combined numerical + descriptor features
        - valid_mask: list of booleans indicating which rows are valid (for predict)
        - error_indices: indices of rows with SMILES parsing failures
    """
    X_num = data.drop(['T80', 'Batch_ID', 'Smiles'], axis=1, errors='ignore')
    smiles_list = data['Smiles']
    smiles_descriptor_matrix = []
    valid_mask = []

    for smiles in smiles_list:
        try:
            descriptor_values = get_all_smiles_features(smiles)
            if any(v is None for v in descriptor_values):
                raise ValueError("Descriptor is None")
            smiles_descriptor_matrix.append(descriptor_values)
            valid_mask.append(True)
        except Exception:
            smiles_descriptor_matrix.append([np.nan] * len(descriptor_names))
            valid_mask.append(False)

    smiles_features = pd.DataFrame(smiles_descriptor_matrix, columns=descriptor_names)
    
    if drop_invalid_rows:
        mask = pd.Series(valid_mask)
        X_num = X_num[mask].reset_index(drop=True)
        smiles_features = smiles_features[mask].reset_index(drop=True)
        return pd.concat([X_num, smiles_features], axis=1), None, None
    else:
        return pd.concat([X_num.reset_index(drop=True), smiles_features.reset_index(drop=True)], axis=1),valid_mask, smiles_features.isna().any(axis=1).to_list()