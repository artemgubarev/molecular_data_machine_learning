from rdkit import Chem
from rdkit.Chem import Descriptors

# get all features from 'Smiles'
def get_all_smiles_features(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    descriptor_funcs = [desc[1] for desc in Descriptors.descList]
    values = []
    for func in descriptor_funcs:
        try:
            values.append(func(mol))
        except Exception:
            values.append(None)
    return values