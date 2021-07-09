# -*- coding: utf-8 -*-
"""Fingerprint.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RD2tgwK_MFPMNuiuM2NuaN0oNhohNRyM
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from chainer_chemistry.datasets.molnet import get_molnet_dataset
"""
data = get_molnet_dataset(
    'qm9', 
    labels = 'cv',
    return_smiles = True, 
    frac_train = 1.0,
    frac_valid = 0.0,
    frac_test = 0.0
)

with open ("Data.pickle", 'wb') as f:
    pickle.dump(data,f)
"""
"""
with open ("Data.pickle", 'rb') as f:
   data = pickle.load(f)
"""
data =  pd.read_csv ('gen_SMILES2_accurjoback_qm9reprem.csv')
SMILES = []
cv = []

for smiles in data['SMILES']:
    SMILES.append(smiles)
    
for d in data['Desired hc']:
    cv.append(d)

print (SMILES[1])
m = AllChem.MolFromSmiles(SMILES[1])
output = Chem.MolToMolBlock(m)
print ("this is output: ", output)
print ("this is re.sub", re.sub('[\W+\d+H]', '', SMILES[1]))

print (len(re.sub('[\W+\d+H]', '', SMILES[1])))

print ("this is m: ", m)

coord = np.array([a.split()[:3] for a in output.split('\n')[4:(4+9)]]).astype(float)


print ("this is 2D coordinates", coord)
plt.scatter(coord[:,0], coord[:,1],
            c = ['black',
                 'blue',
                 'black',
                 'black',
                 'blue',
                 'black',
                 'black',
                 'red',
                 'blue'],
            s = [6 * 40,
                 7 * 40,
                 6 * 40,
                 6 * 40,
                 7 * 40,
                 6 * 40,
                 6 * 40,
                 8 * 40,
                 7 * 40],
            alpha = 0.5)
plt.savefig("smiles.png")
"""
coords = []
pure_atoms = []
_3Ds = 0

for smiles in SMILES:
    m = AllChem.MolFromSmiles(smiles)
    output = Chem.MolToMolBlock(m)
    if output.split('\n')[1].split()[1]=='3D':
        _3Ds += 1
    
    pure_atom = re.sub('[\W+\d+H]', '', smiles)
    stop = len(pure_atom)
    pure_atoms.append(pure_atom)
    
    coord = np.array([a.split()[:3] for a in output.split('\n')[4:(4+stop)]]).astype(float)
    coords.append(coord)
print ("this is _3Ds: ",_3Ds)
print ("len(coords) {}, len(SMILES) {}, len(cv) {}, len(pure_atoms) {}".format(len(coords), len(SMILES), len(cv), len(pure_atoms)))

print (coords[1])
with open('coordinates.pickle', 'wb') as f:
    pickle.dump((coords, SMILES, cv, pure_atoms), f)
"""
"""----"""

####

features = {
    'MolWt': Descriptors.MolWt,
    'HeavyAtomCount': Descriptors.HeavyAtomCount,
    'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt,
    'NumHAcceptors': Descriptors.NumHAcceptors,
    'NumHDonors': Descriptors.NumHDonors,
    'NumHeteroatoms': Descriptors.NumHeteroatoms,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumValenceElectrons': Descriptors.NumValenceElectrons,
    'NumAromaticRings': Descriptors.NumAromaticRings,
    'NumSaturatedRings': Descriptors.NumSaturatedRings,
    'NumAliphaticRings': Descriptors.NumAliphaticRings,
    'NumRadicalElectrons': Descriptors.NumRadicalElectrons,
    'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
    'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
    'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
    'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
    'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles,
    'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles
}

out_data = {}
for f in features.keys():
    out_data[f] = []

valid = []
counter = 0
print ("length SMILES before feature engineering: ",len(SMILES))
for i, smiles in enumerate(SMILES):
    
    if (i + 1) % 5000 == 0:
        print('Currently processed: {}/{}'.format(i+1, len(SMILES)))
    print (smiles)
    m = AllChem.MolFromSmiles(smiles)
    print (m) 
    for k, v in features.items():
        try:
            out_data[k].append(v(m))
            valid.append(i)   
        except:
            print ("None", counter)
print ("valid: ", valid)
valid = np.unique(valid)
valid = np.array(valid)
valid.reshape(-1)
print (valid.shape)
print (type(cv))
print (type(valid))
out_data['heat_capacity'] = np.take(cv, valid)
out_data['SMILES'] = np.take (SMILES, valid)
out_data = pd.DataFrame(out_data)

out_data.to_csv('Features_gen_smiles2.csv', index = False)