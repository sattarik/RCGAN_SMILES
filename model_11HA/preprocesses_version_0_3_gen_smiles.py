# Task:
# Turn data into "images"
# Two networks
# GAN: generate atoms and bonds (adjacency) layers
# simple CNN: turning SMILES layer to atoms+bonds layers

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor, construct_atomic_number_array


from rdkit import Chem

"""Chem.MolFromSmiles('CC1CC(O)C2(CC2)O1')"""

preprocessor = GGNNPreprocessor()
#atom_num = construct_atomic_number_array()
"""
data = get_molnet_dataset('qm9',
                          labels = 'cv',
                          preprocessor = preprocessor,
                          return_smiles = True,
                          frac_train = 1.0,
                          frac_valid = 0.0,
                          frac_test = 0.0
                         )

"""
data_gen = pd.read_csv('./../experiments/regular/DFT_eval/reg_dfteval.csv')
heat_capacity_des, gen_smiles = data_gen.iloc[:,-3].values, data_gen.iloc[:,0].values

print (gen_smiles)
print (heat_capacity_des)

with open ('./../data/trainingsets/20000_train_regular_qm9/image_train.pickle', 'rb') as f: 
    data = pickle.load(f)

X_smiles = []
X_gen_smiles = []

X_atoms = []
X_gen_atoms = []

X_bonds = []
X_gen_bonds = []
y = []

atom_lengths = []
atom_max = []
bonds_lengths = []
# get the mol from rdkit and then get atomic number of atoms from chainer_chemistry
for smile in gen_smiles:
    mol = Chem.MolFromSmiles(smile)
    X_gen_atoms_ = construct_atomic_number_array(mol)
    X_gen_atoms.append(X_gen_atoms_)
    X_gen_bonds_ = preprocessor.get_input_features(mol)[1]
    X_gen_bonds.append(X_gen_bonds_)

# excluding long atoms to use the same model trained on qm9 data
exclude = []
max_gen_atoms = 9
for i,ii in enumerate(X_gen_atoms):
    if len(ii) > max_gen_atoms:
        exclude.append(i)
print (exclude)
exclude = np.asarray(exclude, dtype = 'int')
idx = np.setdiff1d(np.arange(len(heat_capacity_des)), exclude)
idx = np.asarray(idx, dtype = 'int')
X_gen_atoms_ = []
X_gen_bonds_ = []
gen_smiles_ = []
heat_capacity_des_ = []
for i,ii in enumerate (idx):
    X_gen_atoms_.append (X_gen_atoms [ii])
    X_gen_bonds_.append (X_gen_bonds [ii])
    gen_smiles_.append (gen_smiles [ii])
    heat_capacity_des_.append (heat_capacity_des [ii])

X_gen_atoms = X_gen_atoms_
X_gen_bonds =  X_gen_bonds_
gen_smiles = gen_smiles_
heat_capacity_des = heat_capacity_des_

print (len(heat_capacity_des))


for smiles in data['smiles'][0]:
    smiles += '.'
    X_smiles.append(smiles)

for smile in gen_smiles:
    smile += '.'
    X_gen_smiles.append(smiles)

for d in data['dataset'][0]:
    X_atoms.append(d[0])
    X_bonds.append(d[1])

    atom_lengths.append(len(d[0]))
    atom_max.append(np.max(d[0]))
    bonds_lengths.append(d[1].shape[1])

    y.append(d[2])

with open('database_SMILES.pickle', 'wb') as f:
    pickle.dump((X_smiles, X_atoms, X_bonds, y), f)

with open('database_gen_SMILES.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles, X_gen_atoms, X_gen_bonds, y), f)



MAX_NB_WORDS = 23
MAX_SEQUENCE_LENGTH = 35

tokenizer = Tokenizer(num_words = MAX_NB_WORDS,
                      char_level = True,
                      filters = '',
                      lower = False)
tokenizer.fit_on_texts(X_smiles)
#tokenizer.fit_on_texts(X_gen_smiles)

X_smiles = tokenizer.texts_to_sequences(X_smiles)
X_gen_smiles = tokenizer.texts_to_sequences(X_gen_smiles)

X_smiles = pad_sequences(X_smiles,
                         maxlen = MAX_SEQUENCE_LENGTH,
                         padding = 'post')
X_gen_smiles = pad_sequences(X_gen_smiles,
                             maxlen = MAX_SEQUENCE_LENGTH,
                             padding = 'post')

X_smiles = to_categorical(X_smiles)
X_gen_smiles = to_categorical(X_gen_smiles, num_classes = 23)

atom_max = np.max(atom_max)
bonds_max = np.max(bonds_lengths)

X_atoms_ = []
for atom in X_atoms:
    if len(atom) < atom_max:
        pad_len = atom_max - len(atom)
        atom = np.pad(atom, (0, pad_len))

    X_atoms_.append(atom)

X_atoms = np.asarray(X_atoms_)

print (X_atoms.shape)
X_atoms = to_categorical(X_atoms)
print (X_atoms.shape)

X_gen_atoms_ = []
atom_max = atom_max
for atom in X_gen_atoms:
    if len(atom) < max_gen_atoms:
        pad_len = max_gen_atoms - len(atom)
        atom = np.pad(atom, (0, pad_len))

    X_gen_atoms_.append(atom)

#X_gen_atoms = np.asarray(X_gen_atoms_)
X_gen_atoms = np.vstack (X_gen_atoms_)
print (X_gen_atoms)
print ((X_gen_atoms.shape))
X_gen_atoms = to_categorical(X_gen_atoms)
print ("after to_categorical", X_gen_atoms.shape)
X_bonds_ = []
for bond in X_bonds:
    if bond.shape[1] < bonds_max:
        pad_len = bonds_max - bond.shape[1]
        bond = np.pad(bond, ((0,0),(0,pad_len),(0,pad_len)))

    X_bonds_.append(bond)

X_bonds = np.asarray(X_bonds_)
print (X_bonds.shape)
X_gen_bonds_ = []
print (bonds_max )
bonds_max = bonds_max
for bond in X_gen_bonds:
    if bond.shape[1] < bonds_max:
        pad_len = bonds_max - bond.shape[1]
        bond = np.pad(bond, ((0,0),(0,pad_len),(0,pad_len)))

    X_gen_bonds_.append(bond)

X_gen_bonds = np.asarray(X_gen_bonds_)
print (type(X_gen_bonds))
print (X_gen_bonds.shape)
SHAPE = list(X_smiles.shape) + [1]
X_smiles = X_smiles.reshape(SHAPE)
print ("X_smiles shape: ", X_smiles.shape)
SHAPE = list(X_gen_smiles.shape) + [1]
X_gen_smiles = X_gen_smiles.reshape(SHAPE)
print ("X_gen_smiles shape:  ", X_gen_smiles.shape)

y = np.asarray(y).reshape([-1])
print ("line 189, ", X_atoms.shape)
SHAPE = list(X_atoms.shape) + [1]
print (SHAPE)
X_atoms = X_atoms.reshape(SHAPE)
print (X_atoms.shape)
X_bonds = X_bonds.transpose([0,2,3,1])
print (X_bonds.shape)
SHAPE = list(X_gen_atoms.shape) + [1]
X_gen_atoms = X_gen_atoms.reshape(SHAPE)
print (X_gen_atoms.shape)
X_gen_bonds = X_gen_bonds.transpose([0,2,3,1])
print (X_gen_bonds.shape)
####
# ANALYSIS
sns.distplot(y);
plt.savefig('dis_y.png')

sns.distplot(heat_capacity_des);
plt.savefig('dis_hearDesired.png')
####

# TRAIN/VAL split
idx = np.random.choice(len(y), int(len(y) * 0.2), replace = False)
train_idx = np.setdiff1d(np.arange(len(y)), idx)

X_smiles_test, X_atoms_test, X_bonds_test, y_test = X_smiles[idx], X_atoms[idx], X_bonds[idx], y[idx]
X_smiles_train, X_atoms_train, X_bonds_train, y_train = X_smiles[train_idx], X_atoms[train_idx], X_bonds[train_idx], y[train_idx]

# TRAIN/VAL split gen_data
idx = np.random.choice(len(heat_capacity_des), int(len(heat_capacity_des) * 0.15), replace = False)
train_idx = np.setdiff1d(np.arange(len(heat_capacity_des)), idx)

X_gen_smiles, X_gen_atoms, X_gen_bonds, heat_capacity_des = np.asarray(X_gen_smiles), np.asarray(X_gen_atoms), np.asarray(X_gen_bonds), np.asarray(heat_capacity_des)

X_gen_smiles_test, X_gen_atoms_test, X_gen_bonds_test, heat_capacity_des_test = X_gen_smiles[idx], X_gen_atoms[idx], X_gen_bonds[idx], heat_capacity_des[idx]
X_gen_smiles_train, X_gen_atoms_train, X_gen_bonds_train, heat_capacity_des_train = X_gen_smiles[train_idx], X_gen_atoms[train_idx], X_gen_bonds[train_idx], heat_capacity_des[train_idx]

####
# ANALYSIS
X_atoms_train_ = X_atoms_train.reshape([X_atoms_train.shape[0],
                                        9 * 10])
X_bonds_train_ = X_bonds_train.reshape([X_bonds_train.shape[0],
                                        9 * 9 * 4])

X_atoms_test_ = X_atoms_test.reshape([X_atoms_test.shape[0],
                                      9 * 10])
X_bonds_test_ = X_bonds_test.reshape([X_bonds_test.shape[0],
                                      9 * 9 * 4])

# ANALYSIS gen_data
X_gen_atoms_train_ = X_gen_atoms_train.reshape([X_gen_atoms_train.shape[0],
                                        9 * 10])
X_gen_bonds_train_ = X_gen_bonds_train.reshape([X_gen_bonds_train.shape[0],
                                        9 * 9 * 4])
X_gen_atoms_test_ = X_gen_atoms_test.reshape([X_gen_atoms_test.shape[0],
                                        9 * 10])
X_gen_bonds_test_ = X_gen_bonds_test.reshape([X_gen_bonds_test.shape[0],
                                        9 * 9 * 4])


pca_1 = PCA()
X_atoms_train_ = pca_1.fit_transform(X_atoms_train_)
X_atoms_test_ = pca_1.transform(X_atoms_test_)

pca_1 = PCA()
X_gen_atoms_train_ = pca_1.fit_transform(X_gen_atoms_train_)
X_gen_atoms_test_ = pca_1.fit_transform(X_gen_atoms_test_)

pca_2 = PCA()
X_bonds_train_ = pca_2.fit_transform(X_bonds_train_)
X_bonds_test_ = pca_2.transform(X_bonds_test_)

pca_2 = PCA()
X_gen_bonds_train_ = pca_2.fit_transform(X_gen_bonds_train_)
X_gen_bonds_test_ = pca_2.fit_transform(X_gen_bonds_test_)
# Atoms Distribution
plt.clf()
plt.scatter(X_atoms_train_[:,0], X_atoms_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("X_atoms_train_qm9.png")
plt.clf()
plt.scatter(X_atoms_test_[:,0], X_atoms_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("X_atoms_test_qm9.png")
# Atoms Distribution
plt.clf()
plt.scatter(X_gen_atoms_train_[:,0], X_gen_atoms_train_[:,1], alpha = 0.3, c = 'black');
plt.savefig("X_gen_atoms_train.png")
plt.clf()
plt.scatter(X_gen_atoms_test_[:,0], X_gen_atoms_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("X_gen_atoms_test.png")

####

# Bonds Distribution
plt.clf()
plt.scatter(X_bonds_train_[:,0], X_bonds_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("X_bonds_train_qm9.png")
plt.clf()
plt.scatter(X_bonds_test_[:,0], X_bonds_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("X_bonds_test_qm9.png")
plt.clf()

# Bonds Distribution
plt.clf()
plt.scatter(X_gen_bonds_train_[:,0], X_gen_bonds_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("X_gen_bonds_train.png")
plt.clf()
plt.scatter(X_gen_bonds_test_[:,0], X_gen_bonds_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("X_gen_bonds_test.png")
plt.clf()

# subsampling
idx = np.random.choice(len(y_train), int(len(y_train) * 0.25), replace = False)

X_smiles_train, X_atoms_train, X_bonds_train, y_train = (X_smiles_train[idx],
                                                X_atoms_train[idx],
                                                X_bonds_train[idx],
                                                y_train[idx])


idx = np.random.choice(len(y_test), int(len(y_test) * 0.1667), replace = False)

X_smiles_test, X_atoms_test, X_bonds_test, y_test = (X_smiles_test[idx],
                                                     X_atoms_test[idx],
                                                     X_bonds_test[idx],
                                                     y_test[idx])

with open('image_train.pickle', 'wb') as f:
    pickle.dump((X_smiles_train, X_atoms_train, X_bonds_train, y_train), f)

with open('image_test.pickle', 'wb') as f:
    pickle.dump((X_smiles_test, X_atoms_test, X_bonds_test, y_test), f)

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer.index_word, f)

with open('database.pickle', 'wb') as f:
    pickle.dump((X_smiles, X_atoms, X_bonds, y), f)

with open('gen_smiles2_atombond.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles, X_gen_atoms, X_gen_bonds, heat_capacity_des),f)

with open('gen_smiles2_atombond_train.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles_train, X_gen_atoms_train, X_gen_bonds_train, heat_capacity_des_train),f)

with open('gen_smiles2_atombond_test.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles_test, X_gen_atoms_test, X_gen_bonds_test, heat_capacity_des_test),f)


