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

with open('image_train.pickle', 'rb') as f:
    X_smiles_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)
    
with open('image_test.pickle', 'rb') as f:
    X_smiles_val, X_atoms_val, X_bonds_val, y_val = pickle.load(f)

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

tokenizer[0] = ' '

# Outlier removal
IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)

lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR

idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]

def norm(X: ndarray) -> ndarray:
    X = np.where(X == 0, -1.0, 1.0)
    return X

X_atoms_train, X_bonds_train = (norm(X_atoms_train),
                                norm(X_bonds_train))
X_atoms_val, X_bonds_val = (norm(X_atoms_val),
                            norm(X_bonds_val))

def y_norm(y: ndarray) -> ndarray:
    scaler_min = np.min(y)
    scaler_max = np.max(y)
    
    y = (y - scaler_min) / (scaler_max - scaler_min)
    
    return y, scaler_min, scaler_max

y_train, s_min, s_max = y_norm(y_train)
y_val = (y_val - s_min) / (s_max - s_min)

encoder = load_model('encoder.h5')
decoder = load_model('decoder.h5')

# Regressor
inp1 = Input(shape = [6, 6, 1])
inp2 = Input(shape = [6, 6, 1])

yr = Concatenate()([inp1, inp2])

tower0 = Conv2D(32, 1, padding = 'same')(yr)
tower1 = Conv2D(64, 1, padding = 'same')(yr)
tower1 = Conv2D(64, 3, padding = 'same')(tower1)
tower2 = Conv2D(32, 1, padding = 'same')(yr)
tower2 = Conv2D(32, 5, padding = 'same')(tower2)
tower3 = MaxPooling2D(3, 1, padding = 'same')(yr)
tower3 = Conv2D(32, 1, padding = 'same')(tower3)
h = Concatenate()([tower0, tower1, tower2, tower3])
h = ReLU()(h)
h = MaxPooling2D(2, 1, padding = 'same')(h)

for i in range(6):
    tower0 = Conv2D(32, 1, padding = 'same')(h)
    tower1 = Conv2D(64, 1, padding = 'same')(h)
    tower1 = Conv2D(64, 3, padding = 'same')(tower1)
    tower2 = Conv2D(32, 1, padding = 'same')(h)
    tower2 = Conv2D(32, 5, padding = 'same')(tower2)
    tower3 = MaxPooling2D(3, 1, padding = 'same')(h)
    tower3 = Conv2D(32, 1, padding = 'same')(tower3)
    h = Concatenate()([tower0, tower1, tower2, tower3])
    h = ReLU()(h)
    if i % 2 == 0 and i != 0:
        h = MaxPooling2D(2, 1, padding = 'same')(h)
h = BatchNormalization()(h)

yr = Flatten()(h)
o = Dropout(0.2)(yr)
o = Dense(128)(o)

o_reg = Dropout(0.2)(o)
o_reg = Dense(1, activation = 'sigmoid')(o_reg)

regressor = Model([inp1, inp2], o_reg)
regressor_top = Model([inp1, inp2], o)

regressor.compile(loss = 'mse', optimizer = Adam(1e-5))
regressor_top.compile(loss = 'mse', optimizer = Adam (1e-5))


train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])

atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
atoms_val, bonds_val, _ = encoder.predict([X_atoms_val, X_bonds_val])

regressor = load_model('regressor.h5')
regressor_top = load_model('regressor_top.h5')
regressor.fit([atoms_embedding, bonds_embedding], 
              y_train,
              validation_data = ([atoms_val,
                                  bonds_val],
                                 y_val),
              batch_size = 32,
              epochs = 100,
              verbose = 1)

# Validating the regressor
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor: {}'.format(r2_score(y_train, pred.reshape([-1]))))
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_val, X_bonds_val])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor: {}'.format(r2_score(y_val, pred.reshape([-1]))))



