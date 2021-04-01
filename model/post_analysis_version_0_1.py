import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor

from rdkit import Chem

import warnings
warnings.filterwarnings('ignore')

import os
import re
from tensorflow import keras
import random
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import tensorflow as tf

from tensorflow.keras.layers import (Input, Dropout, LSTM, Reshape, LeakyReLU,
                          Concatenate, ReLU, Flatten, Dense, Embedding,
                          BatchNormalization, Activation, SpatialDropout1D,
                          Conv2D, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import  to_categorical
import np_utils
from IPython.display import clear_output
import matplotlib.pyplot as plt

from progressbar import ProgressBar
import seaborn as sns

from sklearn.metrics import r2_score

from rdkit import Chem

from tkinter import *

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2.5

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

preprocessor = GGNNPreprocessor()
"""
data = get_molnet_dataset('qm9',
                          labels = 'cv',
                          preprocessor = preprocessor,
                          return_smiles = True,
                          frac_train = 1.0,
                          frac_valid = 0.0,
                          frac_test = 0.0
                         )
with open('Data.pickle', 'wb') as f:
    pickle.dump(data, f)
"""
with open('Data.pickle', 'rb') as f:
    data = pickle.load(f)

gen_unique = pd.read_csv('generated_SMILES_new.csv')

gen_unique_ = gen_unique['SMILES'].values

smiles = data['smiles'][0]
# find the gen smiles similar to qm9
ins = 0
for G in gen_unique_:
    # smile = G[:-1]
    if G in smiles:
        ins += 1

print ("repetitive: ", ins, \
       "generated SMILES: ", len(gen_unique), \
       "Percent of Unique SMILES", (len(gen_unique_) - ins) / len(gen_unique))

print ("Min, Max, and mean of error desiredVSpred", \
        np.min(gen_unique['Error'].values), np.max(gen_unique['Error'].values), np.mean(gen_unique['Error'].values))

# Two kinds of error devision devide the error in:
# 1) 6 parts, (0-2.5),(2.5-5),(5-10),(10-20),(20-30),(>30)%-----lopo25 -> less than 0.025 (2.5%)
# 2) 3 parts, (0-10),(10-20),(>20)% ------------ b0p3 -> bigger than 0.3 (30%)
error_l0p025 = np.sum(gen_unique['Error'].values < 0.025)
error_b0p025_l0p05 = np.sum((gen_unique['Error'].values >= 0.025) & (gen_unique['Error'].values < 0.05))
error_b0p05_l0p1 =   np.sum((gen_unique['Error'].values >= 0.050) & (gen_unique['Error'].values < 0.10))
error_b0p1_l0p2  =   np.sum((gen_unique['Error'].values >= 0.100) & (gen_unique['Error'].values < 0.20))
error_b0p2_l0p3 =    np.sum((gen_unique['Error'].values >= 0.200) & (gen_unique['Error'].values < 0.30))
error_b0p3 =   np.sum(gen_unique['Error'].values > 0.300)
total = error_l0p025+error_b0p025_l0p05 + error_b0p05_l0p1 + error_b0p1_l0p2+ error_b0p2_l0p3 + error_b0p3

print ("total: ", total, "number of gen samples: ", len(gen_unique_))

# 6 parts relative Error Dist.
plt.close()
plt.figure(figsize = (18, 10))
x_labels = [  ]
frequencies = [ 100 * error_l0p025/total, 100*error_b0p025_l0p05/total, 100*error_b0p05_l0p1/total,
           100 * error_b0p1_l0p2/total, 100*error_b0p2_l0p3/total, 100*error_b0p3/total]
freq_series = pd.Series(frequencies)
ax = freq_series.plot(kind='bar', color = ['lightyellow', 'lemonchiffon', 'yellow', 'limegreen', 'darkgreen', 'black'])
rects = ax.patches

labels = ['<2.5%', '2.5%-5%', '5%-10%', '10%-20%', '20%-30%', '>30%']

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 3, label,fontsize = 15,
            ha='center', va='bottom')
ax.set_xticklabels(x_labels)
plt.ylim(0,100)

plt.yticks(fontsize = 20)
plt.ylabel("Percentage", fontsize=20)
plt.xlabel("Relative Error", fontsize=20)
plt.ylim (0, 100)
plt.savefig('errordist_6part.png')

# 3 parts Relative Error Dist.
plt.close()
plt.figure(figsize = (8, 5))
x_labels = [  ]
frequencies = [ 100*(error_l0p025 + error_b0p025_l0p05 + error_b0p05_l0p1)/total,
                100*error_b0p1_l0p2/total,
                100*(error_b0p2_l0p3 + error_b0p3)/total]
freq_series = pd.Series(frequencies)
ax = freq_series.plot(kind='bar', color = ['greenyellow', 'limegreen', 'darkgreen'])
rects = ax.patches

labels = ["<10%", "10%-20%", ">20%"]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 3, label,fontsize=20,
            ha='center', va='bottom')
ax.set_xticklabels(x_labels)
plt.ylim(0,100)

plt.yticks(fontsize = 20)
plt.ylabel("Percentage", fontsize=20)
plt.xlabel("Relative Error", fontsize=20)
plt.ylim (0, 100)
plt.savefig('errordist_3part.png')
################################
# Random generation and error distribution
####################################
# number of samples
n = 10000000

with open('image_train.pickle', 'rb') as f:
    X_smiles_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)

with open('image_test.pickle', 'rb') as f:
    X_smiles_val, X_atoms_val, X_bonds_val, y_val = pickle.load(f)

# outlier removal
IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)
lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR
idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]

#s = y_train
y_train_norm =   np.random.normal(np.mean(y_train), np.std(y_train), n)
desired_values = np.random.uniform(1.3*np.min(y_train_norm), np.max(y_train_norm), n)
error = np.abs((np.mean(y_train_norm) - desired_values)/desired_values)

error_l0p1_simulation = np.sum(error<0.1)/n
error_b0p1_l0p2_simulation = np.sum((error<0.2) & (error>0.1))/n
error_b0p2_simulation = np.sum(error>0.2)/n

# simulated relative error
plt.close()
plt.figure(figsize = (8, 5))
x_labels = [  ]
frequencies = [100*error_l0p1_simulation, 100*error_b0p1_l0p2_simulation,100*error_b0p2_simulation]
freq_series = pd.Series(frequencies)
ax = freq_series.plot ( kind='bar', color = ['darkgrey','grey','black'] )
rects = ax.patches

labels = ["<10%", "10%-20%", ">20%"]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 3, label,fontsize=20,
            ha='center', va='bottom')
ax.set_xticklabels(x_labels)
plt.ylim(0,100)

plt.yticks(fontsize = 20)
plt.ylabel("Percentage", fontsize=20)
plt.xlabel("Relative Error (mean value)", fontsize=20)
plt.ylim (0, 100)
plt.savefig('errordist_3part_simulated.png')

