# Strategy 1:
# Generate data after each epoch of training, if less than
# 10% error rate, and is a legit SMILES
# append to the real data
# Otherwise, append to fake data

# ADDING REINFORCEMENT MECHANISM
# Regenerate Normal sampling (define ranges), default: uniform

# IMPORTANT!!!!!!!!!!!!! DO NOT DROP DUPLICATE FOR RESULT .CSV

import warnings
warnings.filterwarnings('ignore')

import time
import os
import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

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

import np_utils

from tensorflow.keras.utils import  to_categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt

from progressbar import ProgressBar
import seaborn as sns

from sklearn.metrics import r2_score


print ("!!!!!!!!! we are just before importing rdkit!!!!!")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")

from scipy.stats import truncnorm
from sklearn.decomposition import PCA

import matplotlib.ticker as tk

import ntpath
import re

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

from matplotlib.colors import ListedColormap

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

with open('./../data/trainingsets/20000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)
    
with open('./../data/trainingsets/20000_train_regular_qm9/image_test.pickle', 'rb') as f:
    X_smiles_val, X_atoms_val, X_bonds_val, y_val = pickle.load(f)

with open('./../data/trainingsets/20000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
    
tokenizer[0] = ' '

# Subsampling has been done in the data preprocesses
"""
# Outlier removal
IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)

lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR

idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]


# Outlier removal
IQR = - np.quantile(y_val, 0.25) + np.quantile(y_val, 0.75)

lower_bound, upper_bound = np.quantile(y_val, 0.25) - 1.5 * IQR, np.quantile(y_val, 0.75) + 1.5 * IQR

idx = np.where((y_val >= lower_bound) & (y_val <= upper_bound))

y_val = y_val[idx]
X_smiles_val = X_smiles_val[idx]
X_atoms_val = X_atoms_val[idx]
X_bonds_val = X_bonds_val[idx]
"""

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

s_min1 = np.min (y_train)
s_max1 = np.max (y_train)

s_min2 = np.min(y_val)
s_max2 = np.max(y_val)
s_min = min(s_min1, s_min2)
s_max = max(s_max1, s_max2)

y_val = (y_val - s_min) / (s_max - s_min)
print ("min and max train data and test normalized", s_min, s_max, np.min(y_val), np.max(y_val))
# define s_min and s_max between 10-7s_max = 50
#s_min, s_max = 20, 50
y_train = (y_train - s_min) / (s_max - s_min)
print ("min and max train data and train normalized", s_min, s_max, np.min(y_train), np.max(y_train))

encoder = load_model('./../data/nns/encoder.h5')
decoder = load_model('./../data/nns/decoder.h5')

class Config:
    
    def __init__(self):
        self.Filters = [256, 128, 64]
        self.genFilters = [128, 128, 128]
        self.upFilters = [(2, 2), (2, 2), (2, 2)]
        
config = Config()

# Generator
z = Input(shape = (128, ))
y = Input(shape = (1, ))

h = Concatenate(axis = 1)([z, y])
h = Dense(1 * 1 * 128)(h)
R1 = Reshape([1, 1, 128])(h)
R2 = Reshape([1, 1, 128])(h)

for i in range(3):
    R1 = UpSampling2D(size = config.upFilters[i])(R1)
    C1 = Conv2D(filters = config.genFilters[i], 
               kernel_size = 2, 
               strides = 1, 
               padding = 'same')(R1)
    B1 = BatchNormalization()(C1)
    R1 = LeakyReLU(alpha = 0.2)(B1)

for i in range(3):
    R2 = UpSampling2D(size = config.upFilters[i])(R2)
    C2 = Conv2D(filters = config.genFilters[i], 
               kernel_size = 2, 
               strides = 1, 
               padding = 'same')(R2)
    B2 = BatchNormalization()(C2)
    R2 = LeakyReLU(alpha = 0.2)(B2)
    
R1 = Conv2D(1,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'tanh')(R1)
R2 = Conv2D(1,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'tanh')(R2)

generator = Model([z, y], [R1, R2])
print (generator.summary())
# Discriminator
#X = Input(shape = [6,6,2 ])
inp1 = Input(shape = [6, 6, 1])
inp2 = Input(shape = [6, 6, 1])

X1 = Concatenate()([inp1, inp2])
X = Flatten()(X1)
print (X.shape)
y2 = Concatenate(axis = 1)([X, y])
print (y2.shape)
for i in range(3):
		y2 = Dense(64, activation = 'relu')(y2)
		y2 = LeakyReLU(alpha = 0.2)(y2)
		y2 = Dropout(0.2)(y2)

O_dis = Dense(1, activation = 'sigmoid')(y2)


discriminator = Model([inp1, inp2, y], O_dis)
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 5e-6, beta_1 = 0.5))
print (discriminator.summary()) 
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
print (regressor.summary())
# combined
def build_combined(z, y,
                   regressor,
                   regressor_top,
                   discriminator):
    atoms_emb, bonds_emb = generator([z, y])

    
    discriminator.trainable = False
    regressor_top.trainable = False
    regressor.trainable = False

    y_pred = regressor([atoms_emb, bonds_emb])
    #latent = regressor_top([atoms_emb, bonds_emb])
    
    valid = discriminator([atoms_emb, bonds_emb, y])

    combined = Model([z, y], [valid, y_pred])

    combined.compile(loss = ['binary_crossentropy',
                             'mse'], 
                     loss_weights = [1.0, 25.0], 
                     optimizer = Adam(5e-6, beta_1 = 0.5))

    return combined

combined = build_combined(z, y,
                          regressor,
                          regressor_top,
                          discriminator)

train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])

atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
atoms_val, bonds_val, _ = encoder.predict([X_atoms_val, X_bonds_val])

regressor = load_model('./../data/nns/regressor.h5')
regressor_top = load_model('./../data/nns/regressor_top.h5')

regressor.fit([atoms_embedding, bonds_embedding], 
              y_train,
              validation_data = ([atoms_val,
                                  bonds_val],
                                 y_val),
              batch_size = 32,
              epochs = 1,
              verbose = 1)

# Validating the regressor
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for train data: {}'.format(r2_score(y_train, pred.reshape([-1]))))
print (pred)
print (y_train)
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_val, X_bonds_val])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for validation data: {}'.format(r2_score(y_val, pred.reshape([-1]))))
print ("pred of validation data: ", pred )
print ("True validation values: ", y_val)
# Saving the currently trained models
regressor.save('./../data/nns/regressor.h5')
regressor_top.save('./../data/nns/regressor_top.h5')

regressor = load_model('./../data/nns/regressor.h5')
regressor_top = load_model('./../data/nns/regressor_top.h5')
#generator = load_model    ('./../data/nns/generator.h5')
#discriminator= load_model ('./../data/nns/discriminator.h5')

regressor_top.trainable = False
regressor.trainable = False

epochs = 200
batch_size = 128
threshold = 0.2
# number of fake indices feedback 5or50 
reinforce_n = 5
# number of samples picked for reinforcement, 100or1000
reinforce_sample = 1000

batches = y_train.shape[0] // batch_size

G_Losses = []
D_Losses = []
R_Losses = []
D_Losses_real = []
D_Losses_fake = []

for e in range(epochs):
    start = time.time()
    D_loss = []
    G_loss = []
    R_loss = []
    D_loss_real = []
    D_loss_fake = []
    for b in range(batches):
        
        regressor_top.trainable = False
        regressor.trainable = False

        idx = np.arange(b * batch_size, (b + 1) * batch_size)
        # Subsample started for reinforcements
        idx = np.random.choice(idx, batch_size, replace = False)
        
        atoms_train = X_atoms_train[idx]
        bonds_train = X_bonds_train[idx]
        batch_y = y_train[idx]
        # !!!!!!!!!! SD should be 1
        batch_z = np.random.normal(0, 1, size = (batch_size, 128))
        
        atoms_embedding, bonds_embedding, _ = encoder.predict([atoms_train, bonds_train])
        dec_embedding = np.concatenate([atoms_embedding, bonds_embedding], axis = -1)
		
        smiles = decoder.predict(dec_embedding)[0]
        smiles = np.argmax(smiles, axis = 2).reshape([-1])
        c_smiles = ''
        for s in smiles:
            c_smiles += tokenizer[s]
            c_smiles = c_smiles.rstrip()
        
        #print (c_smiles)
        #print (X_smiles_train[idx],"!!!!!!!!!!!!1")
        gen_atoms_embedding, gen_bonds_embedding = generator.predict([batch_z, batch_y])
        
        r_loss = regressor.train_on_batch([atoms_embedding, bonds_embedding], batch_y)
        R_loss.append(r_loss)
        
        real_latent = regressor_top.predict([atoms_embedding, bonds_embedding])
        fake_latent = regressor_top.predict([gen_atoms_embedding, gen_bonds_embedding])
        
        discriminator.trainable = True
        for _ in range(3):
            d_loss_real = discriminator.train_on_batch([atoms_embedding,bonds_embedding, batch_y],
                                                       [0.9 * np.ones((batch_size, 1))])
            d_loss_fake = discriminator.train_on_batch([gen_atoms_embedding,gen_bonds_embedding, batch_y],
                                                       [np.zeros((batch_size, 1))])

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        D_loss.append(d_loss)
        
        D_loss_real.append (d_loss_real)
        D_loss_fake.append (d_loss_fake)
        
        discriminator.trainable = False
        regressor_top.trainable = False
        regressor.trainable = False

        g_loss = combined.train_on_batch([batch_z, batch_y], [0.9 * np.ones((batch_size, 1)), batch_y])
        G_loss.append(g_loss)
    
    D_Losses.append(np.mean(D_loss))
    D_Losses_real.append(np.mean(D_loss_real))
    D_Losses_fake.append(np.mean(D_loss_fake))
    G_Losses.append(np.mean(G_loss))
    R_Losses.append(np.mean(R_loss))
    
    print('====')
    print('Current epoch: {}/{}'.format((e + 1), epochs))
    print ('D Loss Real: {}'.format(np.mean(D_loss_real)))
    print ('D Loss Fake: {}'.format(np.mean(D_loss_fake)))
    print('D Loss: {}'.format(np.mean(D_loss)))
    print('G Loss: {}'.format(np.mean(G_loss)))
    print('R Loss: {}'.format(np.mean(R_loss)))
    print('====')
    print()

    
    # Reinforcement
    gen_error = []
    gen_smiles = []
    embeddings = []
    sample_ys = []
    for _ in range(reinforce_sample):
        sample_y = np.random.uniform(s_min, s_max, size = [1,])
        sample_y = np.round(sample_y, 4)
        sample_y = (sample_y - s_min) / (s_max - s_min)
        sample_ys.append(sample_y)

        # SD_original should be 1
        sample_z = np.random.normal(0, 1, size = (1, 128))

        sample_atoms_embedding, sample_bonds_embedding = generator.predict([sample_z, sample_y])
        embeddings.append((sample_atoms_embedding,
                           sample_bonds_embedding))
        
        reg_pred = regressor.predict([sample_atoms_embedding, sample_bonds_embedding])
        
        pred, desire = reg_pred[0][0], sample_y[0]
        gen_error.append(np.abs((pred - desire) / desire))

        dec_embedding = np.concatenate([sample_atoms_embedding, sample_bonds_embedding], axis = -1)
        smiles = decoder.predict(dec_embedding)[0]
        smiles = np.argmax(smiles, axis = 2).reshape([-1])
        c_smiles = ''
        for s in smiles:
            c_smiles += tokenizer[s]
        c_smiles = c_smiles.rstrip()
        if _==0:
            print (smiles)
            print (c_smiles)
        gen_smiles.append(c_smiles)
        
    gen_error = np.asarray(gen_error)
        
    valid = 0
    valid0 = 0
    idx_ = []
    idx0_ = []
    for iter_, smiles in enumerate(gen_smiles):
        if ' ' in smiles[:-1]:
            continue
        m = Chem.MolFromSmiles(smiles[:-1],sanitize=True)
        m0 = Chem.MolFromSmiles(smiles[:-1],sanitize=False)
        if m0 is not None:
            valid0 += 1
            idx0_.append(iter_)
        if m is not None:
            valid += 1
            idx_.append(iter_)
            try:
                gen_smiles [iter_] = Chem.MolToSmiles(m)
                print (Chem.MolToSmiles(m))
                print ("Hc_des", sample_ys[iter_]) 
                print ("error", gen_error[iter_])
            except:
                pass
    idx_ = np.asarray(idx_)
    idx0_ = np.asarray(idx0_)

    validity = [gen_smiles[jj] for jj in idx0_ ]
    validity = pd.DataFrame(validity)
    validity = validity.drop_duplicates()

    validity_sanitize = [gen_smiles[jj] for jj in idx_ ]
    validity_sanitize = pd.DataFrame(validity_sanitize)
    validity_sanitize = validity_sanitize.drop_duplicates()
 
    if (e + 1) % 100 == 0:
        reinforce_n += 10
    
    # invalid smiles:
    fake_indices = np.setdiff1d(np.arange(reinforce_sample), np.asarray(idx_))
    fake_indices = np.random.choice(fake_indices, reinforce_n * 5, replace = False)
    
    real_indices_ = np.intersect1d(np.where(gen_error < threshold)[0], idx_)
    sample_size = min(reinforce_n, len(real_indices_))
    real_indices = np.random.choice(real_indices_, sample_size, replace = False)
    
    if e >= 0:
        discriminator.trainable = True
        regressor_top.trainable = False
        regressor.trainable = False
        for real_index in real_indices:
            #real_latent = regressor_top.predict([embeddings[real_index][0], embeddings[real_index][1]])
            _ = discriminator.train_on_batch([embeddings[real_index][0], embeddings[real_index][1], sample_ys[real_index]],
                                             [0.9 * np.ones((1, 1))])

        for fake_index in fake_indices:
            #fake_latent = regressor_top.predict([embeddings[fake_index][0], embeddings[fake_index][1]])
            _ = discriminator.train_on_batch([embeddings[fake_index][0], embeddings[fake_index][1] , sample_ys[fake_index]],
                                             [np.zeros((1, 1))])
        discriminator.trainable = False
        
    # ==== #
    try:    
        print('Currently valid SMILES (No chemical_beauty and sanitize off): {}'.format(valid0))
        print('Currently valid SMILES Unique (No chemical_beauty and sanitize off): {}'.format(len(validity)))
        print('Currently valid SMILES Sanitized: {}'.format(valid))
        print('Currently valid Unique SMILES Sanitized: {}'.format(len(validity_sanitize)))
        print('Currently satisfying SMILES: {}'.format(len(real_indices_)))
        print('Currently unique satisfying generation: {}'.format(len(np.unique(np.array(gen_smiles)[real_indices_]))))
        print('Gen Sample is: {}, for {}'.format(c_smiles, sample_y))
        print('Predicted val: {}'.format(reg_pred))
        print('====')
        print()
    except:
        pass

    if (e + 1) % 5 == 0:
        plt.plot(G_Losses)
        plt.plot(D_Losses)
        #plt.plot(R_Losses)
        plt.legend(['G Loss', 'D Loss'])
        plt.savefig("G_D_R_losses_{}.png".format (e+1))
    n_unique = len(np.unique(np.array(gen_smiles)[real_indices_]))
    n_valid = valid
    if valid > 450 and n_unique > 350:
        print('Criteria has satisified, training has ended')
        break
    
    end = time.time()
    print ("time for current epoch: ", (end - start))
with open('GAN_loss.pickle', 'wb') as f:
    pickle.dump((G_Losses, D_Losses, R_Losses), f)

# Saving the currently trained models
#regressor.save('regressor.h5')
#regressor_top.save('regressor_top.h5')
generator.save('./../data/nns/generator.h5')
discriminator.save('./../data/nns/discriminator.h5')

##====#

# Generation Study

#regressor = load_model('regressor.h5')
#regressor_top = load_model('regressor_top.h5')
generator = load_model    ('./../data/nns/generator.h5')
discriminator = load_model('./../data/nns/discriminator.h5')

encoder = load_model('./../data/nns/encoder.h5')
decoder = load_model('./../data/nns/decoder.h5')

# Generation workflow
# 1. Given a desired heat capacity
# 2. Generate 10,000 samples of SMILES embedding
# 3. Select the ones with small relative errors (< 10%)
# 4. Transfer them to SMILES
# 5. Filter out the invalid SMILES

# Generate 500 different values of heat capacities

from progressbar import ProgressBar

N = 10000
n_sample = 50

gen_error = []
gen_smiles = []
sample_ys = []
preds = []
gen_atoms_embedding = []
gen_bonds_embedding = []

regressor_top.trainable = False
regressor.trainable = False
generator.trainable = False
discriminator.trainable = False

pbar = ProgressBar()
for hc in pbar(range(n_sample)):
    try:
        # get it back to original of s_min to s_max
        sample_y = np.random.uniform(s_min, s_max, size = [1,])
        #X = get_truncated_normal(mean=30, sd=5, low=s_min, upp=s_max)
        #sample_y = X.rvs()
        print (sample_y)
        sample_y = np.round(sample_y, 3)
        sample_y = sample_y * np.ones([N,])
        sample_y_ = (sample_y - s_min) / (s_max - s_min)
        # !!!!!!!!! SD_origianl = 1
        sample_z = np.random.normal(0, 1, size = (N, 128))
        
        regressor_top.trainable = False
        regressor.trainable = False

        sample_atoms_embedding, sample_bonds_embedding = generator.predict([sample_z, sample_y_])
        pred = regressor.predict([sample_atoms_embedding, sample_bonds_embedding]).reshape([-1])
        pred = pred * (s_max - s_min) + s_min

        gen_errors = np.abs((pred - sample_y) / sample_y).reshape([-1])

        accurate = np.where(gen_errors < 0.1)[0]
        gen_errors = gen_errors[accurate]
        pred = pred[accurate]

        sample_y = sample_y[accurate]
        sample_atoms_embedding = sample_atoms_embedding[accurate]
        sample_bonds_embedding = sample_bonds_embedding[accurate]

        dec_embedding = np.concatenate([sample_atoms_embedding, sample_bonds_embedding], axis = -1)
        smiles = decoder.predict(dec_embedding)[0]
        smiles = np.argmax(smiles, axis = 2).reshape(smiles.shape[0], 35)

        generated_smiles = []
        for S in smiles:
            c_smiles = ''
            for s in S:
                c_smiles += tokenizer[s]
            c_smiles = c_smiles.rstrip()
            generated_smiles.append(c_smiles)
        generated_smiles = np.array(generated_smiles)

        all_gen_smiles = []
        idx = []
        for i, smiles in enumerate(generated_smiles):
            all_gen_smiles.append(smiles[:-1])

            if ' ' in smiles[:-1]:
                continue
            #m = Chem.MolFromSmiles(smiles[:-1],sanitize=False)
            m = Chem.MolFromSmiles(smiles[:-1])
            if m is not None:
                idx.append(i)

        idx = np.array(idx)
        all_gen_smiles = np.array(all_gen_smiles)

        gen_smiles.extend(list(all_gen_smiles[idx]))
        gen_error.extend(list(gen_errors[idx]))
        sample_ys.extend(list(sample_y[idx]))
        gen_atoms_embedding.extend(sample_atoms_embedding[idx])
        gen_bonds_embedding.extend(sample_bonds_embedding[idx])

        preds.extend(list(pred[idx]))
    except:
        print('Did not discover SMILES for HC: {}'.format(sample_y))
        pass
    
output = {}

for i, s in enumerate (gen_smiles):
    ss = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    gen_smiles[i] = ss

output['SMILES'] = gen_smiles
output['des_cv'] = sample_ys
output['pred_cv'] = preds
output['Err_pred_des'] = gen_error

plt.close()
plt.hist(gen_error)
plt.savefig("gen_error_hist.png")


## Statistics  (# DFT=True value, Des=prediction)

# total # of samples
N = len(gen_error)
# Explained Variance R2 from sklearn.metrics.explained_variance_score
explained_variance_R2_DFT_des = explained_variance_score(output['pred_cv'], output['des_cv'])
print ("explained_varice_R2_DFT_des", explained_variance_R2_DFT_des)

# mean absolute error 
MAE_DFT_des = mean_absolute_error(output['pred_cv'], output['des_cv'])
print ("MAE_DFT_des", MAE_DFT_des)
# Fractioned MAE, more normalized
Fractioned_MAE_DFT_des = 0
for dft, des in zip(output['pred_cv'], output['des_cv']):
    Fractioned_MAE_DFT_des = Fractioned_MAE_DFT_des +  abs(des-dft)/des
Fractioned_MAE_DFT_des = Fractioned_MAE_DFT_des/N
print ("Fractioned MAE_DFT_des", Fractioned_MAE_DFT_des)

# root mean squared error (RMSE), sqrt(sklearn ouputs MSE)
RMSE_DFT_des = mean_squared_error(output['pred_cv'], output['des_cv'])**0.5
print ("RMSE_DFT_des", RMSE_DFT_des)

Fractioned_RMSE_DFT_des = 0
for dft, des in zip(output['pred_cv'], output['des_cv']):
    Fractioned_RMSE_DFT_des = Fractioned_RMSE_DFT_des + ((des-dft)/des)**2
Fractioned_RMSE_DFT_des = (Fractioned_RMSE_DFT_des/N)**0.5
print ("Fractioned_RMSE_DFT_des", Fractioned_RMSE_DFT_des)

output = pd.DataFrame(output)
# do not drop duplicate
output2 = output.drop_duplicates(['SMILES'])
gen_atoms_embedding = np.array(gen_atoms_embedding)
gen_bonds_embedding = np.array(train_atoms_embedding)

"""
# ANALYSIS
X_atoms_train_ = train_atoms_embedding.reshape([train_atoms_embedding.shape[0], 
                                        6 * 6])
X_bonds_train_ = train_bonds_embedding.reshape([train_bonds_embedding.shape[0], 
                                        6 * 6])

X_atoms_test_ = gen_atoms_embedding.reshape([gen_atoms_embedding.shape[0],
                                      6 * 6])
X_bonds_test_ = gen_bonds_embedding.reshape([gen_bonds_embedding.shape[0], 
                                      6 * 6])

pca_1 = PCA(n_components = 2)
X_atoms_train_ = pca_1.fit_transform(X_atoms_train_)
X_atoms_test_ = pca_1.transform(X_atoms_test_)

pca_2 = PCA(n_components = 2)
X_bonds_train_ = pca_2.fit_transform(X_bonds_train_)
X_bonds_test_ = pca_2.transform(X_bonds_test_)

# Atoms Distribution
plt.close()
plt.scatter(X_atoms_train_[:,0], X_atoms_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("train_atom_dist.png")
#plt.close()
plt.scatter(X_atoms_test_[:,0], X_atoms_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("test_atom_dist.png")
####

# Bonds Distribution
plt.close()
plt.scatter(X_bonds_train_[:,0], X_bonds_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("train_bonds_dist.png")
#plt.close()
plt.scatter(X_bonds_test_[:,0], X_bonds_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("test_bonds_dist.png")
# 31/500 failed (N = 10000)
# 2/50 failed (N = 50000)
"""
output.reset_index(drop = True, inplace = True)
output2.reset_index(drop = True, inplace = True)
output.to_csv ('./../experiments/regular/Regular_500ep.csv', index = False)
output2.to_csv('./../experiments/regular/Regular_NODUP_500ep.csv', index = False)
"""with open('gen_pickles.pickle', 'wb') as f:
    pickle.dump(gen_unique_pickles, f)
"""
