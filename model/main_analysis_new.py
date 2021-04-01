import warnings
warnings.filterwarnings('ignore')

print ("!!!!!!!!! we are just before importing rdkit!!!!!")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")

from thermo.joback import Joback

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor

# load the generated smiles from the RCGAN Model
gen_SMILES = pd.read_csv('generated_SMILES4.csv')

jobacks = []
validated = []
for s in gen_SMILES['SMILES'].values:
    try:
        J = Joback(s)
        jobacks.append(J.Cpig(298.15) * 0.2390057361)
        validated.append(s)
    except:
        pass

print ("length of validated smiles by Joback {} Vs. total gen_smiles {}".\
        format (len(validated), len (gen_SMILES['SMILES'])))

val = {}
val['jobacks'] = jobacks
val['SMILES'] = validated
val = pd.DataFrame(val)
print (val)
print (gen_SMILES)

val = pd.merge(val, gen_SMILES, how = 'left', on = 'SMILES')
print (val)

# error using Joback method mean and median 
# (joback vs. Desired Cv)
mean_err = np.mean(np.abs((val['Desired hc'].values - 
                           val['jobacks'].values) / 
                           val['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs. Sampled_Desired: ", mean_err)
median_err = np.median(np.abs((val['Desired hc'].values - 
                               val['jobacks'].values) / 
                               val['jobacks'].values))
print ("median error Joback(gen_SMILES) Vs. Sampled_Desired: ", median_err)

# error using Joback method mean and median 
# (Joback vs. predicted Cv by regressor)
mean_err = np.mean(np.abs((val['Predicted hc'].values - 
                           val['jobacks'].values) / 
                           val['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs. Predicted from regressor: ", mean_err)
median_err = np.median(np.abs((val['Predicted hc'].values - 
                               val['jobacks'].values) /  
                               val['jobacks'].values))
print ("median error Joback(gen_SMILES) Vs. Predicted from regressor: ", median_err)

# find the best candidates in generated smiles (criteria: <0.05)
val_accurate = pd.DataFrame({'SMILES': [],
                             'Desired hc': [],
                             'Predicted hc': [],
                             'jobacks': []})
accurate = []
print (val_accurate)
"""
for i, s in enumerate (val['SMILES'].values):
    if (np.abs((val['Desired hc'].values[i] - val['jobacks'].values[i]) / val['jobacks'].values[i]) < 0.07 and
        np.abs((val['Desired hc'].values[i] - val['jobacks'].values[i]) / val['jobacks'].values[i]) > 0.03 ) :
        accurate.append(i)
print (accurate)
"""
for i, s in enumerate (val['SMILES'].values):
    if np.abs((val['Desired hc'].values[i] - val['jobacks'].values[i]) / val['jobacks'].values[i]) < 0.15:
        accurate.append(i)
print (accurate)

for ii, a in enumerate (accurate):
    #print (" i and a from accurate",ii, a)
    val_accurate.loc[ii,:] = val.iloc[a,:]
print (val_accurate)
print ("the first smile in val_accurate: ",val_accurate['SMILES'].values[0])
for i, s in enumerate (val_accurate['SMILES'].values):
    print (s)
    m = Chem.MolFromSmiles(s)
    ss = Chem.MolToSmiles(m)
    val_accurate['SMILES'].values[i] = ss
    print (val_accurate['SMILES'].values[i])
    print (ss)
print (val_accurate)

sort_val_accurate = val_accurate.sort_values ('Desired hc')
print (sort_val_accurate) 

# accuracy of the the model Joback vs. predicted and desired Cv (accurate < 5%)
mean_err = np.mean(np.abs((val_accurate['Predicted hc'].values - 
                           val_accurate['jobacks'].values) / 
                           val_accurate['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs.Predicted from regressor (for accurate Cv(<5%): ", mean_err)

median_err = np.median(np.abs((val_accurate['Predicted hc'].values - 
                               val_accurate['jobacks'].values) / 
                               val_accurate['jobacks'].values))
print ("median error Joback(gen_SMILES) Vs.Predicted from regressor(for accurate Cv(<5%) : ", median_err)

mean_err = np.mean(np.abs((val_accurate['Desired hc'].values -
                           val_accurate['jobacks'].values) /
                           val_accurate['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs.Desired from regressor (for accurate Cv(<5%): ", mean_err)

median_err = np.median(np.abs((val_accurate['Desired hc'].values -
                               val_accurate['jobacks'].values) /
                               val_accurate['jobacks'].values))
print ("median error Joback(gen_SMILES) Vs.Desired from regressor (for accurate Cv(<5%) : ", median_err)

num_acc_l0p1 = np.sum(np.abs((val['Desired hc'].values - 
                               val['jobacks'].values) / 
                               val['jobacks'].values) < 0.1)

plt.scatter(val['Desired hc'].values, val['jobacks'].values)
plt.savefig("Desired_VS_joback.png")

plt.clf()
plt.scatter(val_accurate['Desired hc'].values, val_accurate['jobacks'].values)
plt.title("Accurate Desired Cv vs. Joback Cv")
plt.xlabel("Desired Cv")
plt.ylabel("Joback Cv")
plt.savefig("Desired_accurate_VS_joback.png")

plt.clf()
#sns.distplot(np.abs((val['Desired hc'].values - val['jobacks'].values) / val['jobacks'].values))
sns.distplot(((val['Desired hc'].values - val['jobacks'].values) / val['jobacks'].values))

plt.savefig("err_Des_VS_Jobsck.png")

plt.clf()
err_Desacc_job = ((val_accurate['Desired hc'].values - val_accurate['jobacks'].values) / val_accurate['jobacks'].values)
err_Desacc_job = pd.Series(err_Desacc_job, name="err_Des_accurate_VS_Jobsck")
sns.distplot(err_Desacc_job)
plt.savefig("err_Des_accurate_VS_Jobsck.png")
#1131/3020
num_acc_l0p025 = np.sum(gen_SMILES['Error'].values < 0.025, dtype = np.int32)
num_acc_0p025_0p05 =  np.sum((gen_SMILES['Error'].values >= 0.025) & (gen_SMILES['Error'].values < 0.05), dtype = np.int32)
num_acc_g0p05 =  np.sum(gen_SMILES['Error'].values > 0.05, dtype = np.int32)
total = num_acc_l0p025 + num_acc_0p025_0p05 + num_acc_g0p05

print ("type of accurate l0p025: ", type (num_acc_l0p025))
print (num_acc_l0p025)

summ = float(gen_SMILES.shape[0])
print ("total SMILES is : {}".format(gen_SMILES.shape))
plt.figure(figsize = (5, 5))

plt.bar(['< 2.5%', '2.5% - 5%', '5% - 10%'],
        [num_acc_l0p025 / total, num_acc_0p025_0p05 / total,num_acc_g0p05 / total],
        color = ['green', 'blue', 'red'],
        alpha = 0.7)
plt.grid()
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig("error_Distrib_5_2p5percent.png")

val_accurate.reset_index(drop = True, inplace = True)

val_accurate.to_csv('gen_SMILES2_accur_joback.csv', index = False)

preprocessor = GGNNPreprocessor()
data = get_molnet_dataset('qm9',
                          labels = 'cv',
                          preprocessor = preprocessor,
                          return_smiles = True,
                          frac_train = 1.0,
                          frac_valid = 0.0,
                          frac_test = 0.0
                         )

smiles = data['smiles'][0]
"""
smiles_ = []
for s in smiles_:
    print (s)
    m = Chem.MolFromSmiles (s)
    ss = Chem.MolToSmiles(m)
    smiles.append(ss)
    print (ss)
"""
smiles = smiles.astype('str')

data_smiles = []
data_cv = []
for i, s in enumerate(smiles):
    data_smiles.append(s)
    data_cv.append(data['dataset'][0][i][2][0])

jobacks = []
validated = []
for i, s in enumerate(data_smiles):
    try:
        J = Joback(s)
        jobacks.append(J.Cpig(298.15))
        validated.append(i)
    except:
        pass

data_cv = np.array(data_cv)[validated]
data_smiles = np.array(data_smiles)

jobacks = np.array(jobacks)

qm9_joback_Mean_relerr = np.mean(np.abs((data_cv - jobacks * 0.2390057361) / (jobacks * 0.2390057361)))
# 6.0% difference between qm9 cv and joback's cv
print ("qm9_joback_Mean_relerr is :{}".format(qm9_joback_Mean_relerr))
gen_unique = gen_SMILES['SMILES'].values

data = {}
data['SMILES'] = data_smiles[validated]
data['cv'] = data_cv
data = pd.DataFrame(data)

print ("qm9 smiles that Joback could analyze and get the Cv: ", data)

database_samples = pd.merge(gen_SMILES, data, on = 'SMILES', how = 'inner')

print ( "same generated smiles compared to qm9 lib is:{}".format(database_samples))

database_samples_accurate = pd.merge(val_accurate, data, on = 'SMILES', how = 'inner')

print ( "same generated accurate smiles compared to qm9 lib is:{}".format(database_samples_accurate))
# find the repetitive smiles in qm9
rep_index = []
for i, smil in enumerate (database_samples_accurate['SMILES'].values):
    for j, smi in enumerate (val_accurate ['SMILES'].values):
         if (smi == smil):
             rep_index.append(j)
             #val_accurate = val_accurate.drop([j])

print (rep_index)
print ("from {} valid, unique, satisfying (regression model), and Joback accurate \
       (less than 5% err), {} are not in qm9"\
       .format (val_accurate.shape[0], (val_accurate.shape[0] - len(rep_index))))

print ("rep_index", len(rep_index))
val_accurate = val_accurate.drop(rep_index)

val_accurate.reset_index(drop = True, inplace = True)

val_accurate.to_csv('gen_SMILES4_accurjoback_qm9reprem.csv', index = False)


# get the relative error of Desired Cv vs Cv from qm9 for generated smiles that are repetitive and are in qm9
mean_rel_diff_desired_cvqm9 = np.mean(np.abs((database_samples['Desired hc'].values -
                                              database_samples['cv'].values) / 
                                              database_samples['cv'].values))
print ("mean of rel diff BW Desired (sampled in design model) and cv from qm9: {}".
                                                 format(mean_rel_diff_desired_cvqm9))

median_rel_diff_desired_cvqm9 = np.median(np.abs((database_samples['Desired hc'].values
                       - database_samples['cv'].values) / database_samples['cv'].values))
print ("median of rel diff BW Desired (sampled in design model) and cv from qm9: {}".
                                                 format(median_rel_diff_desired_cvqm9))

mean_rel_diff_desired_cvqm9 = np.mean(np.abs((database_samples['Predicted hc'].values
                       - database_samples['cv'].values) / database_samples['cv'].values))
print ("mean of rel diff BW Predicted (from regresor) and cv from qm9: {}".
                                                 format(mean_rel_diff_desired_cvqm9))

median_rel_diff_desired_cvqm9 = np.median(np.abs((database_samples['Predicted hc'].values
                       - database_samples['cv'].values) / database_samples['cv'].values))
print ("median of rel diff BW Predicted (from regressor) and cv from qm9: {}".
                                                 format(median_rel_diff_desired_cvqm9))

# get the relative error of Desired Cv vs Cv from qm9 for generated smiles (Joback accuracy < 5%err) that are repetitive and are in qm9
mean_rel_diff_desired_cvqm9 = np.mean(np.abs((database_samples_accurate['Desired hc'].values -
                                              database_samples_accurate['cv'].values) /
                                              database_samples_accurate['cv'].values))
print ("mean of rel diff BW Desired (Joback Accurate) and cv from qm9: {}".
                                                 format(mean_rel_diff_desired_cvqm9))

median_rel_diff_desired_cvqm9 = np.median(np.abs((database_samples_accurate['Desired hc'].values -
                                                  database_samples_accurate['cv'].values) / 
                                                  database_samples_accurate['cv'].values))
print ("median of rel diff BW Desired (Joback Accurate) and cv from qm9: {}".
                                                 format(median_rel_diff_desired_cvqm9))

mean_rel_diff_desired_cvqm9 = np.mean(np.abs((database_samples_accurate['Predicted hc'].values -
                                              database_samples_accurate['cv'].values) / 
                                              database_samples_accurate['cv'].values))
print ("mean of rel diff BW Predicted (from regresor Joback accurate) and cv from qm9: {}".
                                                 format(mean_rel_diff_desired_cvqm9))

median_rel_diff_desired_cvqm9 = np.median(np.abs((database_samples_accurate['Predicted hc'].values - 
                                                  database_samples_accurate['cv'].values) / 
                                                  database_samples_accurate['cv'].values))
print ("median of rel diff BW Predicted (from regressor Joback accurate) and cv from qm9: {}".
                                                 format(median_rel_diff_desired_cvqm9))

"""
sns.distplot(np.abs((database_samples['Desired hc'].values - database_samples['cv'].values) / database_samples['cv'].values))

ins = 0

for G in gen_unique:
    # smile = G[:-1]
    if G in smiles:
        ins += 1

ins, len(gen_unique), ins / len(gen_unique)

smiles = smiles.astype('str')
gen_unique = gen_unique.astype('str')

print ("len (smiles): {} , len (gen_unique): {}".format(len(smiles), len(gen_unique)))

X_smiles = np.concatenate((smiles, gen_unique))

MAX_NB_WORDS = 23
MAX_SEQUENCE_LENGTH = 35

tokenizer = Tokenizer(num_words = MAX_NB_WORDS,
                      char_level = True,
                      filters = '',
                      lower = False)
tokenizer.fit_on_texts(X_smiles)

X_smiles = tokenizer.texts_to_sequences(X_smiles)
X_smiles = pad_sequences(X_smiles,
                         maxlen = MAX_SEQUENCE_LENGTH,
                         padding = 'post')

X_smiles = to_categorical(X_smiles)

X_train = X_smiles[:-2710]
X_gen = X_smiles[-2710:]

X_train = X_train.reshape([X_train.shape[0], 35 * 23])
X_gen = X_gen.reshape([X_gen.shape[0], 35 * 23])

pca = PCA(n_components = 2)
X_train_ = pca.fit_transform(X_train)
X_gen_ = pca.transform(X_gen)

# Atoms Distribution
plt.scatter(X_train_[:,0], X_train_[:,1], alpha = 0.3, c = 'blue');
plt.scatter(X_gen_[:,0], X_gen_[:,1], alpha = 0.3, c = 'red');
plt.grid()
plt.savefig ("atoms_Distri.png")
"""
