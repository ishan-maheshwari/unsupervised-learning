# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import myGMM,nn_arch,nn_lr, run_NN, appendClusterDimKM, appendClusterDimGMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from helpers import pairwiseDistCorr,reconstructionError, ImportanceSelect
from sklearn.random_projection import SparseRandomProjection
from itertools import product
from sklearn.ensemble import RandomForestClassifier

#%% Setup
np.random.seed(0)
clusters =  [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,25,30]
dims_spam = [2,5,10,15,20,25,30,35,40,45,50,55,57]
dims_letter = range(2,17)

# load spam training set      
spam = pd.read_hdf('datasets.hdf','spam')
spamX = spam.drop('Y',1).copy().values
spamY = spam['Y'].copy().values
le = preprocessing.LabelEncoder()
spamY = le.fit_transform(spamY)

# load letter training set  
letter = pd.read_hdf('datasets.hdf','letter_original')     
letterX = letter.drop('Y',1).copy().values
letterY = letter['Y'].copy().values
le = preprocessing.LabelEncoder()
letterY = le.fit_transform(letterY)

spamX = StandardScaler().fit_transform(spamX)
letterX= StandardScaler().fit_transform(letterX)

#%% Part 1 - Run k-means and EM clustering algorithms on original datasets

print('Part 1 - Running clustering algoirthms on original datasets...')
SSE = defaultdict(dict)
BIC = defaultdict(dict)
homo = defaultdict(lambda: defaultdict(dict))
compl = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(spamX)
    gmm.fit(spamX)
    SSE[k]['spam SSE'] = km.score(spamX)
    BIC[k]['spam BIC'] = gmm.bic(spamX)
    homo[k]['spam']['Kmeans'] = homogeneity_score(spamY,km.predict(spamX))
    homo[k]['spam']['GMM'] = homogeneity_score(spamY,gmm.predict(spamX))
    compl[k]['spam']['Kmeans'] = completeness_score(spamY,km.predict(spamX))
    compl[k]['spam']['GMM'] = completeness_score(spamY,gmm.predict(spamX))
    adjMI[k]['spam']['Kmeans'] = ami(spamY,km.predict(spamX))
    adjMI[k]['spam']['GMM'] = ami(spamY,gmm.predict(spamX))
    
    km.fit(letterX)
    gmm.fit(letterX)
    SSE[k]['letter'] = km.score(letterX)
    BIC[k]['letter BIC'] = gmm.bic(letterX)
    homo[k]['letter']['Kmeans'] = homogeneity_score(letterY,km.predict(letterX))
    homo[k]['letter']['GMM'] = homogeneity_score(letterY,gmm.predict(letterX))
    compl[k]['letter']['Kmeans'] = completeness_score(letterY,km.predict(letterX))
    compl[k]['letter']['GMM'] = completeness_score(letterY,gmm.predict(letterX))
    adjMI[k]['letter']['Kmeans'] = ami(letterY,km.predict(letterX))
    adjMI[k]['letter']['GMM'] = ami(letterY,gmm.predict(letterX))
    print(k, clock()-st)
    
    
SSE = (-pd.DataFrame(SSE)).T
#SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
BIC = pd.DataFrame(BIC).T
#ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
homo = pd.Panel(homo)
compl = pd.Panel(compl)
adjMI = pd.Panel(adjMI)


SSE.to_csv('./P1_Clustering_Algorithms_Original/Cluster_Select_Kmeans.csv')
BIC.to_csv('./P1_Clustering_Algorithms_Original/Cluster_Select_GMM.csv')
homo.ix[:,:,'letter'].to_csv('./P1_Clustering_Algorithms_Original/letter_homo.csv')
homo.ix[:,:,'spam'].to_csv('./P1_Clustering_Algorithms_Original/spam_homo.csv')
compl.ix[:,:,'letter'].to_csv('./P1_Clustering_Algorithms_Original/letter_compl.csv')
compl.ix[:,:,'spam'].to_csv('./P1_Clustering_Algorithms_Original/spam_compl.csv')
adjMI.ix[:,:,'letter'].to_csv('./P1_Clustering_Algorithms_Original/letter_adjMI.csv')
adjMI.ix[:,:,'spam'].to_csv('./P1_Clustering_Algorithms_Original/spam_adjMI.csv')

#%% Part 2A & 4A - Run Dimensionality Reduction Algorithm PCA, Run NN with reduced dims

print('Part 2A - Starting PCA for spam dataset...')
pca = PCA(random_state=5)
pca.fit(spamX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,58))
tmp.to_csv('./P2_Dimensionality_Reduction/spam_PCA_explained_variance_ratio.csv')

print('Part2A - Starting PCA for letter dataset...')
pca = PCA(random_state=5)
pca.fit(letterX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,17))
tmp.to_csv('./P2_Dimensionality_Reduction/letter_PCA_explained_variance_ratio.csv')

# Run Neural Networks
pca = PCA(random_state=5)  
nn_results = run_NN(dims_spam, pca, spamX, spamY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/spam_PCA_nn_results.csv')

pca = PCA(random_state=5)    
nn_results = run_NN(dims_letter, pca, letterX, letterY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/letter_PCA_nn_results.csv')

#%% Part 2B & 4B - Run Dimensionality Reduction Algorithm ICA, Run NN with reduced dims

print('Part 2B & 4B - Starting ICA for spam dataset...')
ica = FastICA(random_state=5)
kurt = {}
svm = {}
for dim in dims_spam:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(spamX)
    tmp = pd.DataFrame(tmp)
    tmp2 = tmp.kurt(axis=0)
    kurt[dim] = tmp2.abs().mean()
  
kurt = pd.Series(kurt) 
kurt.to_csv('./P2_Dimensionality_Reduction/spam_ICA_kurtosis.csv')

print('Part 2B - Starting ICA for letter dataset...')
ica = FastICA(random_state=5)
kurt = {}
for dim in dims_letter:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(letterX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv('./P2_Dimensionality_Reduction/letter_ICA_kurtosis.csv')

# Run Neural Networks
ica = FastICA(random_state=5)  
nn_results = run_NN(dims_spam, ica, spamX, spamY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/spam_ICA_nn_results.csv')

ica = FastICA(random_state=5)  
nn_results = run_NN(dims_letter, ica, letterX, letterY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/letter_ICA_nn_results.csv')

#%% Part 2C & 4C - Run Dimensionality Reduction Algorithm RP, Run NN with reduced dims

print('Part 2C - Starting RP, pairwise distance correlation, for spam dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_spam):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(spamX), spamX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/spam_RP_pairwise_distance_corr.csv')

print('Part 2C - Starting RP, pairwise distance correlation, for letter dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(1),dims_letter):
    print(dim)
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(letterX), letterX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/letter_RP_pairwise_distance_corr.csv')

print('Part 2C - Starting RP, reconstruction error, for spam dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(1),dims_spam):
    print(dim)
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(spamX)    
    tmp[dim][i] = reconstructionError(rp, spamX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/spam_RP_reconstruction_error.csv')

print('Part 2C - Starting RP, reconstruction error, for letter dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(1),dims_letter):
    print(dim)
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(letterX)  
    tmp[dim][i] = reconstructionError(rp, letterX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/letter_RP_reconstruction_error.csv')

# Run Neural Networks
rp = SparseRandomProjection(random_state=5) 
nn_results = run_NN(dims_spam, rp, spamX, spamY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/spam_RP_nn_results.csv')

rp = SparseRandomProjection(random_state=5) 
nn_results = run_NN(dims_letter, rp, letterX, letterY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/letter_RP_nn_results.csv')

#%% Part 2D & 4D - Run Dimensionality Reduction Algorithm RF, Run NN with reduced dims

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)

print('Part 2D - Starting RF for spam dataset...')
fs_spam = rfc.fit(spamX,spamY).feature_importances_ 
print('Part 2D - Starting RF for letter dataset...')
fs_letter = rfc.fit(letterX,letterY).feature_importances_ 

tmp = pd.Series(np.sort(fs_spam)[::-1])
tmp.to_csv('./P2_Dimensionality_Reduction/spam_RF_feature_importance.csv')

tmp = pd.Series(np.sort(fs_letter)[::-1])
tmp.to_csv('./P2_Dimensionality_Reduction/letter_RF_feature_importance.csv')

# Run Neural Networks
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims_spam,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}  
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
gs.fit(spamX,spamY)
nn_results = pd.DataFrame(gs.cv_results_)
nn_results.to_csv('./P4_Neural_Networks_Reduced/spam_RF_nn_results.csv')

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims_letter,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}  
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
gs.fit(letterX,letterY)
nn_results = pd.DataFrame(gs.cv_results_)
nn_results.to_csv('./P4_Neural_Networks_Reduced/letter_RF_nn_results.csv')

#%% Part 2E - Run Dimensionality Reduction Algorithms to create dimension reduced datasets

# Best number of dimensions chosen for each algorithm in Part 1 of analysis doc
dim_spam_PCA = 2 
dim_spam_ICA = 5 
dim_spam_RP = 15 
dim_spam_RF = 10 
dim_letter_PCA = 7 
dim_letter_ICA = 9 
dim_letter_RP = 11 
dim_letter_RF = 11

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)

algo_name = ['PCA', 'ICA', 'RP', 'RF']
print('Part 2E - Storing dimensionally reduced datasets for each algorithm...')
filtr = ImportanceSelect(rfc,dim_spam_RF)
algos_spam = [PCA(n_components=dim_spam_PCA,random_state=10), 
           FastICA(n_components=dim_spam_ICA,random_state=10), 
           SparseRandomProjection(n_components=dim_spam_RP,random_state=5),
           ImportanceSelect(rfc,dim_spam_RF)]

filtr = ImportanceSelect(rfc,dim_letter_RF)
algos_letter = [PCA(n_components=dim_letter_PCA,random_state=10), 
           FastICA(n_components=dim_letter_ICA,random_state=10), 
           SparseRandomProjection(n_components=dim_letter_RP,random_state=5),
           ImportanceSelect(rfc,dim_letter_RF)]

for i in range(len(algos_spam)):
    if i == 3:
        spamX2 = algos_spam[i].fit_transform(spamX, spamY)
    else:   
        spamX2 = algos_spam[i].fit_transform(spamX)
    spam2 = pd.DataFrame(np.hstack((spamX2,np.atleast_2d(spamY).T)))
    cols = list(range(spam2.shape[1]))
    cols[-1] = 'Class'
    spam2.columns = cols
    spam2.to_hdf('datasets.hdf','spam_'+algo_name[i],complib='blosc',complevel=9)

for i in range(len(algos_letter)):
    if i ==3:
        letterX2 = algos_letter[i].fit_transform(letterX, letterY)
    else:
        letterX2 = algos_letter[i].fit_transform(letterX)
    letter2 = pd.DataFrame(np.hstack((letterX2,np.atleast_2d(letterY).T)))
    cols = list(range(letter2.shape[1]))
    cols[-1] = 'Class'
    letter2.columns = cols
    letter2.to_hdf('datasets.hdf','letter_'+algo_name[i],complib='blosc',complevel=9)
#    
#%% Part 3 - Run k-means and EM clustering algorithms on each dimensionally reduced dataset

print('Part 3 - Running clustering algoirthms on dimensionally reduced datasets...')
for i in range(len(algo_name)):
    # load datasets      
    spam = pd.read_hdf('datasets.hdf','spam_'+algo_name[i]) 
    spamX = spam.drop('Class',1).copy().values
    spamY = spam['Class'].copy().values
    
    letter = pd.read_hdf('datasets.hdf','letter_'+algo_name[i])    
    letterX = letter.drop('Class',1).copy().values
    letterY = letter['Class'].copy().values
    
    spamX = StandardScaler().fit_transform(spamX)
    letterX= StandardScaler().fit_transform(letterX)
    
    SSE = defaultdict(dict)
    BIC = defaultdict(dict)
    homo = defaultdict(lambda: defaultdict(dict))
    compl = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    
    st = clock()
    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(spamX)
        gmm.fit(spamX)
        SSE[k]['spam SSE'] = km.score(spamX)
        BIC[k]['spam BIC'] = gmm.bic(spamX)
        homo[k]['spam']['Kmeans'] = homogeneity_score(spamY,km.predict(spamX))
        homo[k]['spam']['GMM'] = homogeneity_score(spamY,gmm.predict(spamX))
        compl[k]['spam']['Kmeans'] = completeness_score(spamY,km.predict(spamX))
        compl[k]['spam']['GMM'] = completeness_score(spamY,gmm.predict(spamX))
        adjMI[k]['spam']['Kmeans'] = ami(spamY,km.predict(spamX))
        adjMI[k]['spam']['GMM'] = ami(spamY,gmm.predict(spamX))
        
        km.fit(letterX)
        gmm.fit(letterX)
        SSE[k]['letter'] = km.score(letterX)
        BIC[k]['letter BIC'] = gmm.bic(letterX)
        homo[k]['letter']['Kmeans'] = homogeneity_score(letterY,km.predict(letterX))
        homo[k]['letter']['GMM'] = homogeneity_score(letterY,gmm.predict(letterX))
        compl[k]['letter']['Kmeans'] = completeness_score(letterY,km.predict(letterX))
        compl[k]['letter']['GMM'] = completeness_score(letterY,gmm.predict(letterX))
        adjMI[k]['letter']['Kmeans'] = ami(letterY,km.predict(letterX))
        adjMI[k]['letter']['GMM'] = ami(letterY,gmm.predict(letterX))
        print(k, clock()-st)
        
        
    SSE = (-pd.DataFrame(SSE)).T
    BIC = pd.DataFrame(BIC).T
    homo = pd.Panel(homo)
    compl = pd.Panel(compl)
    adjMI = pd.Panel(adjMI)
    
    SSE.to_csv('./P3_Clustering_Algorithms_Reduced/SSE_'+algo_name[i]+'.csv')
    BIC.to_csv('./P3_Clustering_Algorithms_Reduced/BIC_'+algo_name[i]+'.csv')
    homo.ix[:,:,'letter'].to_csv('./P3_Clustering_Algorithms_Reduced/letter_'+algo_name[i]+'_homo.csv')
    homo.ix[:,:,'spam'].to_csv('./P3_Clustering_Algorithms_Reduced/spam_'+algo_name[i]+'_homo.csv')
    compl.ix[:,:,'letter'].to_csv('./P3_Clustering_Algorithms_Reduced/letter_'+algo_name[i]+'_compl.csv')
    compl.ix[:,:,'spam'].to_csv('./P3_Clustering_Algorithms_Reduced/spam_'+algo_name[i]+'_compl.csv')
    adjMI.ix[:,:,'letter'].to_csv('./P3_Clustering_Algorithms_Reduced/letter_'+algo_name[i]+'_adjMI.csv')
    adjMI.ix[:,:,'spam'].to_csv('./P3_Clustering_Algorithms_Reduced/spam_'+algo_name[i]+'_adjMI.csv')

#%% Part 5 - Rerun neural network learner with dimensionally reduced letter dataset with additional cluster feature
    
print('Part 5 - Running neural network with dimensionally reduced letter dataset...')

# Run NN on original dataset without cluster dimension for comparison
grid ={'learning_rate_init':nn_lr,'hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
gs = GridSearchCV(mlp,grid,verbose=10,cv=5)
gs.fit(letterX,letterY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/letter_original.csv')

algo_name.append('original')

# Run NN on dimensionally reduced and original datasets with addition cluster dimension
for i in range(len(algo_name)):      
    #for i in range(4,5):
    # load datasets      
    letter = pd.read_hdf('datasets.hdf','letter_'+algo_name[i])    
    letterX = letter.drop('Y',1).copy().values
    letterY = letter['Y'].copy().values
    le = preprocessing.LabelEncoder()
    letterY = le.fit_transform(letterY)
     
    km = kmeans(random_state=5)
    gmm = myGMM(random_state=5)

    grid ={'addClustKM__n_clusters':clusters,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('addClustKM',appendClusterDimKM(cluster_algo = km)),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(letterX,letterY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/letter_km_'+algo_name[i]+'.csv')
    
    grid ={'addClustGMM__n_clusters':clusters,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('addClustGMM',appendClusterDimGMM(cluster_algo = gmm)),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(letterX,letterY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/letter_gmm_'+algo_name[i]+'.csv')
