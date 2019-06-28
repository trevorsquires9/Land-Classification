# =============================================================================
#     
#    Atmospheric Correction Model Initialization Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Initializes regression model
# 
#   TODO
#
# =============================================================================
# Necessary Packages
# =============================================================================.
from __future__ import division
import numpy as np
from joblib import load
from sklearn.neural_network import MLPRegressor
import pickle

# =============================================================================
# Initialize Variables and Network
# =============================================================================.
modelName = 'atmCorrectionModel.pkl'
iniDataName = 'trainSet1'
iniTruthName = 'truthSet1'

batchSize = 256
tol = 1e-8
alpha = 0.001
maxIter = 1000


hLayers = np.arange(25,4,step=-1)
mlp = MLPRegressor(hidden_layer_sizes = hLayers, 
                   max_iter = maxIter, 
                   batch_size = batchSize,
                   n_iter_no_change=maxIter,
                   tol=tol,
                   alpha=alpha,
                   verbose=True,
                   warm_start=True)

# =============================================================================
# Load, train, and save model
# =============================================================================
trainSamples = load(iniDataName)
trainTruth = load(iniTruthName)
mlp.fit(trainSamples,trainTruth)

pickle.dump(mlp,open(modelName,'wb'))
