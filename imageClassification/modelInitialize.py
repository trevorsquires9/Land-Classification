# =============================================================================
#     
#    Landcover Classification Model Initialization Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Initializes classifier model
# 
#   TODO
#
# =============================================================================
# Necessary Packages
# =============================================================================.
from __future__ import division
import numpy as np
from joblib import load
from sklearn.neural_network import MLPClassifier
import pickle

# =============================================================================
# Initialize Variables and Network
# =============================================================================.
mlpLayer = 3
mlpNeuron = 100
modelName = 'landcoverMLPModel.pkl'
iniDataName = 'samples1'
iniLabelsName = 'labels1'

layers = np.ones(mlpLayer,dtype = np.int)*mlpNeuron
mlp = MLPClassifier(solver='adam',
                    hidden_layer_sizes = layers.tolist(),
                    tol=1e-8,
                    warm_start=True,
                    verbose=True,
                    max_iter = 1000, 
                    batch_size = 256,
                    n_iter_no_change=500000,)

# =============================================================================
# Load, train, and save model
# =============================================================================
trainSamples = load(iniDataName)
trainTruth = load(iniLabelsName)
mlp.fit(trainSamples,trainTruth)

pickle.dump(mlp,open(modelName,'wb'))