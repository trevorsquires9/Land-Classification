# =============================================================================
#     
#    Atmospheric Correction Preprocessing Script Training Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Trains a shallow neural network over preprocessed training data
# 
#   TODO
#
# =============================================================================
# Necessary Packages
# =============================================================================.
from __future__ import division
import numpy as np
from joblib import load
import pickle


# =============================================================================
# Load the network
# =============================================================================
modelName = 'atmCorrectionModel.pkl'
mlp = pickle.load(open(modelName,'rb'))

# =============================================================================
# Load and train model
# =============================================================================  
numDataSets = 1
maxIt = 100
for j in np.arange(numDataSets):
    trainName = 'trainSet%d' %(j+1)
    truthName = 'truthSet%d' %(j+1)
    trainSamples = load(trainName)
    trainTruth = load(truthName)

    
    for i in np.arange(maxIt):
        mlp.partial_fit(trainSamples,trainTruth)

pickle.dump(mlp,open(modelName,'wb'))
