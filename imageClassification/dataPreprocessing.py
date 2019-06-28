# =============================================================================
#     
#   Landcover Classification Preprocessing Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#         Prepares an image for training using NDVI selection criterion
# 
#   TODO
#       - Add comments where helpful
#
#
# =============================================================================
# Necessary Packages
# =============================================================================.
from __future__ import division
import numpy as np
from joblib import dump,load

# =============================================================================
# Filename setup
# =============================================================================
sampleSetNum = load('sampleSetNum')
sampleSetNum += 1

trainFileName = 'rawTrainingData.txt'
trainDataName = 'samples%d' %(sampleSetNum)
truthDataName = 'labels%d' %(sampleSetNum)

# =============================================================================
# Read in data and shave
# =============================================================================
trainFile = np.genfromtxt(trainFileName,delimiter=',',dtype='float64')
trainFile = trainFile[1:,:]
allSamples = trainFile[:,2:20]#allSamples = trainFile[:,3:]
allLabels = trainFile[:,21]#allLabels = trainFile[:,1]
sampleNum = trainFile.shape[0]

# =============================================================================
# Save cut data
# =============================================================================
dump(allSamples,trainDataName)
dump(allLabels,truthDataName)
dump(sampleSetNum,'sampleSetNum')
