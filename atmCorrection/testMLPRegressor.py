# =============================================================================
#     
#   Sentinnel 2 Atmospheric Correction Neural Network Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Trains a neural network to learn how to perform atmospheric correction
#       on Sentinnel 2 satellite images
# 
#   Notes
#       - Feature scaling is a big deal for inputs like sun angle (solved by output scaling)
#       - Summary statistics are hard to come by (solved by r2 score)
#
# =============================================================================

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score
import numpy as np

# =============================================================================
# Get data (make up data at the moment)
# =============================================================================
inputFeatNum = 4
outputFeatNum = 2

SampleNum = 100000
Samples = np.random.randint(40,size=(SampleNum,inputFeatNum))+1
Truth = np.zeros((SampleNum,outputFeatNum))
for i in range(SampleNum):
    Truth[i,0] = Samples[i,:].sum()
    Truth[i,1] = Samples[i,:].prod()
#    Truth[i,1] = Samples[i,:].mean()
    

# =============================================================================
# Preprocess Data
# =============================================================================
trainSamples,testSamples,trainTruth,testTruth = train_test_split(Samples,Truth,test_size=0.1)

inputScaler = StandardScaler()
inputScaler.fit(trainSamples)

trainSamples = inputScaler.transform(trainSamples)
testSamples = inputScaler.transform(testSamples)

#Should we? I don't think so
outputScaler = StandardScaler()
outputScaler.fit(trainTruth)

trainTruth = outputScaler.transform(trainTruth)
testTruth = outputScaler.transform(testTruth)

# =============================================================================
# Create, train and evaluate the network
# =============================================================================
layerNum = 25
neuronNum = 50
hLayers = np.ones(layerNum,dtype=np.int32)*neuronNum
mlp = MLPRegressor(hidden_layer_sizes = hLayers, max_iter = 100,n_iter_no_change=20,batch_size = int(SampleNum/200),
                   learning_rate='adaptive',tol=1e-6, verbose=True)

mlp.fit(trainSamples,trainTruth)

testPred = mlp.predict(testSamples)


# =============================================================================
# Compute Summary Statistics
# =============================================================================
testTruth = outputScaler.inverse_transform(testTruth)
testPred = outputScaler.inverse_transform(testPred)

relErr = np.abs(testPred-testTruth)/testTruth
relErrAvg = relErr.sum()/SampleNum
worstCase = relErr.max()
r2Score = r2_score(testTruth,testPred)
