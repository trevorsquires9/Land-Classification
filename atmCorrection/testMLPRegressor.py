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
#       - Summary statistics are hard to come by (solved by r2 score and MSE)
#
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np
from joblib import dump,load



# =============================================================================
# Create the network (alternatively, load the network)
# =============================================================================
firstTime = True

if firstTime:
    layerNum = 3
    neuronNum = 50
    batchSize = int(np.sqrt (100000))
    exitCond = 20
    maxIt = 200
    tol = 1e-6
    alpha = 0.0005
    hLayers = np.ones(layerNum,dtype=np.int32)*neuronNum
    mlp = MLPRegressor(hidden_layer_sizes = hLayers, 
                       max_iter = maxIt, 
                       batch_size = batchSize,
                       n_iter_no_change=5000,
                       tol=tol,
                       alpha=alpha,
                       verbose=True,
                       warm_start=True)
else:
    mlp = load('myModel.joblib')
    outputScaler = load('outputScale.joblib')
    inputScaler = load('inputScale.joblib')


# =============================================================================
 # Get data (make up data at the moment)
# =============================================================================
tableData = np.array([[0.905, -8.604],[0.94,-5.809],[0.938,-4.996],[0.962,-3.649],
                      [0.964,-3.021],[1,-4.521],[0.961,-5.522],[0.978,-2.992]])
 
inputFeatNum = 40
outputFeatNum = 20

SampleNum = 100000
Samples = np.random.randint(40,size=(SampleNum,inputFeatNum))+1
Truth = np.zeros((SampleNum,outputFeatNum))
for i in range(SampleNum):
    Truth[i,0] = Samples[i,:].sum()
    Truth[i,1] = Samples[i,:].prod()
#    Truth[i,1] = Samples[i,:].mean()
    

trainSamples,testSamples,trainTruth,testTruth = train_test_split(Samples,Truth,test_size=0.1)

if firstTime:
    inputScaler = StandardScaler()
    inputScaler.fit(trainSamples)
    outputScaler = StandardScaler()
    outputScaler.fit(trainTruth)

trainSamples = inputScaler.transform(trainSamples)
testSamples = inputScaler.transform(testSamples)

trainTruth = outputScaler.transform(trainTruth)
testTruth = outputScaler.transform(testTruth)


# =============================================================================
# Train and predict
# =============================================================================
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


# =============================================================================
# Store new model
# =============================================================================
dump(mlp,'myModel.joblib')
dump(inputScaler,'inputScale.joblib')
dump(outputScaler,'outputScale.joblib')
