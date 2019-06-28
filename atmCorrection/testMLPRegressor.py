# =============================================================================
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
import geoio



# =============================================================================
# Create the network (alternatively, load the network)
# =============================================================================
firstTime = True

if firstTime:
    layerNum = 3
    neuronNum = 50
    batchSize = 256
    exitCond = 20
    maxIt = 5
    tol = 1e-6
    alpha = 0.0005
    hLayers = np.ones(layerNum,dtype=np.int32)*neuronNum
    mlp = MLPRegressor(hidden_layer_sizes = hLayers, 
                       max_iter = maxIt, 
                       batch_size = batchSize,
                       n_iter_no_change=5000,
                       tol=tol,
                       alpha=alpha,
                       verbose=False,
                       warm_start=True)
else:
    mlp = load('myModel.joblib')



# =============================================================================
 # Get data (make up data at the moment)
# =============================================================================
inputFeatNum = 4
outputFeatNum = 2
SampleNum = 10000
Samples1 = np.random.randint(40,size=(SampleNum,inputFeatNum))+1
Samples2 = np.random.randint(200,size=(SampleNum,inputFeatNum))+41
Truth1 = np.zeros((SampleNum,outputFeatNum))
Truth2 = np.zeros((SampleNum,outputFeatNum))
for i in range(SampleNum):
    Truth1[i,0] = Samples1[i,:].mean()
    Truth2[i,0] = Samples2[i,:].mean()
    Truth1[i,1] = Samples1[i,:].std()
    Truth2[i,1] = Samples2[i,:].std()    

trainSamples1,testSamples1,trainTruth1,testTruth1 = train_test_split(Samples1,Truth1,test_size=0.1)
trainSamples2,testSamples2,trainTruth2,testTruth2 = train_test_split(Samples2,Truth2,test_size=0.1)



# =============================================================================
# Train and predict
# =============================================================================
trials = 30
for i in np.arange(maxIt):
    mlp.partial_fit(trainSamples1,trainTruth1)

for i in np.arange(maxIt):
    mlp.partial_fit(trainSamples2,trainTruth2)

for j in np.arange(trials):
    print '\nTrial ' + str(j) + ':'
    
    for i in np.arange(maxIt):
        mlp.partial_fit(trainSamples1,trainTruth1)
        
    testPred1 = mlp.predict(testSamples1)
    testPred2 = mlp.predict(testSamples2)
    r2Score1 = r2_score(testTruth1,testPred1)
    r2Score2 = r2_score(testTruth2,testPred2)
    print 'Trained on set 1'
    print r2Score1
    print r2Score2
    
    for i in np.arange(maxIt):
        mlp.partial_fit(trainSamples2,trainTruth2)
     
    testPred1 = mlp.predict(testSamples1)
    testPred2 = mlp.predict(testSamples2)
    r2Score1 = r2_score(testTruth1,testPred1)
    r2Score2 = r2_score(testTruth2,testPred2)
    print 'Trained on set 2'
    print r2Score1
    print r2Score2
# =============================================================================
# Store new model
# =============================================================================
dump(mlp,'myModel.joblib')

