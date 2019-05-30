# =============================================================================
#     
#   Satellite Image Classification Tool
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Given satellite imagery in the form of a .tif file and a respectable 
#       number of labeled areas, this function allows the user to select between
#       multiple different classification algorithms while viewing summary 
#       statistics in the form of confusion matricesand out of bag cross validation. 
# 
#   Notes
#       - Figure out why it doesn't like to read .csv file and why I have to remove
#         the first training entry
#       - Add comments where helpful
#       - Figure out a better way to handle file IO (wrt directories)
#       - Handle the last not so block-block
#       - Save feature importances
#
# =============================================================================
# Necessary Packages
# =============================================================================
from __future__ import division
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import pandas as pd
from Tkinter import *
import gdal
from osgeo import gdal_array

# =============================================================================
# Nice mode function I found online https://stackoverflow.com/a/35674754
# =============================================================================
def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]

# =============================================================================
# Define functions for GUI
# =============================================================================
def getMainData():
    global svmDo,sgdDo,rfDo,mlpDo
    global image, train, imageFileName, trainFileName
    svmDo = svmRun.get()
    sgdDo = sgdRun.get()
    rfDo = rfRun.get()
    mlpDo = mlpRun.get()
    
    imageFileName = imageLabel.get()
    trainFileName = trainLabel.get()
    master.destroy()
    image = gdal.Open(imageFileName)
    train = gdal.Open(trainFileName)
    
    if svmDo:
        svmMaster = Tk()
        Label(svmMaster, text="Select your Kernel Function:").grid(row=0,column=0)
        kernelLabel = StringVar(svmMaster)
        kernelLabel.set('rbf')
        selectKernel = OptionMenu(svmMaster, kernelLabel,'rbf','poly','linear')
        selectKernel.grid(row=0,column=1)
        svmCommand = lambda : runSVM(svmMaster,kernelLabel)
        Button(svmMaster,text = "Continue",command = svmCommand).grid(row=5)
        svmMaster.mainloop()
        
    if sgdDo:
        sgdMaster = Tk()
        sgdAlert = Toplevel(sgdMaster)
        sgdAlert.title = 'Warning'
        sgdmsg = Message(sgdAlert,text='Warning: SGD is only a linear classifier and should not be relied upon for complicated classification')
        sgdmsg.pack()
        
        Label(sgdMaster, text="Select your Loss Function:").grid(row=0,column=0)
        lossLabel = StringVar(sgdMaster)
        lossLabel.set('l2')
        selectLoss = OptionMenu(sgdMaster, lossLabel,'l2','l1','elasticnet')
        selectLoss .grid(row=0,column=1)
        sgdCommand = lambda : runSGD(sgdMaster,lossLabel)
        Button(sgdMaster,text = "Continue",command = sgdCommand).grid(row=5)
        sgdMaster.mainloop()
        
    if rfDo:
        rfMaster = Tk()
        Label(rfMaster, text="Indicate the number of trees to use:").grid(row=0,column=0)
        treeNum = IntVar(rfMaster)
        treeNum.set(256)
        selectTree = OptionMenu(rfMaster, treeNum,32,64,128,256,512,1024,2048)
        selectTree.grid(row=0,column=1)
        rfCommand = lambda : runRF(rfMaster,treeNum)
        Button(rfMaster,text = "Continue",command = rfCommand).grid(row=5)
        rfMaster.mainloop()
    
    if mlpDo:
        mlpMaster = Tk()
        Label(mlpMaster, text="Indicate the number of hidden layers to use:").grid(row=0,column=0)
        layerNum = IntVar(mlpMaster)
        layerNum.set(10)
        selectLayer = OptionMenu(mlpMaster, layerNum,2,5,10,15,20,25,50,100)
        selectLayer.grid(row=0,column=1)
        Label(mlpMaster, text="Indicate the number of neurons to use:").grid(row=1,column=0)
        neuronNum = IntVar(mlpMaster)
        neuronNum.set(25)
        selectLayer = OptionMenu(mlpMaster, neuronNum,10,25,50,100,250,500)
        selectLayer.grid(row=1,column=1)
        mlpCommand = lambda : runMLP(mlpMaster,layerNum,neuronNum)
        Button(mlpMaster,text = "Continue",command = mlpCommand).grid(row=5)
        mlpMaster.mainloop()
    
    print 'Running the following classifiers:'
    if svmDo:
        print '\t- Support Vector Machine with an ' + kernelLabel.get() + ' kernel'
    if sgdDo:
        print '\t- Stochastic Gradient Descent with ' + lossLabel.get() + ' loss function'
    if rfDo:
        print '\t- Random Forest with ' + str(treeNum.get()) + ' trees'
    if mlpDo:
        print '\t- Multi-Layered Perception with ' + str(layerNum.get()) + ' layers and ' + str(neuronNum.get()) + ' neurons in each layer\n'

                
def runSVM(svmMaster,kernelLabel,):
    svmKernel = kernelLabel.get()
    svmMaster.destroy()
    global svmclf
    svmclf = svm.SVC(gamma='scale',kernel=svmKernel)

def runSGD(sgdMaster,lossLabel,):
    sgdLoss = lossLabel.get()
    sgdMaster.destroy()
    global sgdclf
    sgdclf = linear_model.SGDClassifier(max_iter=5000,penalty=sgdLoss,n_iter_no_change=50,tol=1e-7,warm_start=True)

def runRF(rfMaster,treeNum,):
    rfTree = treeNum.get()
    rfMaster.destroy()
    global rfclf
    rfclf = GradientBoostingClassifier(n_estimators = rfTree,max_features='sqrt',warm_start=True)

def runMLP(mlpMaster,layerNum,neuronNum,):
    mlpLayer = layerNum.get()
    mlpNeuron = neuronNum.get()
    mlpMaster.destroy()
    global mlpclf
    layers = np.ones(mlpLayer,dtype = np.int)*mlpNeuron
    mlpclf = MLPClassifier(solver='adam',hidden_layer_sizes = layers.tolist(),max_iter=4000,tol=1e-8,warm_start=True)

def readBlockData(image,rowStart, colStart, row,col,featNum):
    blockImage = image.ReadAsArray(colStart,rowStart,col,row)
    blockImage = blockImage.reshape((featNum,col*row)).transpose()
    return blockImage
    
    
# =============================================================================
# Define the GUI used for obtaining the data and specifications
# =============================================================================
master = Tk()
Label(master, text="Image to classify (should be .tif file)").grid(row=0)
Label(master, text="Labeled data ").grid(row=1)

imageLabel = Entry(master)
trainLabel = Entry(master)
imageLabel.insert(10,'myImage.tif')
imageLabel.grid(row=0, column=1)
trainLabel.insert(10,'myTrainingData.csv')
trainLabel.grid(row=1, column=1)
Label(master,text="Check classifiers that you would like to use").grid(row=2)

svmRun = IntVar() 
rfRun = IntVar() 
sgdRun = IntVar() 
mlpRun = IntVar() 

Checkbutton(master, text='SVM', variable=svmRun).grid(row=3,column=0, sticky=W) 
Checkbutton(master, text='Random Forest', variable=rfRun).grid(row=3,column=1, sticky=W) 
Checkbutton(master, text='SGD', variable=sgdRun).grid(row=4,column=0, sticky=W) 
Checkbutton(master, text='MLP', variable=mlpRun).grid(row=4,column=1, sticky=W) 


Button(master, text='Quit', command=master.quit).grid(row=5, column=0, sticky=W, pady=4)
Button(master, text='Continue', command=getMainData).grid(row=5, column=1, sticky=W, pady=4)

mainloop()

# =============================================================================
# Preprocessing of input data
# =============================================================================
trainFile = np.genfromtxt(trainFileName,delimiter=',',dtype='float64')
trainFile = trainFile[1:,:]
allSamples = trainFile[:,3:]
allLabels = trainFile[:,1]
sampleNum = trainFile.shape[0]

testingProportion = 0.25 #proportion of data to hold for testing

trainSampleNum = int(np.floor(sampleNum*(1-testingProportion)))
testSampleNum = sampleNum-trainSampleNum

testSampleInd = np.random.choice(sampleNum,testSampleNum,replace=False)
testSamples = allSamples[testSampleInd,:]
testLabels = allLabels[testSampleInd]

trainSampleInd = np.setdiff1d(np.arange(sampleNum),testSampleInd)
trainSamples = allSamples[trainSampleInd,:]
trainLabels = allLabels[trainSampleInd]

imageToClassify = gdal.Open(imageFileName)
imageXSize = imageToClassify.RasterXSize
imageYSize = imageToClassify.RasterYSize
featNum = imageToClassify.RasterCount
imageSize = imageXSize*imageYSize


# =============================================================================
# Do the training
# =============================================================================    
if svmDo:
    svmclf.fit(trainSamples,trainLabels)
if sgdDo:
    sgdclf.fit(trainSamples,trainLabels)
if rfDo:
    rfclf.fit(trainSamples,trainLabels)
if mlpDo:
    mlpclf.fit(trainSamples,trainLabels)

# =============================================================================
# Compute summary statistics and other analysis
# =============================================================================    
print '\nSummary Statistics'

if svmDo:
    svmTestPred = svmclf.predict(testSamples)
    svmTestAcc = (svmTestPred==testLabels).sum() / testLabels.size
    df = pd.DataFrame()
    df['truth'] = testLabels
    df['predict'] = svmTestPred
    print '\nSupport Vector Machine'
    print 'Out of bag accuracy: ' + str(svmTestAcc*100)
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True)) 
if sgdDo:
    sgdTestPred = sgdclf.predict(testSamples)
    sgdTestAcc = (sgdTestPred==testLabels).sum() / testLabels.size
    df = pd.DataFrame()
    df['truth'] = testLabels
    df['predict'] = sgdTestPred
    print '\nStochastic Gradient Descent'
    print 'Out of bag accuracy: ' + str(sgdTestAcc*100)
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True)) 
if rfDo:
    rfTestPred = rfclf.predict(testSamples)
    rfTestAcc = (rfTestPred==testLabels).sum() / testLabels.size
    df = pd.DataFrame()
    df['truth'] = testLabels
    df['predict'] = rfTestPred
    print '\nRandom Forest'
    print 'Out of bag accuracy: ' + str(rfTestAcc*100)
    print 'Feature importances:'
    for i in range(17):
        print '\tFeature ' + str(i+1) + ' has importance ' + str(rfclf.feature_importances_[i+1])
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True)) 
if mlpDo:
    mlpTestPred = mlpclf.predict(testSamples)
    mlpTestAcc = (mlpTestPred==testLabels).sum() / testLabels.size
    df = pd.DataFrame()
    df['truth'] = testLabels
    df['predict'] = mlpTestPred
    print '\nMulti-Layered Perceptron'
    print 'Out of bag accuracy: ' + str(mlpTestAcc*100)
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True))     

# =============================================================================
# Get ensemble specifications
# =============================================================================    
ensembleDialogue = Tk()
Label(ensembleDialogue, text='Based on summary statistics, select which of the following classifiers you would like to use.  (Note: selecting more than one will result in an ensemble classification)').grid(row=0)

svmUse = IntVar()
sgdUse = IntVar()
rfUse = IntVar()
mlpUse = IntVar() 
Checkbutton(ensembleDialogue, text='SVM', variable=svmUse).grid(row=1, sticky=W) 
Checkbutton(ensembleDialogue, text='SGD', variable=sgdUse).grid(row=2, sticky=W) 
Checkbutton(ensembleDialogue, text='Random Forest', variable=rfUse).grid(row=3, sticky=W) 
Checkbutton(ensembleDialogue, text='MLP', variable=mlpUse).grid(row=4, sticky=W) 
Button(ensembleDialogue,text = "Continue",command = ensembleDialogue.destroy).grid(row=5)
ensembleDialogue.mainloop()

# =============================================================================
# Perform ensemble classification
# =============================================================================
ensemble = np.array([svmUse.get(), sgdUse.get(), rfUse.get(), mlpUse.get()])
ensemblePred = np.ones((imageSize))*-1

colPerIt = 64
blockSize = imageXSize*colPerIt
blockPred = np.zeros((blockSize,4))
pred = np.ones((imageSize,4))*-1

lastEntry = np.arange(imageSize,step=blockSize)[-1:][0]

for i in np.arange(lastEntry,step=blockSize):
    tmpData = readBlockData(imageToClassify,int(i/imageXSize),0,colPerIt,imageXSize,featNum)
    if svmUse.get():
        svmPred = svmclf.predict(tmpData)
        blockPred[:,0] = svmPred
    if sgdUse.get():
        sgdPred = sgdclf.predict(tmpData)
        blockPred[:,1] = sgdPred
    if rfUse.get():
        rfPred = rfclf.predict(tmpData)
        blockPred[:,2] = rfPred
    if mlpUse.get():
        mlpPred = mlpclf.predict(tmpData)
        blockPred[:,3] = mlpPred
        
    blockPredValid = blockPred[:,np.nonzero(ensemble)].reshape((blockSize,ensemble.sum()))
    blockEnsemble = mode(blockPredValid.transpose())
    ensemblePred[i:i+blockSize] = blockEnsemble[0].transpose()
    pred[i:i+blockSize] = blockPred

tmpData = readBlockData(imageToClassify,int(lastEntry/imageXSize),0,int(imageYSize-lastEntry/imageXSize),imageXSize,featNum)
blockPred = np.zeros((imageXSize*(int(imageYSize-lastEntry/imageXSize)),4))
if svmUse.get():
    svmPred = svmclf.predict(tmpData)
    blockPred[:,0] = svmPred
if sgdUse.get():
    sgdPred = sgdclf.predict(tmpData)
    blockPred[:,1] = sgdPred
if rfUse.get():
    rfPred = rfclf.predict(tmpData)
    blockPred[:,2] = rfPred
if mlpUse.get():
    mlpPred = mlpclf.predict(tmpData)
    blockPred[:,3] = mlpPred
pred[lastEntry:] = blockPred

ensemblePred[lastEntry:] = mode(blockPred[:,np.nonzero(ensemble)].reshape((imageSize-lastEntry,ensemble.sum())).transpose())[0].transpose()

# =============================================================================
# Write to file
# =============================================================================
driver = gdal.GetDriverByName('GTiff')
outdata = driver.Create('Predictions.tif', imageYSize,imageXSize,ensemble.sum()+1,gdal.GDT_UInt16)
outdata.SetGeoTransform(imageToClassify.GetGeoTransform())
outdata.SetProjection(imageToClassify.GetProjection())
bandIt = 1
print 'Your .tif file will come with bands in the following order'
if svmUse.get():
    print 'SVM predictions'
    outdata.GetRasterBand(bandIt).WriteArray(pred[:,0].reshape(imageXSize,imageYSize))
    bandIt += 1
if sgdUse.get():
    print 'SGD predictions'
    outdata.GetRasterBand(bandIt).WriteArray(pred[:,1].reshape(imageXSize,imageYSize))
    bandIt += 1
if rfUse.get():
    print 'Random forest predictions'
    outdata.GetRasterBand(bandIt).WriteArray(pred[:,2].reshape(imageXSize,imageYSize))
    bandIt += 1
if mlpUse.get():
    print 'MLP predictions'
    outdata.GetRasterBand(bandIt).WriteArray(pred[:,3].reshape(imageXSize,imageYSize))
    bandIt += 1
    
print 'Ensemble predictions'
outdata.GetRasterBand(bandIt).WriteArray(ensemblePred.reshape(imageXSize,imageYSize))
outdata.FlushCache()
outdata = None
imageToClassify = None

    
    
    