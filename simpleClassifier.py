# =============================================================================
#     
#   Satellite Image Classification Tool
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Given satellite imagery in the form of a .tiff file and a respectable 
#       number of labeled areas, this function allows the user to select between
#       multiple different classification algorithms while viewing summary 
#       statistics in the form of confusion matrices, out of bag errors, and 
# 
#   Notes
#       - Need to figure out how to read in .tiff files and preprocess
#       - Make sure to read data in small chunks
#       - Would like confusion matrix to be on testing set instead of training
#       - If there is a tie for best classification, ensemble currently chooses
#         the lowest classification (definitely not optimal)
#
# =============================================================================
# Necessary Packages
# =============================================================================
from __future__ import division
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import pandas as pd
from Tkinter import *

# =============================================================================
# Define functions for GUI
# =============================================================================
def getMainData():
    master.destroy()
    global svmDo,sgdDo,rfDo,mlpDo
    svmDo = svmRun.get()
    sgdDo = sgdRun.get()
    rfDo = rfRun.get()
    mlpDo = mlpRun.get()
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
    if svmDo:
        svmclf.fit(Xtrain,yTrain)
    if sgdDo:
        sgdclf.fit(Xtrain,yTrain)
    if rfDo:
        rfclf.fit(Xtrain,yTrain)
    if mlpDo:
        mlpclf.fit(Xtrain,yTrain)
                
def runSVM(svmMaster,kernelLabel,):
    svmKernel = kernelLabel.get()
    svmMaster.destroy()
    global svmclf
    svmclf = svm.SVC(gamma='scale',kernel=svmKernel)

def runSGD(sgdMaster,lossLabel,):
    sgdLoss = lossLabel.get()
    sgdMaster.destroy()
    global sgdclf
    sgdclf = linear_model.SGDClassifier(max_iter=5000,penalty=sgdLoss,n_iter_no_change=50,tol=1e-7)

def runRF(rfMaster,treeNum,):
    rfTree = treeNum.get()
    rfMaster.destroy()
    global rfclf
    rfclf = RandomForestClassifier(n_estimators = rfTree, oob_score=True)

def runMLP(mlpMaster,layerNum,neuronNum,):
    mlpLayer = layerNum.get()
    mlpNeuron = neuronNum.get()
    mlpMaster.destroy()
    global mlpclf
    layers = np.ones(mlpLayer,dtype = np.int)*mlpNeuron
    mlpclf = MLPClassifier(solver='adam',hidden_layer_sizes = layers.tolist(),max_iter=4000,tol=1e-8)


# =============================================================================
# Initialization of data - to be changed later
# =============================================================================
m = 80
n = 200
image = np.random.rand(m,n,18)
train = np.random.randint(20,size=(m,n))

sampleNum = (train>10).sum()
labels = np.unique(train[train>10])


Xall = image[train>10,:]
testSize = np.floor((Xall.shape[0]*3)/4)

Xtrain = Xall[0:int(testSize),:]
Xtest = Xall[Xtrain.shape[0]:Xall.shape[0],:]

yAll = train[train>10]
yTrain = yAll[0:int(testSize)]
yTest = yAll[Xtrain.shape[0]:Xall.shape[0]]

newShape = (image.shape[0]*image.shape[1], image.shape[2])
imageVectorized = image.reshape(newShape)

# =============================================================================
# Define the GUI used for obtaining
# =============================================================================
master = Tk()
Label(master, text="Image to classify (should be .tif file)").grid(row=0)
Label(master, text="Labeled data ").grid(row=1)

imageLabel = Entry(master)
trainLabel = Entry(master)
imageLabel.grid(row=0, column=1)
trainLabel.grid(row=1, column=1)
Label(master,text="Check the methods that will classify the data").grid(row=2)

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
# Compute summary statistics and other analysis
# =============================================================================    
print '\nSummary Statistics'

if svmDo:
    svmTestPred = svmclf.predict(Xtest)
    svmTestAcc = (svmTestPred==yTest).sum() / yTest.size
    df = pd.DataFrame()
    df['truth'] = yTest
    df['predict'] = svmTestPred
    print '\nSupport Vector Machine'
    print 'Out of bag accuracy: ' + str(svmTestAcc*100)
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True)) 
if sgdDo:
    sgdTestPred = sgdclf.predict(Xtest)
    sgdTestAcc = (sgdTestPred==yTest).sum() / yTest.size
    df = pd.DataFrame()
    df['truth'] = yTest
    df['predict'] = sgdTestPred
    print '\nStochastic Gradient Descent'
    print 'Out of bag accuracy: ' + str(sgdTestAcc*100)
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True)) 
if rfDo:
    rfTestPred = rfclf.predict(Xtest)
    rfTestAcc = (rfTestPred==yTest).sum() / yTest.size
    df = pd.DataFrame()
    df['truth'] = yTest
    df['predict'] = rfTestPred
    print '\nRandom Forest'
    print 'Out of bag accuracy: ' + str(rfTestAcc*100)
    print 'Feature importances:'
    for i in range(17):
        print '\tFeature ' + str(i+1) + ' has importance ' + str(rfclf.feature_importances_[i+1])
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True)) 
if mlpDo:
    mlpTestPred = mlpclf.predict(Xtest)
    mlpTestAcc = (mlpTestPred==yTest).sum() / yTest.size
    df = pd.DataFrame()
    df['truth'] = yTest
    df['predict'] = mlpTestPred
    print '\nMulti-Layered Perceptron'
    print 'Out of bag accuracy: ' + str(mlpTestAcc*100)
    print 'Confusion Matrix'
    print(pd.crosstab(df['truth'], df['predict'], margins=True))     

# =============================================================================
# Ensemble Techniques
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

ensemble = [svmUse.get(), sgdUse.get(), rfUse.get(), mlpUse.get()]
pred = np.zeros((imageVectorized.shape[0],4))

if svmUse.get():
    svmPred = svmclf.predict(imageVectorized)
    pred[:,0] = svmPred
if sgdUse.get():
    sgdPred = sgdclf.predict(imageVectorized)
    pred[:,1] = sgdPred
if rfUse.get():
    rfPred = rfclf.predict(imageVectorized)
    pred[:,2] = rfPred
if mlpUse.get():
    mlpPred = mlpclf.predict(imageVectorized)
    pred[:,3] = mlpPred
    
pred = pred[:,np.nonzero(ensemble)]    

ensemblePred = np.zeros(imageVectorized.shape[0]) 
count = 0

for i in pred:
    ensemblePred[count] = scipy.stats.mode(i,axis=None)[0]
    count = count + 1
    
    
    