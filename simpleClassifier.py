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
#        - 
#
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import pandas as pd
from Tkinter import *
# =============================================================================
## Define functions for GUI
def getMainData():
    master.destroy()
    if svmRun.get():
        svmMaster = Tk()
        Label(svmMaster, text="Select your Kernel Function:").grid(row=0,column=0)
        kernelLabel = StringVar(svmMaster)
        kernelLabel.set('rbf')
        selectKernel = OptionMenu(svmMaster, kernelLabel,'rbf','poly','linear')
        selectKernel.grid(row=0,column=1)
        svmVerbose = IntVar() 
        Checkbutton(svmMaster, text='Verbose', variable=svmVerbose).grid(row=1,column=0, sticky=W) 
        svmCommand = lambda : runSVM(svmMaster,kernelLabel,svmVerbose)
        Button(svmMaster,text = "Continue",command = svmCommand).grid(row=5)
        svmMaster.mainloop()
        
    if sgdRun.get():
        sgdMaster = Tk()
        sgdAlert = Toplevel(sgdMaster)
        sgdAlert.title = 'Warning'
        sgdmsg = Message(sgdAlert,text='Warning: SGD is a linear classifier and should not be relied upon for nonlinear classification')
        sgdmsg.pack()
        
        Label(sgdMaster, text="Select your Loss Function:").grid(row=0,column=0)
        lossLabel = StringVar(sgdMaster)
        lossLabel.set('l2')
        selectLoss = OptionMenu(sgdMaster, lossLabel,'l2','l1','elasticnet')
        selectLoss .grid(row=0,column=1)
        sgdVerbose = IntVar() 
        Checkbutton(sgdMaster, text='Verbose', variable=sgdVerbose).grid(row=1,column=0, sticky=W) 
        sgdCommand = lambda : runSGD(sgdMaster,lossLabel,sgdVerbose)
        Button(sgdMaster,text = "Continue",command = sgdCommand).grid(row=5)
        sgdMaster.mainloop()
        
    if rfRun.get():
        rfMaster = Tk()
        Label(rfMaster, text="Indicate the number of trees to use:").grid(row=0,column=0)
        treeNum = IntVar(rfMaster)
        treeNum.set(256)
        selectTree = OptionMenu(rfMaster, treeNum,32,64,128,256,512,1024,2048)
        selectTree.grid(row=0,column=1)
        rfVerbose = IntVar() 
        Checkbutton(rfMaster, text='Verbose', variable=rfVerbose).grid(row=1,column=0, sticky=W) 
        rfCommand = lambda : runRF(rfMaster,treeNum,rfVerbose)
        Button(rfMaster,text = "Continue",command = rfCommand).grid(row=5)
        rfMaster.mainloop()
    
    if mlpRun.get():
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
        mlpVerbose = IntVar() 
        Checkbutton(mlpMaster, text='Verbose', variable=mlpVerbose).grid(row=2,column=0, sticky=W) 
        mlpCommand = lambda : runMLP(mlpMaster,layerNum,neuronNum,mlpVerbose)
        Button(mlpMaster,text = "Continue",command = mlpCommand).grid(row=5)
        mlpMaster.mainloop()
        
def runSVM(svmMaster,kernelLabel,svmVerbose,):
    svmKernel = kernelLabel.get()
    svmMaster.destroy()
    global svmclf
    svmclf = svm.SVC(gamma='scale',kernel=svmKernel)
    svmclf.fit(X,y)
    if svmVerbose.get():
        df = pd.DataFrame()
        df['truth'] = y
        df['predict'] = svmclf.predict(X)
        print "\nClassifier: Support Vector Machine\n \tKernel: " + svmKernel + "\n\tConfusion Matrix:\n"
        print(pd.crosstab(df['truth'], df['predict'], margins=True))

def runSGD(sgdMaster,lossLabel,sgdVerbose,):
    sgdLoss = lossLabel.get()
    sgdMaster.destroy()
    global sgdclf
    sgdclf = linear_model.SGDClassifier(max_iter=5000,penalty=sgdLoss,n_iter_no_change=250,tol=1e-7)
    sgdclf.fit(X,y)
    if sgdVerbose.get():
        df = pd.DataFrame()
        df['truth'] = y
        df['predict'] = sgdclf.predict(X)
        print "\nClassifier: Stochastic Gradient Descent\n \tLoss function: " + sgdLoss + "\n\tConfusion Matrix:\n"
        print(pd.crosstab(df['truth'], df['predict'], margins=True))

def runRF(rfMaster,treeNum,rfVerbose,):
    rfTree = treeNum.get()
    rfMaster.destroy()
    global rfclf
    rfclf = RandomForestClassifier(n_estimators = rfTree, oob_score=rfVerbose)
    rfclf.fit(X,y)
    if rfVerbose.get():
        df = pd.DataFrame()
        df['truth'] = y
        df['predict'] = rfclf.predict(X)
        print "\nClassifier: Random Forest\n \tNumber of Trees: " + str(rfTree) + "\n\tOut of Bag Accuracy: " + str(rfclf.oob_score_)
        print "\n\tFeature importances:"
        for i in range(17):
            print '\t\tFeature ' + str(i+1) + ' has importance ' + str(rfclf.feature_importances_[i+1])
        print('\n\tConfusion Matrix')
        print(pd.crosstab(df['truth'], df['predict'], margins=True))
        
def runMLP(mlpMaster,layerNum,neuronNum,mlpVerbose,):
    mlpLayer = layerNum.get()
    mlpNeuron = neuronNum.get()
    mlpMaster.destroy()
    global mlpclf
    layers = np.ones(mlpLayer,dtype = np.int)*mlpNeuron
    mlpclf = MLPClassifier(solver='adam',hidden_layer_sizes = layers.tolist(),max_iter=4000,tol=1e-8)
    mlpclf.fit(X,y)
    if mlpVerbose.get():
        df = pd.DataFrame()
        df['truth'] = y
        df['predict'] = mlpclf.predict(X)
        print "\nClassifier: Multi-Layered Perceptron\n\tNumber of Hidden Layers: " + str(mlpLayer) + "\n\tNumber of Neurons: " + str(mlpNeuron)+"\n\tConfusion Matrix:\n"
        print(pd.crosstab(df['truth'], df['predict'], margins=True))

# =============================================================================
## Initialization of data - to be changed later
m = 80
n = 200
image = np.random.rand(m,n,18)
train = np.random.randint(20,size=(m,n))

sampleNum = (train>10).sum()
labels = np.unique(train[train>10])

X = image[train>10,:]
y = train[train>10]

newShape = (image.shape[0]*image.shape[1], image.shape[2])
imageVectorized = image.reshape(newShape)
# =============================================================================


# =============================================================================
## Define the GUI used for data
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

mainloop( )

    





