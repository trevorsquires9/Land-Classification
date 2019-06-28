Landcover Classification Model
Notes
	- Much more flexible than atmospheric correction.  Allows for multiple models and a stand alone application. 

Method of the stand alone application (simpleClassifier.py)
	1. Provide an image and training data in the desired format to the GUI. Data will be written as a .tif file in the same directory. This pretrained model takes in 18 band .tif files (9 from two separate dates of Sentinnel 2 data). The application will run the user through specifications. 

Method of training new model
	1. Run the dataPreprocessing.py script on each of your training data sets before applying it to the model. Be sure not to overwrite any data the model has already trained on! Also keep the nomenclature constant so that the trainer can read the files correctly. The trained model will be saved under a specified name. 
	2. If you are training a new model, run the modelInitialize.py function to instantiate an instance.  
	3. Run trainingScript.py with the desired number of iterations. Be sure to provide the name of the model you wish to train. Note that this will run maxIt number of iterations over each training data set (including the ones trained previously). 

Method of predicting
	1. Run the predictor.py script. The GUI will take in all information necessary. Data will be written as a .tif file in the same directory.



-------------------------------Possibly discontinued---------------------------------------------
Atmospheric Correction Model
Notes
	- the only nice changes to the model that can be made are the hidden layers.  It is not set up to train on another satellite (like wv3/sentinnel) and is very restrictive on what bands it uses.
	- only runs on 4 band data [B,G,R,NIR]
Method of training
	1. Run the dataPreprocessing.py script on each of your training data set before applying it to the model. From here, you may indicate the number of samples taken. Note that any image larger than 5000x5000 will be cropped. Be sure not to overwrite any data the model has already trained on! 
	2. If you are training a new model, run the modelInitialize.py function to instantiate an instance
	3. Run trainingScript.py with the desired number of iterations.  Note that this will run maxIt number of iterations over each training data set (including the ones trained previously). 

Method of predicting
	1. Run the predictor.py script. The GUI will take in all information necessary. Data will be written as a .tif file in the same directory.

-------------------------------Possibly discontinued---------------------------------------------

