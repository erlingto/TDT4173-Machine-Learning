# TDT4137 Machine Learning

## Project Description
This is the main project of the NTNU course TDT4137 - Machine Learning.
The purpose of this project is to compare classic Convolutional Neural Network (CNN or convpool in this project) to the new Capsule neural network (Capsnet) on the same dataset.
We developed a program with our implementation of CNN based and an adaptation of a Capsnet implementation, both based on the Pythorch framework.
The original Capsnet implementation (All rights to jindongwang) can be found here: https://github.com/jindongwang/Pytorch-CapsuleNet
Modules for Image Augmentation and Standardization of the dataset are also implemented in this project.

The runtime of this project is configurable by choosing witch classifier utilize (convpool or capsnet) and a mode between "train and evaluate" or "conduct a study".
Both the implementations supports the use of a CUDA enabled graphic card for speeding up calculations on both modes.

### Models
A 

If not specified, the program will create or load a "convpool" model. In the user guide is shown how to customize the runtime of the program with parameters.
Some pre-trained models can be loaded in the program for evaluation purposes. Due to the high size, you can download it [from this link](https://drive.google.com/drive/folders/1EGAMN5IUjCIAX7SfmnecB9kryGvJJiMQ?usp=sharing) and place it in the project root. More instructions follows the download.

### "Train and Evaluate" mode
In this mode, the program will train the selected model on the whished parameters. The program gives the possibility to save and load trained classifier weights. There are options for plotting statistics about the training prosess like the loss plot per epoch. 
The program then proceedes to evaluate the model [here](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/classifier.py#L455).
The training phase can be skipped, in that case the program will only evaluate the model.
Training of the model is conducted by [this](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/classifier.py#L327) function that redirects the program to the right training routine for [capsnet](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/classifier.py#L333) and [convpool](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/classifier.py#L395).

### "Conduct a study" mode 
In this mode, the program will conduct a study on the selected classifier for finding the best hyperparameters. Both the number of trials and hyperparameters' value range is configurable.
A study is then saved as a CSV file and as a object dump. Graphs are generating visualising importance of each hyperparameter in the final score, trial-score and a empirical distribution of all the results. 

This mode is implemented in [Source\paramstudy.py](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/paramstudy.py), where the [conduct_study(n_trials, classifier_type)](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/paramstudy.py#L55) will start a new study and initializate a model with the configuration found on the top of the file. Then it procedes by iterate through the [objective(trial)](https://github.com/erlingto/TDT4173-Machine-Learning/blob/438386a52c396dfe8b8d47f246b47e9e452b67a9/Source/paramstudy.py#L20) function that will fully train the model with the passed values, evaluated and saved in the study object. Since the program evaluate the model after each training epoch, a non promising trial can be pruned for time optimization.
Parameter optimalization is implemented with the Optuna library, documentation can be found here: https://optuna.readthedocs.io/en/stable/

### Dataset
Dataset is composed of 4242 pictures of flowers divided in 5 categories: Sunflower, Rose, Dendelion, Daisy and Tulip.
Source of the dataset : https://www.kaggle.com/rishitchs/final-flowers-course-project-dataset

## Installation Guide
The project is developed used Python version 3.7, but newer version should be compatible too. Compatibility with Python 2.7 is not assured.
### CUDA Toolkit installation
CUDA Toolkit is necessary if you want to utilize your NVIDIA GPU for tensors operations.
Check if you have a compatibel GPU here: https://developer.nvidia.com/cuda-gpus
CUDA Toolkit used when development on this project: CUDA 10.2
Other versions should work too but not assured.
Installation guide for Windows and Linux: https://docs.nvidia.com/cuda/index.html#installation-guides

### Pythorch installation
If pythorch is not installed, follow the guide found on the official Pythorch website for installing the correct version for your machine.
Different versions are available, both with support of different CUDA versions and only CPU support.
Pythorch packages version used when developing this project: "torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0"
Official starting guide: https://pytorch.org/get-started/locally/

### Install required packages
Run "pip install -r requirements.txt" for installing the lasts of the required packages for this project.

If you use a Python version older then  3.2, run this program for installing ulterior needed packages:
"pip install argparse pathlib".

## User Guide
Run "main.py" for starting the program. The "main.py" script accepts the following parameters:
* **--model** (""convpool" or "capsnet") Which model to run. Default is "convpool".
* **--train** (True or false) Wether to train or not under the "Train and Evaluate" mode. Default is False.
* **--load_weights** (True or false) Load saved weights when in "Train and Evaluate" mode. Takes the filename as parameter. Default is None.
* **--plot** (True or false) Wether to plot graphs after training process or not. Default is True.
* **--study** (True or False) Wether to run in "Conduct a study" mode or not. This will overwrite the --train parameter. Default is False.
* **--n_trials** (Int value) Number of trials when in "Conduct a study" mode. Default is 10.

### Some examples of use:

* "**python main.py --train True**". This will run the program in "Train and Evaluate" mode on the "convpool" model without either skipping training nor loading weights. 
* "**python main.py --model convpool --load_weights GDRHSD_capsnett_accuracy_0.49.weights**". The program will load the weigts of a previously trained "capsnet" model and will evaluate on it without training it first.
* "**python main.py --study True --n_trials 100**". With this command, the program will then conduct a study with 100 trials on a convpool model.

### Variables
#### Train and Evaluate mode
The variables for both convpool and capsnett models are found in the "variables.py" file.
```
variables.py
...
convpool_cfg =  {
        "type": "ConvPool", #CapsNet or ConvPool
        "image_size": (100, 100), # (X, Y)                                         
        "learning_rate": 6.34192248576476e-05,
        "mini_batch_size": 32, #Amount of images per batch
        "test_batch_size": 20, #Images per category to test on
        "step_size": 32, #Amount of steps per Epoch
        "epochs": 200,
        "dropout": True,
        "dropout_rate": 0.4,
        "prnt": True,  #Print more informations about the errors after evaluation
        "optimizer": optim.Adam,
        "criterion": nn.MSELoss(),
        "save_weights": True
    }

### Settings for tweaking training and testing with Capsule neural network
capsnet_cfg = {
        "type": "CapsNet", #CapsNet or ConvPool
        "image_size": (100, 100), # (X, Y)                                         
        "learning_rate": 0.0077,
        "mini_batch_size": 10, #Amount of images per batch
        "test_batch_size": 20, #Images per category to test on
        "step_size": 10, #Amount of steps per Epoch
        "epochs": 50,
        "dropout": True,
        "dropout_rate": 0.4,
        "prnt": True, #Print more informations about the errors after evaluation
        "optimizer": optim.Adam,
        "criterion": nn.MSELoss(),
        "save_weights": True
    }
```
#### Conduct a study mode
The variables used when conduct a study are found under the "objective" function in the "paramstudy.py" file.
It utilizes a Optuna.trial.trial object that, for each study, will suggest parameters out of the value range manually defined in the cfg object.
There are different "suggest" methods with different values distributions. Documentation is found here: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
Results of the conducted study can be found under Results\Study\csv. A dump of the "study" object is found in Results\study, usefull for future analisis or fuctionality implementation.
Documentation 
```
paramstudy.py
...
def objective(trial):
    
    cfg = {
        "type": "CapsNet", #CapsNet or ConvPool
        "image_size": trial.suggest_categorical('image_size', [(224, 224), (180, 180), (150, 150),(300, 300)]),
        "learning_rate": trial.suggest_loguniform('lr', low=1e-3, high=1e-2),
        "mini_batch_size": 32, #Amount of images per batch
        "test_batch_size": 20,  #Images per category to test on
        "step_size": 10, #Amount of steps per Epoch
        "epochs": trial.suggest_int('epochs', low=30, high=60, step=5),
        "dropout": True,
        "dropout_rate":trial.suggest_discrete_uniform('droput_rate', low=0.1, high=0.5, q=0.1),
        "prnt": False,
        "optimizer": optim.Adam, #trial.suggest_categorical('optimizer', [optim.Adam, optim.SGD]),
        "criterion": nn.MSELoss(),
        "save_weights": False
    }
    ...
 ```
 
## Folder structure
```
Root
|
|__Data
|  |
|  |___Processed
|      |
|      |__test
|      |
|      |__train
|
|__Models
|
|
|__Source
|
|
|__Results
   |
   |__Plots
   |  |
   |  |__capsnet
   |  |
   |  |__convpool
   |
   |__Study
      |
      |__csv
```
## Further Work
This project is presented in a dedicated paper that would be the goal for this course and it will be considered finished upon delivery of such paper. Nevertheless the design and functionality of this program can be improved and extended. Faster training time can be reached by redesining the way the program manages the pictures and tensors. Better file structure, variable separation and user interface are also point where improvement is possible.
