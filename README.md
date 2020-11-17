# TDT4137 Machine Learning

## Project Description
This is the main project of the NTNU course TDT4137 - Machine Learning.
The purpose of this project is to compare classic Convolutional Neural Network (CNN) to the new Capsule neural network (Capsnet) on the same dataset.
We developed a program with our implementation of CNN based and an adaptation of a Capsnet implementation, both based on the Pythorch framework.
The original Capsnet implementation (All rights to jindongwang) can be found here: https://github.com/jindongwang/Pytorch-CapsuleNet

The runtime of this project is configurable by choosing witch classifier utilize (CNN or Capsnet) and a mode between "train and evaluate" or "conduct a study".
Both the implementations supports the use of a CUDA enabled graphic card for speeding up calculations on both modes.

### Train and Evaluate
In this mode, the program will train the selected model on the whished parameters. The program gives the possibility to save and load trained classifier weights. There are options for plotting statistics about the training prosess like the loss plot per epoch. 

### Conduct a study 
In this mode, the program will conduct a study on the selected classifier for finding the best hyperparameters. Both the number of trials and hyperparameters' value range is configurable.
A study is then saved as a CSV file and as a object dump. Graphs are generating visualising importance of each hyperparameter in the final score, trial-score and a empirical distribution of all the results. 
During a study, the program can early prune a non promising trial for time optimization of the study.
Parameter optimalization is implemented with the Optuna library, documentation can be found here: https://optuna.readthedocs.io/en/stable/

### Dataset
Dataset is composed of 4242 pictures of flowers divided in 5 categories: Sunflower, Rose, Dendelion, Daisy and Tulip.
Source of the dataset : https://www.kaggle.com/rishitchs/final-flowers-course-project-dataset

## Installation Guide

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
* **--type** (""convpool" or "capsnet") Which model to run. Default is "convpool".
* **--mode** ("train" or "study") Which mode to run. Default is "train".
* **--load_weights** (True or false) Wether to load the previuosly saved weights or not. Default is False.
* **--plot** (True or false) Wether to plot graphs after training process or not. Default is True.
* **--n_trials** (Int value) Number of trials when in study mode. Default is 10.
* **--use_cuda** (True or false) Wether do tensions operations on GPU or CPU. Default is False.

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
|__Results
   |
   |__Study
      |
      |__csv
```
