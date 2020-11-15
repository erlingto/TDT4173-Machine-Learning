import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from imutils import paths
import cv2
from PIL import Image
import glob
import random
import optuna
import joblib
import pathlib
from classifier import Classifier, evaluation
import variables

def objective(trial):
    
    cfg = {
        "type": "ConvPool", #CapsNet or ConvPool
        "image_size": trial.suggest_categorical('image_size', [(224, 224), (180, 180), (150, 150),
                                                         (300, 300)]),
        # 0.000134,
        "learning_rate": trial.suggest_loguniform('lr', low=1e-6, high=1e-4),
        "mini_batch_size": 32,
        "test_batch_size": 150,
        "step_size": 32,
        "epochs": trial.suggest_int('epochs', low=50, high=60, step=5),
        # trial.suggest_categorical('dropout', [True, False]),
        "dropout": True,
        "dropout_rate": trial.suggest_discrete_uniform('droput_rate', low=0.1, high=0.5, q=0.1),
        "prnt": False,
        "optimizer": trial.suggest_categorical('optimizer', [optim.Adam, optim.SGD]),
        "criterion": nn.MSELoss(),
        "save_weights": False
        
    }
    
    print("Beginning Trial nr. " + str(trial.number) + " with params:")
    print(trial.params)
    TClassifier = Classifier(cfg)
    TClassifier.load_images()
    TClassifier.train(cfg["epochs"],
                      cfg["step_size"], cfg["test_batch_size"], trial)
    cfg["dropout"] = False
    EvClassifier = Classifier(cfg)
    EvClassifier.copy_weights(TClassifier)
    accuracy = evaluation(EvClassifier, cfg["test_batch_size"], cfg["prnt"])
    return accuracy


def conduct_study(n_trials):    
    # Conduct a study with n numbers of trials

    # choose a sampler. Docs here: https://optuna.readthedocs.io/en/stable/reference/samplers.html#module-optuna.samplers
    sampler = optuna.samplers.TPESampler()
    # choose a pruner. Docs here: https://optuna.readthedocs.io/en/stable/reference/pruners.html
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        sampler=sampler, direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    save_study_to_file(study)
    return study


def save_study_to_file(study):
    # save results as a joblib dump
    filename = 'classifier_study_' + str(len(glob.glob('trial_results/*')))
    file_path = pathlib.Path().absolute().joinpath(
        variables.study_result_path, filename + '.pkl')
    joblib.dump(study, file_path)
    # Save a copy as a csv file
    data_frame = study.trials_dataframe()
    csv_path = pathlib.Path().absolute().joinpath(
        variables.study_csv_path, filename + '.csv')
    data_frame.to_csv(csv_path)


def read_study_from_file(filename):
    # read results from file with name=filename and return the study object
    file_path = pathlib.Path().absolute().joinpath(variables.study_result_path, filename)
    return joblib.load(file_path)


def generate_graphs_from_study(study):
    # plot graph representing weight of each hyperparameter
    figure = optuna.visualization.plot_param_importances(study)
    figure.show()
    # plot graph representing history of evaluation per trial
    figure = optuna.visualization.plot_optimization_history(study)
    figure.show()
    # plot empirical distribution function of the result of the study
    figure = optuna.visualization.plot_edf(study)
    figure.show()


if __name__ == '__main__':
    cfg = variables.convpool_cfg
    # To conduct a study with n number of trials as parameter, comment this if you only want to read a
    TClassifier = Classifier(cfg)
    #TClassifier.load_weights('classifier')
    TClassifier.load_images()
    
    #accuracy = capsnet_evaluation(TClassifier, 50, cfg["prnt"])
    TClassifier.train(cfg["epochs"],
                      cfg["step_size"], cfg["test_batch_size"])
    TClassifier.plot_loss()
    #study = read_study_from_file("classifier_study_0.pkl")
    EvClassifier = Classifier(cfg)
    EvClassifier.copy_weights(TClassifier)
    accuracy = evaluation(EvClassifier, cfg["test_batch_size"], cfg["prnt"])
