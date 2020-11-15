#Variables used in this project
import glob
import torch.optim as optim
import torch.nn as nn


##Data Paths
train_set_path = "Data/Processed/train"

test_set_path = "Data/Processed/test"

study_result_path = "Results/Study"

study_csv_path = "Results/Study/csv"

train_set_paths_by_category = {"tulip": glob.glob(train_set_path + "/tulip/*"),
                         "sunflower": glob.glob(train_set_path + "/sunflower/*"),
                         "rose": glob.glob(train_set_path + "/rose/*"),
                         "dandelion": glob.glob(train_set_path + "/dandelion/*"),
                         "daisy": glob.glob(train_set_path + "/daisy/*")}

test_set_path_by_category = {"tulip": glob.glob(test_set_path + "/tulip/*"),
                             "sunflower":  glob.glob(test_set_path + "/sunflower/*"),
                             "rose":  glob.glob(test_set_path + "/rose/*"),
                             "dandelion":  glob.glob(test_set_path + "/dandelion/*"),
                             "daisy":  glob.glob(test_set_path + "/daisy/*")}


##Normalization

mean=[0.4557, 0.4188, 0.2996]

std=[0.2510, 0.2236, 0.2287]

## Model Configuration 


convpool_cfg =  {
        "type": "ConvPool", #CapsNet or ConvPool
        "image_size": (100, 100), # (X, Y)                                         
        "learning_rate": 6.34192248576476e-05,
        "mini_batch_size": 32, #Amount of images per batch
        "test_batch_size": 20, #Images per category to test on
        "step_size": 32, #Amount of batches per Epoch
        "epochs": 3,
        # trial.suggest_categorical('dropout', [True, False]),
        "dropout": True,
        "dropout_rate": 0.4,
        "prnt": True,
        "optimizer": optim.Adam,
        "criterion": nn.MSELoss(),
        "save_weights": True
    }
capsnet_cfg = {
        "type": "CapsNet", #CapsNet or ConvPool
        "image_size": (28, 28), # (X, Y)                                         
        "learning_rate": 6.34192248576476e-05,
        "mini_batch_size": 32, #Amount of images per batch
        "test_batch_size": 20, #Images per category to test on
        "step_size": 32, #Amount of batches per Epoch
        "epochs": 3,
        # trial.suggest_categorical('dropout', [True, False]),
        "dropout": True,
        "dropout_rate": 0.4,
        "prnt": True,
        "optimizer": optim.Adam,
        "criterion": nn.MSELoss(),
        "save_weights": True
    }