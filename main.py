import sys
sys.path.append("Source/")
import classifier, paramstudy, variables
import argparse

import joblib


if __name__ == '__main__':

    parser = argparse.ArgumentParser("TDT4137 Project")
    parser.add_argument('--study', default=False, type=bool, help="Set True to conduct a study (Default False)")
    parser.add_argument('--train', default=False, type=bool, help="Set false to skip training (Default True)")
    parser.add_argument('--n_trials', default=10, type=int, help="Set number of study to conduct (Default 10)")
    parser.add_argument('--model', default="convpool", type=str, help="Work with convpool or capsnet model (Deafault convpool)")
    parser.add_argument('--plot', default=True, type=bool, help="Plot graphs at the end (Default True)")
    parser.add_argument('--load_weights', default=None, type=str, help="Load previous saved weights (Default is None)")
    

    args = parser.parse_args()
    # Load and print the logo in console
    logo = open("logo.txt", "r")
    for line in logo:
        line = line.strip()
        print(line)
    logo.close()

    print(" ")
    print(" ")
    print("************* TDT4137 Project loaded with following parameters *************")
    print(" ")
    print(args)
    print(" ")
    if args.study:
        # To conduct a study with n number of trials as parameter and the type of the model
        study = paramstudy.conduct_study(args.n_trials, args.model)
        if args.plot == True:
            # Functionality for loading a study dump is dropped in the last build since it was not a necessity
            #   and only used during development, but it can be used by removing the comment sign on the next line.
            # study = joblib.load("Results/Study/convpool_study_6.pkl")
            paramstudy.generate_graphs_from_study(study)
        exit()
    #Load the cfg for the selected model
    if (args.model == "capsnet"):
            cfg = variables.capsnet_cfg
    elif (args.model == "convpool"):
        cfg = variables.convpool_cfg
    # Initializate the model
    TClassifier = classifier.Classifier(cfg)
    # If selected, load the weights from file
    if args.load_weights is not None:
        TClassifier.load_weights(variables.saved_weights_path + args.load_weights)
        print("Weights loaded.")
    #If train is selected, train the model
    if args.train:
        TClassifier.load_images()
        TClassifier.train(cfg["epochs"], cfg["mini_batch_size"], cfg["test_batch_size"])
        #Plot the results of the training if required
        if args.plot:
            TClassifier.plot_loss()
            TClassifier.plot_accuracy()
            TClassifier.plot_test_accuracy()
    
    # Evaluate the model
    TClassifier.model.eval()
    classifier.evaluation(TClassifier, cfg["test_batch_size"], cfg["prnt"])

    # Reconstruct a random picture (only capsnet)
    if args.model =="capsnet":
        TClassifier.view_random_reconstruction()