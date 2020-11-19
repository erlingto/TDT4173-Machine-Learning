import sys
sys.path.append("Source/")
import classifier, paramstudy, variables
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser("TDT4137 Project")
    parser.add_argument('--study', default=False, type=bool, help="Set True to conduct a study (Default False)")
    parser.add_argument('--train', default=False, type=bool, help="Set false to skip training (Default True)")
    parser.add_argument('--n_trials', default=10, type=int, help="Set number of study to conduct (Default 10)")
    parser.add_argument('--model', default="convpool", type=str, help="Work with convpool or capsnet model (Deafault convpool)")
    parser.add_argument('--plot', default=True, type=bool, help="Plot graphs at the end (Default True)")
    parser.add_argument('--load_weights', default=None, type=str, help="Load previous saved weights (Default is None)")
    

    args = parser.parse_args()
    print("TDT4137 Project loaded with following parameters")
    print(args)
    
    if args.study:
        # To conduct a study with n number of trials as parameter and the type of the model
        paramstudy.conduct_study(args.n_trials, args.type)
        exit()

    if (args.model == "capsnet"):
            cfg = variables.capsnet_cfg
    elif (args.model == "convpool"):
        cfg = variables.convpool_cfg
    TClassifier = classifier.Classifier(cfg)
    if args.load_weights is not None:
        TClassifier.load_weights(variables.saved_weights_path + args.load_weights)
        print("Weights loaded.")
    if args.train:
        TClassifier.load_images()
        TClassifier.train(cfg["epochs"], cfg["mini_batch_size"], cfg["test_batch_size"])
        if args.plot:
            TClassifier.plot_loss()
            TClassifier.plot_accuracy()
            TClassifier.plot_test_accuracy()
    TClassifier.model.eval()
    classifier.evaluation(TClassifier, cfg["test_batch_size"], cfg["prnt"])