import classifier, paramstudy, variables
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser("TDT4137 Project")
    parser.add_argument('--train', default=True, type=bool, help="Set false to skip training (Default True)")
    parser.add_argument('--study', default=False, type=bool, help="Set True to conduct a study (Default False)")
    parser.add_argument('--n_trials', default=10, type=int, help="Set number of study to conduct (Default 10)")
    parser.add_argument('--type', default="convpool", type=str, help="Work with convpool or capsnet (Deafault convpool)")
    parser.add_argument('--plot', default=True, type=bool, help="Plot graphs at the end (Default True)")
    parser.add_argument('--load_weights', default=False, type=bool, help="Load previous saved weights (Default False)")


    args = parser.parse_args()
    print("TDT4137 Project loaded with following parameters")
    print(args)
    
    if args.study:
        # To conduct a study with n number of trials as parameter and the type of the model
        paramstudy.conduct_study(args.n_trials, args.type)

    elif args.train:
        cfg = variables.convpool_cfg
        TClassifier = classifier.Classifier(cfg)
        if args.load_weights:
            TClassifier.load_weights('classifier')
        TClassifier.load_images()
    
        TClassifier.train(cfg["epochs"],
                      cfg["step_size"], cfg["test_batch_size"])
        if args.plot:
            TClassifier.plot_loss()

        #To remove
        EvClassifier = classifier.Classifier(cfg)
        EvClassifier.copy_weights(TClassifier)
        accuracy = classifier.evaluation(EvClassifier, cfg["test_batch_size"], cfg["prnt"])
