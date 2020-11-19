import numpy as np
import matplotlib.pyplot as plt
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
#import glob
import random
import optuna
import joblib
import pathlib
import string
import variables
#models
import cnn_model
import capsnet_model

USE_CUDA = True if torch.cuda.is_available() else False

def standardize_image(image): 
        image = transforms.ToTensor()(image) 
        image = transforms.Normalize(mean=variables.mean, std=variables.std)(image) #calculated mean and standard deviation for whole dataset  
        return image

def image_augmentation(image):
        x = np.random.randint(0,15)
        augImg = image

        if x > 12:
            return image

        #adjustable randomized selection of augmentation option
        if x <= 3:
                augImg = image.transpose(method=Image.FLIP_LEFT_RIGHT)  #flip around vertical axis

        elif x > 3 and x <= 6:
                augImg = image.transpose(method=Image.FLIP_TOP_BOTTOM)  #flip around horizontal axis

        elif x > 6 and x <= 9:
                deg = np.random.randint(0,360)  #rotate random degrees
                augImg = image.rotate(deg)

        elif x > 9 and x <= 12 :

                horizontal = np.random.randint(4,10)   #range of possible shift
                vertical = np.random.randint(4,10)

                a = 1
                b = 0
                d = 0
                e = 1

                augImg = image.transform(image.size, Image.AFFINE, (a, b, horizontal, d, e, vertical))
        return augImg


class Classifier:
    def __init__(self, cfg):

        learning_rate = cfg["learning_rate"]
        batch_size = cfg["mini_batch_size"]
        image_size = cfg["image_size"]
        dropout = cfg["dropout"]
        optimizer = cfg["optimizer"]
        criterion = cfg["criterion"]
        dropout_rate = cfg["dropout_rate"]
        save_weights = cfg['save_weights']
        model_type = cfg['type'] #capsnet or convpool


        if USE_CUDA:
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("Cuda available")
        else:
            self.device = torch.device("cpu")
            self.cuda = False
        print("Selected device is :" + str(self.device))

        #Parameters
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.save = save_weights
        

        # Model type 
        self.type = model_type
        if self.type == "capsnet":
            self.model = capsnet_model.CapsNet()
        elif self.type == "convpool":
            self.model = cnn_model.ClassifierNet(self.image_size[0], self.image_size[1], dropout, dropout_rate)

        #optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

        self.paths = variables.train_set_paths_by_category
        #images and labels
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

        #loss and accuracy of model
        self.loss_list = []
        self.acc_list = []
        self.test_acc_list = []

    # Plot functions
    def plot_accuracy(self):
        indexes = [i+1 for i in range(len(self.acc_list))]
        plt.plot(indexes, self.acc_list)
        plt.ylabel('Training Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def plot_loss(self):
        indexes = [i+1 for i in range(len(self.loss_list))]
        plt.plot(indexes, self.loss_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    def plot_test_accuracy(self):
        indexes = [i+1 for i in range(len(self.test_acc_list))]
        print(self.acc_list)
        plt.plot(indexes, self.test_acc_list)
        plt.ylabel('Test Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def reset_batches(self):
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

    #TODO implement image visualizations
    def view_image(self):

        # open an image, resize it and print it
        groups = list(self.paths.keys())
        group = random.choice(groups)

        imagePath = np.random.choice(self.paths[group])
        im = Image.open(imagePath)
        im.thumbnail(self.image_size, Image.ANTIALIAS)
        im = np.array(im)
        #TODO resize without converting to numpy array? --Possible to resize directly in Pillow
        im = cv2.resize(im, self.image_size)
        image = Image.fromarray(im, "RGB")
        image.show()

        # convert to tensor for processing
        im = torch.from_numpy(im)
        if self.cuda:
            im = im.cuda().to(self.device)
        im = im.transpose(0, -1)
        im = im[None, :, :, :]
        x = im

        # tensor goes to CNN layers and print picture for each layer
        x = F.relu(self.model.conv1(x.float()))
        Classifier.tensor_to_image(self, x)
        x = self.model.pool1(x)
        Classifier.tensor_to_image(self, x)
        x = F.relu(self.model.conv2(x.float()))
        Classifier.tensor_to_image(self, x)
        x = self.model.pool2(x)
        Classifier.tensor_to_image(self, x)
        x = F.relu(self.model.conv3(x.float()))
        Classifier.tensor_to_image(self, x)
        x = self.model.pool3(x)
        Classifier.tensor_to_image(self, x)
        x = F.relu(self.model.conv4(x.float()))
        Classifier.tensor_to_image(self, x)
        x = self.model.pool4(x)
        Classifier.tensor_to_image(self, x)
        x = F.relu(self.model.conv5(x.float()))
        Classifier.tensor_to_image(self, x)
        x = self.model.pool5(x)
        Classifier.tensor_to_image(self, x)

    def tensor_to_image(self, tensor):
        if self.cuda:
            tensor = tensor.cpu()
        image = tensor.detach().clone().numpy()
        image = Image.fromarray(image[0][1], "RGB")
        image.show()

    # TODO plot loss, cross validation
    def plot_results(self):

        return None

    def reset_epoch(self):
        self.paths = variables.train_set_paths_by_category

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.name = path

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())

    def random_string_generator(self, length):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k = length))

        
    def save_weights(self, accuracy):
        path = variables.saved_weights_path + self.random_string_generator(6) + "_"  + self.type + "_accuracy_" + str(accuracy) + ".weights"
        torch.save(self.model.state_dict(), path)

    def predict(self, image):
        return self.model(image)
       
    def load_images(self):
        counter = 0
        for i in range(self.batch_size):
            groups = list(self.paths.keys())
            group = random.choice(groups)
            imagePath = np.random.choice(self.paths[group])
            # load the image, pre-process it, and store it in the data list
            im = Image.open(imagePath)
            im.thumbnail(self.image_size, Image.ANTIALIAS)
            im = np.array(im)
            # TODO resize without converting to numpy array?
            im = cv2.resize(im, self.image_size)
            im = standardize_image(im)
            if self.cuda:
                im = im.cuda().to(self.device)
            # TODO implement transpose, rotate, etc, randomly

            im = im[None, :, :, :]
            label = np.zeros(5)

            if group == "daisy":
                label[0] = 1
            elif group == "dandelion":
                label[1] = 1
            elif group == "rose":
                label[2] = 1
            elif group == "sunflower":
                label[3] = 1
            elif group == "tulip":
                label[4] = 1
            label = torch.Tensor([label])


            self.batch_images.update({str(counter): im})
            self.batch_labels.update({str(counter): label})
            self.batch_path.update({str(counter): imagePath})
            counter += 1


    def train(self,  number_of_epochs, number_of_batches, test_batch_size, trial = None):
        if self.type == "capsnet":
            self.train_capsNet(trial, number_of_epochs, number_of_batches, test_batch_size)
        elif self.type == "convpool":
            self.train_ConvPool(trial, number_of_epochs, number_of_batches, test_batch_size)

    def train_capsNet(self, trial,  number_of_epochs, number_of_batches, test_batch_size):
        print("Starting training on capsnet.")
        last_accuracy = 0
        for epoch in range(number_of_epochs):
            epoch_loss = 0
            epoch_acc = 0   
            for step in range(number_of_batches):
                correct = 0
                for i in range(self.batch_size):
                    # Run the forward pass
                    im = self.batch_images[str(i)]
                    output, reconstructions, masked = self.model(im)
                    

                    label = self.batch_labels[str(i)]
                    if self.cuda:   
                        label = label.cuda().to(self.device)
                    # TODO change to tensor in load_images
                    loss = self.model.loss(im, output, label, reconstructions)
                    epoch_loss += loss.item()

                    # Backprop and perform optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Track the accuracy
                    if torch.argmax(masked.data) == torch.argmax(label):
                        correct += 1
                    epoch_acc += (correct / self.batch_size)

                if (i + 1) % self.batch_size == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, number_of_epochs, step + 1, number_of_batches, loss.item(),
                                  (correct / self.batch_size) * 100))
                self.reset_batches()
                self.load_images()

            # Evaluate accuracy of model after this epoch
            # TODO: implement limit of testing size
            #accuracy = evaluation(self, test_batch_size, False)
            #Turn on model evaluation mode
            self.model.eval()
            #Evaluate on test set
            self.loss_list.append(epoch_loss)
            self.acc_list.append(epoch_acc / number_of_batches)
            accuracy = evaluation(self, 50, False)
            self.test_acc_list.append(accuracy)
            #Turn model training mode back on
            self.model.train()
            last_accuracy = accuracy
            # report accuracy of model now and evaluate if the current trial should prune
            if trial:
                trial.report(accuracy, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            self.reset_epoch()
        if(self.save):
            self.save_weights(last_accuracy)

    def train_ConvPool(self, trial, number_of_epochs, number_of_batches, test_batch_size):
        print("Starting training on convpool.")
        last_accuracy = 0
        for epoch in range(number_of_epochs):
            epoch_loss = 0
            epoch_acc = 0
            for step in range(number_of_batches):
                correct = 0
                for i in range(self.batch_size):
                    # Run the forward pass
                    im = self.batch_images[str(i)]
                    output = self.model(im)
                    
                    label = self.batch_labels[str(i)]
                    if self.cuda:   
                        label = label.cuda().to(self.device)
                    # TODO change to tensor in load_images
                    loss = self.criterion(output, label)
                    epoch_loss += loss.item()

                    # Backprop and perform Adam optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Track the accuracy
                    predicted = torch.round(output.data[0])
                    if torch.argmax(predicted) == torch.argmax(label):
                        correct += 1
                    epoch_acc += (correct / self.batch_size)

                if (i + 1) % self.batch_size == 0:  
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, number_of_epochs, step + 1, number_of_batches, loss.item(),
                                  (correct / self.batch_size) * 100))
                self.reset_batches()
                self.load_images()

            # Evaluate accuracy of model after this epoch
            # TODO: implement limit of testing size
            self.model.eval()
            #Evaluate on test set
            self.loss_list.append(epoch_loss)
            self.acc_list.append(epoch_acc / number_of_batches)
            accuracy = evaluation(self, 50, False)
            last_accuracy = accuracy
            self.test_acc_list.append(accuracy)
            #Turn model training mode back on
            self.model.train()
            # report accuracy of model now and evaluate if the current trial should prune
            if trial:
                trial.report(accuracy, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            self.reset_epoch()
        if(self.save):
            self.save_weights(last_accuracy)


def evaluation(Classifier, test_batch_size, prnt):
    paths = variables.train_set_paths_by_category        
    groups = list(paths.keys())
    counter = 0
    batch_images = {}
    batch_labels = {}
    error_results = {}
    for group in groups:
        for i in range(test_batch_size):
            imagePath = paths[group][i]
            # load the image, pre-process it, and store it in the data list
            im = Image.open(imagePath)
            im.thumbnail(Classifier.image_size, Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.resize(im, Classifier.image_size)
            im = standardize_image(im)
            if Classifier.cuda:
                im = im.cuda().to(Classifier.device)
            im = im[None, :, :, :]

            label = np.zeros(5)

            if group == "daisy":
                label[0] = 1
            elif group == "dandelion":
                label[1] = 1
            elif group == "rose":
                label[2] = 1
            elif group == "sunflower":
                label[3] = 1
            elif group == "tulip":
                label[4] = 1

            batch_images.update({str(counter): im})
            batch_labels.update({str(counter): label})

            counter += 1

    correct = 0
    total = 0

    errors = np.zeros(5)
    for i in range(test_batch_size*5):
        if Classifier.type == "capsnet":
            with torch.no_grad():
                _, _, output = Classifier.model(batch_images[str(i)])
            predicted = torch.argmax(output)
        elif Classifier.type == "convpool":
            output = Classifier.model(batch_images[str(i)]).detach()
            predicted = torch.argmax(output)
        label = batch_labels[str(i)]
        label = torch.argmax(torch.Tensor([label]))

        if predicted == label:
            correct += 1
        else:
            if label == 0:
                errors[0] += 1
                error_results.update({("daisy" + str(errors[0])): output})
            elif label == 1:
                errors[1] += 1
                error_results.update({("dandelion" + str(errors[1])): output})
            elif label == 2:
                errors[2] += 1
                error_results.update({("rose" + str(errors[2])): output})
            elif label == 3:
                errors[3] += 1
                error_results.update({("sunflower" + str(errors[3])): output})
            elif label == 4:
                errors[4] += 1
                error_results.update({("tulip" + str(errors[4])): output})

        total += 1
    accuracy = correct/total
    print("Cross validation:", accuracy)
    print("The agent managed", correct, "out of a total of:", total)
    if prnt:
        print("Errors in daisy images:", errors[0], "out of", test_batch_size)
        print("Errors in dandelion images:", errors[1], "out of", test_batch_size)
        print("Errors in rose images:", errors[2], "out of", test_batch_size)
        print("Errors in sunflower images:", errors[3], "out of", test_batch_size)
        print("Errors in tulip images:", errors[4], "out of", test_batch_size)
        print("-----------------------ERRORS-----------------------")
        for error in error_results:
            print(error, ":", error_results[error])
    # Returning accuracy for tuning of the model
    return accuracy
