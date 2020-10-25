import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from imutils import paths
import cv2
from PIL import Image
import glob
import random

def calculate_conv_output(W, K, P, S):
    return int((W-K+2*P)/S)+1
    
def calculate_flat_input(dim_1, dim_2, dim_3):
    return int(dim_1*dim_2*dim_3)

class ClassifierNet(nn.Module):

    def __init__(self, img_rows, img_cols):
        super(ClassifierNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32, kernel_size = 6, stride= 3, padding= 2)
        self.pool1 = nn.MaxPool2d(3,2, padding= 1)
        self.conv2 = nn.Conv2d(in_channels = 32 ,out_channels = 96, kernel_size = 3, stride= 2, padding= 1)
        self.pool2 = nn.MaxPool2d(2,2, padding= 1)
        self.conv3 = nn.Conv2d(in_channels = 96 ,out_channels = 192, kernel_size = 3, stride= 2, padding= 1)
        self.pool3 = nn.MaxPool2d(2,2, padding= 1)
        """ CONV DIMENSIONS CALCULATIONS """
        self.conv_output_H = calculate_conv_output(img_rows, 6, 2, 3)
        self.conv_output_W = calculate_conv_output(img_cols, 6, 2, 3)
        """ POOLING DIMENSIONS CALCULATIONS """
        self.conv_output_H = calculate_conv_output(self.conv_output_H , 3, 1, 2)
        self.conv_output_W = calculate_conv_output(self.conv_output_W , 3, 1, 2)
        
        for _ in range(2):
            self.conv_output_H = calculate_conv_output(self.conv_output_H , 3, 1, 2)
            self.conv_output_W = calculate_conv_output(self.conv_output_W , 3, 1, 2)
            self.conv_output_H = calculate_conv_output(self.conv_output_H , 2, 1, 2)
            self.conv_output_W = calculate_conv_output(self.conv_output_W , 2, 1, 2)
        
        self.linear = nn.Sequential(
            torch.nn.Linear(calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*192,  512), nn.ReLU(True),
            nn.Dropout(), nn.Linear(512, 5),
            )

        

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x=self.pool1(x)
        x = F.relu(self.conv2(x.float()))
        x=self.pool2(x)
        x = F.relu(self.conv3(x.float()))
        x=self.pool3(x)
        x = x.view(-1, calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*192)
        x= self.linear(x)
        return x

class Classifier:
    def __init__(self, learning_rate, batch_size):
        self.image_size = (320, 240)
        self.model = ClassifierNet(self.image_size[0],self.image_size[0])
        tulip =  glob.glob("Flowers/tulip/*")
        sunflower =  glob.glob("Flowers/sunflower/*")
        rose =  glob.glob("Flowers/rose/*")
        dandelion =  glob.glob("Flowers/dandelion/*")
        daisy =  glob.glob("Flowers/daisy/*")

        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()

        self.paths = {"tulip": tulip, "sunflower": sunflower, "rose": rose, "dandelion": dandelion, "daisy": daisy}
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

    def reset_batches(self):
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}


    def reset_epoch(self): 
        tulip =  glob.glob("Flowers/tulip/*")
        sunflower =  glob.glob("Flowers/sunflower/*")
        rose =  glob.glob("Flowers/rose/*")
        dandelion =  glob.glob("Flowers/dandelion/*")
        daisy =  glob.glob("Flowers/daisy/*")

        self.paths = {"tulip": tulip, "sunflower": sunflower, "rose": rose, "dandelion": dandelion, "daisy": daisy}

    def predict(self, image):
        return self.model(image)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.name = path

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def predict_test(self, imagePath):
        im = Image.open(imagePath)
        im.thumbnail(self.image_size, Image.ANTIALIAS)
        im = np.array(im)
        im = cv2.resize(im, self.image_size) 
        im = torch.from_numpy(im)
        im = im.transpose(0,-1)
        im = im[None, :, :]

        output = self.model(im)
        predicted = (output.data[0])
        if np.argmax(predicted) == 0:
            print("Bildet er av en daisy")
        elif np.argmax(predicted) == 1:
            print("Bildet er av en dandelion")
        elif np.argmax(predicted) == 2:
            print("Bildet er av en rose")
        elif np.argmax(predicted) == 3:
            print("Bildet er av en sunflower")
        elif np.argmax(predicted) == 4:
            print("Bildet er av en tulip")
        print(predicted)
        return predicted


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
            im = cv2.resize(im, self.image_size) 
            im = torch.from_numpy(im)
            im = im.transpose(0,-1)
            im = im[None, :, :]

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

            self.batch_images.update({str(counter): im})
            self.batch_labels.update({str(counter): label})
            self.batch_path.update({str(counter): imagePath})
            counter+= 1
    
    def train(self, number_of_epochs, number_of_batches):
        loss_list = []
        acc_list = []
        for epoch in range(number_of_epochs):
            for step in range(number_of_batches):
                correct = 0
                for i in range(self.batch_size):
                    # Run the forward pass
                    output = self.model(self.batch_images[str(i)])

                    label = self.batch_labels[str(i)]
                    label = torch.Tensor([label])
                    loss = self.criterion(output, label)
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                

                    # Track the accuracy
                    predicted = torch.round(output.data[0])
                    if np.argmax(predicted) == np.argmax(label):
                        correct+= 1
                    acc_list.append(correct / self.batch_size)

                if (i + 1) % self.batch_size == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                                .format(epoch + 1, number_of_epochs, step , number_of_batches, loss.item(),
                                        (correct / self.batch_size) * 100))
                self.reset_batches()
                self.load_images()
            self.reset_epoch()

        self.save_weights("Classifier")

def evaluation(Classifier, test_batch_size):
    tulip =  glob.glob("Test_Flowers/tulip/*")
    sunflower =  glob.glob("Test_Flowers/sunflower/*")
    rose =  glob.glob("Test_Flowers/rose/*")
    dandelion =  glob.glob("Test_Flowers/dandelion/*")
    daisy =  glob.glob("Test_Flowers/daisy/*")
    paths = {"tulip": tulip, "sunflower": sunflower, "rose": rose, "dandelion": dandelion, "daisy": daisy}
    groups = list(paths.keys())
    counter = 0
    batch_images = {}
    batch_labels = {}
    error_results = {}
    for group in groups:
        for imagePath in paths[group]:
            # load the image, pre-process it, and store it in the data list
            im = Image.open(imagePath)
            im.thumbnail(Classifier.image_size, Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.resize(im, Classifier.image_size) 
            im = torch.from_numpy(im)
            im = im.transpose(0,-1)
            im = im[None, :, :]

            while im.size() != torch.Size([1, 3, 320, 240]):
                imagePath = np.random.choice(paths[group])
                # load the image, pre-process it, and store it in the data list
                im = Image.open(imagePath)
                im.thumbnail(Classifier.image_size, Image.ANTIALIAS)
                im = np.array(im)
                im = cv2.resize(im, Classifier.image_size) 
                im = torch.from_numpy(im)
                im = im.transpose(0,-1)
                im = im[None, :, :]

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
            
            counter+= 1
    
    print("Cross validation:")
    correct = 0
    total = 0

    errors = np.zeros(5)
    for i in range(test_batch_size*5):
        output = DClassifier.model(batch_images[str(i)]).detach()
        predicted = np.argmax(output)
        label = batch_labels[str(i)]
        label = np.argmax(torch.Tensor([label]))
        

        if predicted == label:
            correct += 1
        else:
            if label == 0:
                errors[0] +=1
                error_results.update({("daisy" + str(errors[0])) : output } )
            elif label == 1:
                errors[1] +=1
                error_results.update({("dandelion" + str(errors[1])) : output } )
            elif label == 2:
                errors[2] +=1
                error_results.update({("rose" +  str(errors[2])) : output } )
            elif label == 3:
                errors[3] +=1
                error_results.update({("sunflower" + str(errors[3])) : output } )
            elif label == 4:
                errors[4] +=1
                error_results.update({("tulip" +  str(errors[4])) : output } )
        
        total += 1
    print("Cross validation:",correct/(total))
    print("The agent managed", correct, "out of a total of:", total)        
    print("Errors in daisy images:", errors[0], "out of", test_batch_size)
    print("Errors in dandelion images:", errors[1], "out of", test_batch_size)
    print("Errors in rose images:", errors[2], "out of", test_batch_size)
    print("Errors in sunflower images:", errors[3], "out of", test_batch_size)
    print("Errors in tulip images:", errors[4], "out of", test_batch_size)


    print ("-----------------------ERRORS-----------------------")
    for error in error_results:
        print(error ,":", error_results[error])

DClassifier = Classifier(0.000146, 32)
evaluation(DClassifier, 25)
DClassifier.load_images()
DClassifier.train(90, 32)
DClassifier.load_weights("classifier")
evaluation(DClassifier, 25)

#DClassifier.load_images()




