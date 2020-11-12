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

def standardize_image(image): 
    image = transforms.ToTensor()(image) 
    image = transforms.Normalize(mean=[0.4557, 0.4188, 0.2996], std=[0.2510, 0.2236, 0.2287])(image) #calculated mean and std for whole dataset  
    return image

def calculate_conv_output(W, K, P, S):
    return int((W-K+2*P)/S)+1

def calculate_flat_input(dim_1, dim_2, dim_3):
    return int(dim_1*dim_2*dim_3)


class ClassifierNet(nn.Module):

    def __init__(self, img_rows, img_cols, dropout, dropout_rate):

        super(ClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=0)
        self.conv4 = nn.Conv2d(
            in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=0)
        self.conv5 = nn.Conv2d(
            in_channels=192, out_channels=350, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=0)
        """ CONV DIMENSIONS CALCULATIONS """
        self.conv_output_H = img_rows
        self.conv_output_W = img_cols

        for _ in range(4):
            """ CONV DIMENSIONS CALCULATIONS """
            self.conv_output_H = calculate_conv_output(
                self.conv_output_H, 3, 1, 1)
            self.conv_output_W = calculate_conv_output(
                self.conv_output_W, 3, 1, 1)
            """ POOLING DIMENSIONS CALCULATIONS """
            self.conv_output_H = calculate_conv_output(
                self.conv_output_H, 2, 0, 2)
            self.conv_output_W = calculate_conv_output(
                self.conv_output_W, 2, 0, 2)

        self.conv_output_H = calculate_conv_output(self.conv_output_H, 3, 1, 1)
        self.conv_output_W = calculate_conv_output(self.conv_output_W, 3, 1, 1)
        """ POOLING DIMENSIONS CALCULATIONS """
        self.conv_output_H = calculate_conv_output(self.conv_output_H, 2, 0, 2)
        self.conv_output_W = calculate_conv_output(self.conv_output_W, 2, 0, 2)
        print("Convolutional output: " + str(self.conv_output_H) +
              "X" + str(self.conv_output_W))

        if dropout:
            self.linear = nn.Sequential(
                torch.nn.Linear(calculate_flat_input(
                    1, self.conv_output_H, self.conv_output_W)*350,  1024), nn.ReLU(True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(1024, 5),
                nn.Softmax(dim=1)
            )
        else:
            self.linear = nn.Sequential(
                torch.nn.Linear(calculate_flat_input(
                    1, self.conv_output_H, self.conv_output_W)*350,  1024), nn.ReLU(True),
                nn.Dropout(0),
                nn.Linear(1024, 5),
                nn.Softmax(dim=1)
            )
        if torch.cuda.is_available():
            self.cuda()
        else:
            print("NO CUDA to activate")

    # Ovveride the forward function in nn.Module
    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = self.pool1(x)
        x = F.relu(self.conv2(x.float()))
        x = self.pool2(x)
        x = F.relu(self.conv3(x.float()))
        x = self.pool3(x)
        x = F.relu(self.conv4(x.float()))
        x = self.pool4(x)
        x = F.relu(self.conv5(x.float()))
        x=self.pool5(x)
        x = x.view(-1, calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*350)
        x= self.linear(x)
        return x

""" TAKEN FROM WEB: https://github.com/jindongwang/Pytorch-CapsuleNet/blob/master/capsnet.py?fbclid=IwAR30r7IwgFTkAvvM9aNB9GTvN2vrdF5Uc1fnsvYEf-hUDt1ckMAkfFed9Wk """ 

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=5, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=3):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        
        classes = torch.sqrt((x ** 2).sum(2))
        
        classes = F.softmax(classes, dim=1)
        

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(5))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()
        if USE_CUDA:
            self.cuda()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

""" NOT TAKEN FROM WEB """ 

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

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("Cuda available")
        else:
            self.device = torch.device("cpu")
            self.cuda = False
        print("Selected device is :" + str(self.device))

        self.image_size = image_size
        #self.model = ClassifierNet(
        #    self.image_size[0], self.image_size[1], dropout, dropout_rate)

        self.model = CapsNet()
        # Todo generalize classifier, take out paths maybe in a data loader function and pass it to classifier
        tulip = glob.glob("Flowers/tulip/*")
        sunflower = glob.glob("Flowers/sunflower/*")
        rose = glob.glob("Flowers/rose/*")
        dandelion = glob.glob("Flowers/dandelion/*")
        daisy = glob.glob("Flowers/daisy/*")
        self.paths = {"tulip": tulip, "sunflower": sunflower,
                      "rose": rose, "dandelion": dandelion, "daisy": daisy}

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion  # nn.MSELoss()

        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

    def reset_batches(self):
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

    #TODO implement randomized image augmentation
    def image_augmentation(self, image):

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

    # TODO implement capsule net
    def capsulenet(self):
        return None

    # TODO plot loss, cross validation
    def plot_results(self):
        return None

    def reset_epoch(self):
        # TODO Generalize reset function
        tulip = glob.glob("Flowers/tulip/*")
        sunflower = glob.glob("Flowers/sunflower/*")
        rose = glob.glob("Flowers/rose/*")
        dandelion = glob.glob("Flowers/dandelion/*")
        daisy = glob.glob("Flowers/daisy/*")

        self.paths = {"tulip": tulip, "sunflower": sunflower,
                      "rose": rose, "dandelion": dandelion, "daisy": daisy}

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.name = path

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, image):
        return self.model(image)

    def predict_test(self, imagePath):
        im = Image.open(imagePath)
        im.thumbnail(self.image_size, Image.ANTIALIAS)
        im = np.array(im)
        im = cv2.resize(im, self.image_size)
        im = torch.from_numpy(im).cuda().to(self.device)
        im = im.transpose(0, -1)
        im = im[None, :, :, :]

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
            im = self.image_augmentation(im)
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

            self.batch_images.update({str(counter): im})
            self.batch_labels.update({str(counter): label})
            self.batch_path.update({str(counter): imagePath})
            counter += 1



    def train_capsulenet(self, number_of_epochs, number_of_batches, test_batch_size):
        loss_list = []
        acc_list = []
        for epoch in range(number_of_epochs):
            for step in range(number_of_batches):
                correct = 0
                for i in range(self.batch_size):
                    # Run the forward pass
                    im = self.batch_images[str(i)]
                    output, reconstructions, masked = self.model(im)
                    
                    label = self.batch_labels[str(i)]
                    label = torch.Tensor([label])
                    if self.cuda:   
                        label = label.cuda().to(self.device)
                    # TODO change to tensor in load_images
                    loss = self.model.loss(im, output, label, reconstructions)
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Track the accuracy
                    if torch.argmax(masked.data) == torch.argmax(label):
                        correct += 1
                    acc_list.append(correct / self.batch_size)

                if (i + 1) % self.batch_size == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, number_of_epochs, step + 1, number_of_batches, loss.item(),
                                  (correct / self.batch_size) * 100))
                self.reset_batches()
                self.load_images()

            # Evaluate accuracy of model after this epoch
            # TODO: implement limit of testing size
            #accuracy = evaluation(self, test_batch_size, False)
            self.model.eval()
            accuracy = capsnet_evaluation(self, 50, False)
            self.model.train()
            # report accuracy of model now and evaluate if the current trial should prune
            #trial.report(accuracy, epoch)
            #if trial.should_prune():
            #    raise optuna.exceptions.TrialPruned()
            self.reset_epoch()
        if(self.save_weights):
            self.save_weights("Classifier")

    def train(self, number_of_epochs, number_of_batches, test_batch_size):
        loss_list = []
        acc_list = []
        for epoch in range(number_of_epochs):
            for step in range(number_of_batches):
                correct = 0
                for i in range(self.batch_size):
                    # Run the forward pass
                    im = self.batch_images[str(i)]
                    output = self.model(im)
                    
                    label = self.batch_labels[str(i)]
                    label = torch.Tensor([label])
                    if self.cuda:   
                        label = label.cuda().to(self.device)
                    # TODO change to tensor in load_images
                    loss = self.criterion(output, label)
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Track the accuracy
                    predicted = torch.round(output.data[0])
                    if torch.argmax(predicted) == torch.argmax(label):
                        correct += 1
                    acc_list.append(correct / self.batch_size)

                if (i + 1) % self.batch_size == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, number_of_epochs, step + 1, number_of_batches, loss.item(),
                                  (correct / self.batch_size) * 100))
                self.reset_batches()
                self.load_images()

            # Evaluate accuracy of model after this epoch
            # TODO: implement limit of testing size
            accuracy = evaluation(self, test_batch_size, False)
            # report accuracy of model now and evaluate if the current trial should prune
            #trial.report(accuracy, epoch)
            #if trial.should_prune():
            #    raise optuna.exceptions.TrialPruned()
            self.reset_epoch()
        if(self.save_weights):
            self.save_weights("Classifier")

def capsnet_evaluation(Classifier, test_batch_size, prnt):
    tulip = glob.glob("Test_Flowers/tulip/*")
    sunflower = glob.glob("Test_Flowers/sunflower/*")
    rose = glob.glob("Test_Flowers/rose/*")
    dandelion = glob.glob("Test_Flowers/dandelion/*")
    daisy = glob.glob("Test_Flowers/daisy/*")
    paths = {"tulip": tulip, "sunflower": sunflower,
             "rose": rose, "dandelion": dandelion, "daisy": daisy}
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
        output, _, masked = Classifier.model(batch_images[str(i)])
        predicted = torch.argmax(masked)
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
        print("Errors in dandelion images:",
              errors[1], "out of", test_batch_size)
        print("Errors in rose images:", errors[2], "out of", test_batch_size)
        print("Errors in sunflower images:",
              errors[3], "out of", test_batch_size)
        print("Errors in tulip images:", errors[4], "out of", test_batch_size)

        print("-----------------------ERRORS-----------------------")
        for error in error_results:
            print(error, ":", error_results[error])
    # Returning accuracy for tuning of the model
    return accuracy


def evaluation(Classifier, test_batch_size, prnt):
    tulip = glob.glob("Test_Flowers/tulip/*")
    sunflower = glob.glob("Test_Flowers/sunflower/*")
    rose = glob.glob("Test_Flowers/rose/*")
    dandelion = glob.glob("Test_Flowers/dandelion/*")
    daisy = glob.glob("Test_Flowers/daisy/*")
    paths = {"tulip": tulip, "sunflower": sunflower,
             "rose": rose, "dandelion": dandelion, "daisy": daisy}
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
        print("Errors in dandelion images:",
              errors[1], "out of", test_batch_size)
        print("Errors in rose images:", errors[2], "out of", test_batch_size)
        print("Errors in sunflower images:",
              errors[3], "out of", test_batch_size)
        print("Errors in tulip images:", errors[4], "out of", test_batch_size)

        print("-----------------------ERRORS-----------------------")
        for error in error_results:
            print(error, ":", error_results[error])
    # Returning accuracy for tuning of the model
    return accuracy


def objective(trial):

    cfg = {
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
    TClassifier.train(trial, cfg["epochs"],
                      cfg["step_size"], cfg["test_batch_size"])
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
        'trial_results', filename + '.pkl')
    print(len(glob.glob('\trial_result*')))
    joblib.dump(study, file_path)
    # Save a copy as a csv file
    data_frame = study.trials_dataframe()
    csv_path = pathlib.Path().absolute().joinpath(
        'trial_results', 'csv', filename + '.csv')
    data_frame.to_csv(csv_path)


def read_study_from_file(filename):
    # read results from file with name=filename and return the study object
    file_path = pathlib.Path().absolute().joinpath('trial_results', filename)
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
    cfg = {
        "image_size": (28, 28),                                          
        "learning_rate": 6.34192248576476e-05,
        "mini_batch_size": 32,
        "test_batch_size": 150,
        "step_size": 32,
        "epochs": 200,
        # trial.suggest_categorical('dropout', [True, False]),
        "dropout": True,
        "dropout_rate": 0.4,
        "prnt": False,
        "optimizer": optim.Adam,
        "criterion": nn.MSELoss(),
        "save_weights": True
    }
    # To conduct a study with n number of trials as parameter, comment this if you only want to read a
    TClassifier = Classifier(cfg)
    #TClassifier.load_weights('classifier')
    TClassifier.load_images()
    accuracy = capsnet_evaluation(TClassifier, 50, cfg["prnt"])
    TClassifier.train_capsulenet(cfg["epochs"],
                      cfg["step_size"], cfg["test_batch_size"])
    #study = read_study_from_file("classifier_study_0.pkl")
    EvClassifier = Classifier(cfg)
    EvClassifier.copy_weights(TClassifier)
    accuracy = evaluation(EvClassifier, cfg["test_batch_size"], cfg["prnt"])

    