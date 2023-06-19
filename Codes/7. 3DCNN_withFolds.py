
FOLDER_NAME = '../../'

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
from tqdm import tqdm
from sklearn.metrics import *
import random

def fix_the_random(seed_val = 2021):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

fix_the_random(2021)


## ---------------------- Dataloader ---------------------- ##
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, minFrames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.minFrames = minFrames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        currFrameCount = 0
        videoFrameCount = len([name for name in os.listdir(os.path.join(path, selected_folder))])
        if videoFrameCount <= minFrameCount:
          for i in range(videoFrameCount):
              image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i))).convert('L')         # Convert method convert RGB image into black and white image

              if use_transform is not None:
                  image = use_transform(image)

              X.append(image.squeeze_(0))
              currFrameCount += 1
              if(currFrameCount==minFrameCount):
                break

          paddingImage = Image.fromarray(np.zeros((100,100)), '1')
          if use_transform is not None:
            paddingImage = use_transform(paddingImage)
          while currFrameCount < self.minFrames:
            X.append(paddingImage.squeeze_(0))
            currFrameCount+=1
          X = torch.stack(X, dim=0)
        else:
          step = int(videoFrameCount/minFrameCount)
          for i in range(0,videoFrameCount,step):
              image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i))).convert('L')         # Convert method convert RGB image into black and white image

              if use_transform is not None:
                  image = use_transform(image)

              X.append(image.squeeze_(0))
              currFrameCount += 1
              if(currFrameCount==minFrameCount):
                break

          paddingImage = Image.fromarray(np.zeros((100,100)), '1')
          if use_transform is not None:
            paddingImage = use_transform(paddingImage)
          while currFrameCount < self.minFrames:
            X.append(paddingImage.squeeze_(0))
            currFrameCount+=1
          X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]
        try:
            # Load data
            X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
            y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor
        except:
            return None
        return X, y

## -------------------- (reload) model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=2):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2], self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = binary-clasification

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

## --------------------- end of 3D CNN module ---------------- ##

def evalMetric(y_true, y_pred):
   accuracy = accuracy_score(y_true, y_pred)
   mf1Score = f1_score(y_true, y_pred, average='macro')
   f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred))
   fpr, tpr, _ = roc_curve(y_true, y_pred)
   area_under_c = auc(fpr, tpr)
   recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred))
   precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred))
   return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
           'precision': precisionScore, 'recall': recallScore})

# set path
data_path = FOLDER_NAME + "Dataset_Images/"    # define UCF-101 spatial data path
save_model_path = FOLDER_NAME + "Models/Conv3D_ckpt/"  # save Pytorch models

# 3D CNN parameters
fc_hidden1, fc_hidden2 = 128, 128
dropout = 0.0        # dropout probability

# training parameters
k = 2            # number of target category
epochs = 20
batch_size = 10
learning_rate = 1e-4
log_interval = 1
minFrameCount = 100
img_x, img_y = 100, 125  # resize video 2d frame size
#img_x, img_y = 224, 224  # resize video 2d frame size

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 0, minFrameCount, 0

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y, weight=torch.FloatTensor([0.41, 0.59]).to(device))
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        metrics = evalMetric(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(metrics)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))

    return losses, scores

def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    pred_prob = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            pred_prob.extend(np.array(output.cpu()))

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    # try:
    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    # except:
    #   metrics = None

    # show information
    print('\nTest set: ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(
                len(all_y), test_loss, 100 * metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))
    # # save Pytorch models of best record
    # torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pt'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pt'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, metrics, list(all_y_pred.cpu().data.squeeze().numpy()), pred_prob

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
params2 = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}



with open(FOLDER_NAME+'allFoldDetails.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)




# image transformation
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

selected_frames = np.arange(begin_frame, end_frame).tolist()
                                                                                                                                                                                                                                                                                                                                                                        

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


allF = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']


finalOutputAccrossFold ={}


for fold in allF:
    # train, test split
    train_list, train_label= allDataAnnotation[fold]['train']
    val_list, val_label  =  allDataAnnotation[fold]['val']
    test_list, test_label  =  allDataAnnotation[fold]['test']


    train_set = Dataset_3DCNN(data_path, train_list, train_label, selected_frames, minFrameCount, transform=transform)
    valid_set = Dataset_3DCNN(data_path, val_list, val_label,   selected_frames, minFrameCount, transform=transform)
    test_set = Dataset_3DCNN(data_path, test_list, test_label, selected_frames, minFrameCount, transform=transform)
    train_loader = data.DataLoader(train_set, collate_fn = collate_fn, **params)
    valid_loader = data.DataLoader(valid_set, collate_fn = collate_fn, **params2)
    test_loader = data.DataLoader(test_set, collate_fn = collate_fn, **params2)


    # create model
    cnn3d = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y, drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2, num_classes=k).to(device)



    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn3d = nn.DataParallel(cnn3d)

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=learning_rate)   # optimize all cnn parameters


    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    validFinalValue = None
    testFinalValue = None
    finalScoreAcc =0
    prediction  = None

    # start training
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, cnn3d, device, train_loader, optimizer, epoch)
        test_loss, test_scores, veTest_pred , veTest_prob= validation(cnn3d, device, optimizer, test_loader)
        test_loss1, test_scores1 , veValid_pred, veValid_prob = validation(cnn3d, device, optimizer, valid_loader)
        if (test_scores1['mF1Score']>finalScoreAcc):
            finalScoreAcc = test_scores1['mF1Score']
            validFinalValue = test_scores1
            testFinalValue = test_scores
            prediction = {'test_list': test_list , 'test_label': test_label, 'test_pred': veTest_pred}

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(list(x['accuracy'] for x in train_scores))
        epoch_test_losses.append(test_loss)
        epoch_test_scores.append(test_scores['accuracy'])


        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
    finalOutputAccrossFold[fold] = {'validation':validFinalValue, 'test': testFinalValue, 'test_prediction': prediction}


with open('foldWiseRes_3dcnn.p', 'wb') as fp:
    pickle.dump(finalOutputAccrossFold,fp)
        
allValueDict ={}
for fold in allF:
    for val in finalOutputAccrossFold[fold]['test']:
        try:
            allValueDict[val].append(finalOutputAccrossFold[fold]['test'][val])
        except:
            allValueDict[val]=[finalOutputAccrossFold[fold]['test'][val]]



import numpy as np
for i in allValueDict:
    print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

