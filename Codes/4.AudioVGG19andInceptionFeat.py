

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



# Video image feature extractor
inception_v3 = models.inception_v3(pretrained=True)


# Audio feature extractor
vgg19 = models.vgg19(pretrained=True)



k = 2
epochs = 1
batch_size = 1
learning_rate = 1e-4
log_interval = 1
minFrames = 100
img_x1, img_y1 = 299, 299
img_x2, img_y2 = 224, 224

begin_frame, end_frame, skip_frame = 0, minFrames, 0




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



import pickle
with open(FOLDER_NAME+'final_allNewData.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

# train, test split
train_list, train_label= allDataAnnotation['train']
val_list, val_label  =  allDataAnnotation['val']
test_list, test_label  =  allDataAnnotation['test']


# In[27]:


allVidList = []
allVidLab = []

allVidList.extend(train_list)
allVidList.extend(val_list)
allVidList.extend(test_list)

allVidLab.extend(train_label)
allVidLab.extend(val_label)
allVidLab.extend(test_label)




# image transformation
transform1 = transforms.Compose([transforms.Resize([img_x1, img_y1]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

transform2 = transforms.Compose([transforms.Resize([img_x2, img_y2]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

selected_frames = np.arange(begin_frame, end_frame).tolist()




def read_images(path, selected_folder, use_transform):
    X = []
    currFrameCount = 0
    videoFrameCount = len([name for name in os.listdir(os.path.join(path, selected_folder))])
    if videoFrameCount <= minFrames:
        for i in range(videoFrameCount):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        if use_transform is not None:
            paddingImage = use_transform(paddingImage)
        while currFrameCount < minFrames:
            X.append(paddingImage.squeeze_(0))
            currFrameCount+=1
        X = torch.stack(X, dim=0)
    else:
        step = int(videoFrameCount/minFrames)
        for i in range(0,videoFrameCount,step):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        if use_transform is not None:
            paddingImage = use_transform(paddingImage)
        while currFrameCount < minFrames:
            X.append(paddingImage.squeeze_(0))
            currFrameCount+=1
        X = torch.stack(X, dim=0)

    return X



def read_audio(path, selected_folder, use_transform):
    X = []
    path = os.path.join(path, selected_folder+'.png')
    X_audio = use_transform(Image.open(path))
    X.append((X_audio[:3,:,:]).squeeze_(0))
    X = torch.stack(X, dim=0)
    return X




# set path
data_image_path = FOLDER_NAME + "/Dataset_Images/"   
data_audio_path = FOLDER_NAME + "/Audio/Audio_plots/"


# In[25]:


from tqdm import tqdm

X_Video = []
X_Audio = []


inception_v3.eval()
vgg19.eval()

# inception_v3 = inception_v3.to(device)
# vgg19 = vgg19.to(device)

for folder in tqdm(allVidList):
    video = read_images(data_image_path, folder, transform1)
    audio = read_audio(data_audio_path, folder, transform2)

    video_features = torch.tensor(inception_v3(video.to(device)))

    U, S, V = torch.pca_lowrank(video_features.view(-1,1), center = True)
    video_features = torch.matmul(video_features.view(-1,1), V[:, :num_video_features])
    video_features = video_features.view(-1).tolist()

    audio_features = vgg19(audio.to(device))
    U, S, V = torch.pca_lowrank(audio_features.view(-1,1), center = True)
    audio_features = torch.matmul(audio_features.view(-1,1), V[:, :num_audio_features])
    audio_features = audio_features.view(-1).tolist()
    
    del video
    del audio
    

    X_Video.append(video_features)
    X_Audio.append(audio_features)






for i in zip(allVidList, X_Video):
    vidFeatureMap[i[0]]=i[1]
    
with open('inception_vidFeatures.p', 'wb') as fp:
    pickle.dump(vidFeatureMap, fp)





audFeatureMap = {}

for i in zip(allVidList, X_Audio):
    audFeatureMap[i[0]]=i[1]
    
with open('vgg19_audFeatureMap.p', 'wb') as fp:
    pickle.dump(audFeatureMap, fp)
