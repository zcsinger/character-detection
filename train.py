import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


import matplotlib.pyplot as plt

from dataset import TwitchClipDataset
from net import CharacterStageNet

from utils import char_to_names, stage_to_names

"""============================ GLOBAL VARIABLES ==============================="""

BATCH_SIZE = 10
BASE_LR = 0.001
WEIGHT_DECAY = 0.0001

NUM_CHARS = 5 
NUM_STAGES = 4
NUM_FILTERS = 8

NUM_EPOCHS = 20
DISPLAY_ITER = 10

data_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((72, 128)),
        transforms.ToTensor()
        ])

"""============================ TRAINING DATA ==============================="""

print("Creating training set...")
training_set = TwitchClipDataset(csv_file='clips/train_csv.csv',
                                 clip_dir='clips/',
                                 transforms=data_transforms)

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

print("Creating testing set...")
testing_set = TwitchClipDataset(csv_file='clips/test_csv.csv',
                                 clip_dir='clips/',
                                 transforms=data_transforms)

testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)
num_test_batches = len(testing_loader)


"""============================ MODEL PARAM ==============================="""

print("Creating model...")
model = CharacterStageNet(channels_in=1, num_chars=NUM_CHARS, num_stages=NUM_STAGES, num_filters=NUM_FILTERS)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

"""============================ TRAINING  ==============================="""

print("Beginning training...")
for epoch in range(NUM_EPOCHS):
    print("Epoch {}:".format(epoch))
    for i, data in enumerate(training_loader, 0):
        frames, labels_char, labels_stage = data
        labels_char1 = labels_char[:, 0]
        labels_char2 = labels_char[:, 1]

        outputs_char1, outputs_char2, outputs_stage = model(frames)
        loss = criterion(outputs_char1, labels_char1) + \
               criterion(outputs_char2, labels_char2) + \
               criterion(outputs_stage, labels_stage)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % DISPLAY_ITER == 0:
            print("Loss at iteration {}: {:.6f}".format(i, loss.item()))


    for i, valid_batch in enumerate(testing_loader, 0):
        valid_frames, valid_char, valid_stage = valid_batch
        valid_char1 = valid_char[:, 0]
        valid_char2 = valid_char[:, 1]

        valid_outputs_char1, valid_outputs_char2, valid_outputs_stage = model(valid_frames)
        char1_names = char_to_names(valid_outputs_char1)
        char2_names = char_to_names(valid_outputs_char2)
        stage_names = stage_to_names(valid_outputs_stage)

        results = list(zip(char1_names, char2_names, stage_names))
        print(results)
        
        """
        valid_loss = criterion(valid_outputs_char1, valid_char1) + \
                     criterion(valid_outputs_char2, valid_char2) + \
                     criterion(valid_outputs_stage, valid_stage)
        print("Validation Loss for batch {}: {:.6f}".format(i, valid_loss.item()))
        """

print("Success.")
