import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from utils import Characters, Stages


class TwitchClipDataset(Dataset):

    def __init__(self, csv_file, clip_dir, transforms=None):
        """
        Args:
            csv_file (string): Path to csv file containing info about clips such as name, frame_max, frame_delay.
            clip_dir (string): Directory containing frames.
            transforms (callable, optional): Optional transform for frames.
        """
        self.clips_frames = self.expand_clips(csv_file)
        self.clip_dir = clip_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.clips_frames)

    def __getitem__(self, idx):
        frame_name = os.path.join(self.clip_dir, 
                                  self.clips_frames.iloc[idx, 0])
        frame = Image.open(frame_name)
        if self.transforms:
            frame = self.transforms(frame)
        chars = torch.tensor(self.clips_frames.iloc[idx, 1:3].values.astype(np.float32)).long()
        stage = self.clips_frames.iloc[idx, 3]
        return frame, chars, stage

    # Expand csv file to contain several frames per clip
    def expand_clips(self, csv_file):
        """
        Args:
            csv_file (string): Path to csv file containing info about clips. Assumed column structure:
            Name    FrameMax   FrameDelay Character1  Character2  Stage
        Returns:
            clips_frames (DataFrame): DataFrame with expanded info. Assumed column structure:
            Name    Character1  Character2  Stage
        """
        info_frame = pd.read_csv(csv_file, sep="\t")
        clips_list = []

        for i in range(len(info_frame)):
            row = info_frame.iloc[i]
            clip = pd.DataFrame(columns=['Name', 'Character1', 'Character2', 'Stage'])
            # Map characters and stage to proper index for training
            char1 = row['Character1']
            char2 = row['Character2']
            stage = row['Stage']
            new_char1 = Characters.toIndexPartial[char1]
            new_char2 = Characters.toIndexPartial[char2]
            new_stage = Stages.toIndexPartial[stage]
            # find frame max and delay to create individual frames
            frame_max = row['FrameMax']
            frame_delay = row['FrameDelay']
            for frame in range(0, frame_max + frame_delay, frame_delay):
                new_row = row.drop(['FrameMax', 'FrameDelay'])
                new_row['Name'] = row['Name'] + "_frame{}.jpg".format(frame)
                new_row['Character1'] = new_char1
                new_row['Character2'] = new_char2
                new_row['Stage'] = new_stage
                clip = clip.append(new_row, ignore_index=True)
            clips_list.append(clip)

        clips_frames = pd.concat(clips_list, ignore_index=True)
        return clips_frames

