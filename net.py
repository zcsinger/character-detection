import torch.nn as nn


class CharacterStageNet(nn.Module):

    def __init__(self, channels_in, num_chars, num_stages, num_filters=8):
        super(CharacterStageNet, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, num_filters, 3, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=2)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, dilation=2)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, dilation=3)
        self.conv5 = nn.Conv2d(num_filters, num_filters, 3, dilation=2)

        self.fc1 = nn.Linear(408, 64)
        self.fc_char1 = nn.Linear(64, num_chars)
        self.fc_char2 = nn.Linear(64, num_chars)
        self.fc_stage = nn.Linear(64, num_stages)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.bn4 = nn.BatchNorm2d(num_filters)        
        self.bn5 = nn.BatchNorm2d(num_filters)

    def forward(self, x):

        x1 = self.bn1(self.relu(self.conv1(x)))
        x2 = self.bn2(self.relu(self.conv2(x1)))
        x3 = self.bn3(self.relu(self.conv3(x2)))
        x4 = self.bn4(self.relu(self.conv4(x3)))
        x5 = self.bn5(self.relu(self.conv5(x4)))

        x_fc = x5.view(-1, 408)
        x_fc1 = self.relu(self.fc1(x_fc))

        x_char1 = self.fc_char1(x_fc1)
        x_char2 = self.fc_char2(x_fc1)
        x_stage = self.fc_stage(x_fc1)

        return x_char1, x_char2, x_stage
