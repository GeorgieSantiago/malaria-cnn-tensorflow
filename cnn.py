import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, lr, input_params, output_params, fc1_dims):
        super(CNN, self).__init__()
        self.input_params = input_params
        self.output_params = output_params
        self.cv1 = nn.Conv2d(3, 32, 3, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(0.4)
        self.cv2 = nn.Conv2d(32, 32, 3, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.cv3 = nn.Conv2d(32, 64, 6, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(13312, 6000)
        self.fc2 = nn.Linear(6000, output_params)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    def forward(self, x):
        x = T.Tensor(x).to(self.device)
        x = F.relu(self.bn1(self.cv1(x)))
        x = F.relu(self.bn2(self.cv2(x)))
        x = F.relu(self.bn3(self.cv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x