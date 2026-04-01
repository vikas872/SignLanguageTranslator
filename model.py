import torch
import torch.nn as nn
import torch.nn.functional as F

class ASLNet(nn.Module):
    def __init__(self, num_classes=25):
        super(ASLNet, self).__init__()
        # Input shape: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # After two pools of 2x2, the 28x28 image becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # (32, 14, 14)
        x = self.pool(F.relu(self.conv2(x))) # (64, 7, 7)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
