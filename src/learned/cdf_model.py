import torch
import torch.nn as nn
import torch.nn.functional as F


class CDFNet(nn.Module):
    # Learns to approximate the Cumulative Query Distribution Function (CDF).
    # 32-bit binary representation of an integer key, Output: Probability [0.0, 1.0]
    def __init__(self, input_dim=1, hidden_dim=128):
        super(CDFNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Sigmoid forces output to be between 0 and 1
        return torch.sigmoid(self.fc3(x)).squeeze()
