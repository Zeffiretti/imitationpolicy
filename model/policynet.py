"""
 * File Name : 
 * Author    : ZEFFIRETTI, HESH
 * College   : Beijing Institute of Technology
 * E-Mail    : zeffiretti@bit.edu.cn, hiesh@mail.com
"""

# import required packages
import numpy as np
import matplotlib.pyplot as plt
import torch


class PolicyNet(torch.nn.Module):
    """
    PolicyNet: 4-hidden layer MLP
    input shape: [b,100,16]
    flatten layer [b,100,16] -> [b,100*16]
    1st hidden layer: [b,1600] -> [b, 640]
    2nd hidden layer: [b,640] -> [b, 160]
    3rd hidden layer: [b,160] -> [b, 16]
    output layer: [b,16] -> [b,8]
    """

    def __init__(self):
        super(PolicyNet, self).__init__()
        # flatten layer
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(1600, 640)
        self.fc2 = torch.nn.Linear(640, 160)
        self.fc3 = torch.nn.Linear(160, 16)
        self.fc4 = torch.nn.Linear(16, 8)
        # combine all layers into sequential model with ReLu
        self.model = torch.nn.Sequential(self.flatten,
                                         self.fc1,
                                         torch.nn.ReLU(),
                                         self.fc2,
                                         torch.nn.ReLU(),
                                         self.fc3,
                                         torch.nn.ReLU(),
                                         self.fc4)

    # forward pass
    def forward(self, x):
        return self.model(x)
