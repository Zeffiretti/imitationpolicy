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

    def __init__(self, input=1000, hidden1=1024, hidden2=512, hidden3=64):
        super(PolicyNet, self).__init__()
        # normilize the input
        self.norm = torch.nn.LayerNorm(16)
        # flatten layer: [b,100,16] -> [b,100*16]
        self.flatten = torch.nn.Flatten()
        # 1st hidden layer: 200 relu units, [b,100*16] -> [b,200]
        self.hidden1 = torch.nn.Linear(in_features=input * 16, out_features=hidden1)
        # 2nd hidden layer: 50 relu units, [b,200] -> [b,50]
        self.hidden2 = torch.nn.Linear(hidden1, hidden2)
        # 3rd hidden layer: 50 relu units, [b,50] -> [b,16]
        self.hidden3 = torch.nn.Linear(hidden2, hidden3)
        # output layer: 1 relu units, [b,16] -> [b,1]
        self.output = torch.nn.Linear(hidden3, 8)
        # combine all layers into sequential model with ReLu, and dropout as p=0.5
        self.model = torch.nn.Sequential(self.norm,
                                         self.flatten,
                                         self.hidden1,
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         self.hidden2,
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         self.hidden3,
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         self.output)

    # forward pass
    def forward(self, x):
        return self.model(x)
