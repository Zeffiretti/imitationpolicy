# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
import torch
import model
import dataset
from torch.utils.data import Dataset, DataLoader


def train_bc(idx):
    j = 0
    file = r'dual-result.txt'
    device = torch.device('cuda')
    data_set = dataset.InteractionData(file, index=idx, time_scale=1000)
    data_loader = DataLoader(data_set, batch_size=1000, shuffle=False, pin_memory=True)
    bc_model = model.BehaviorCost(input=1000)
    bc_model = bc_model.to(device)
    # L1 loss
    criterion = torch.nn.L1Loss().to(device)
    # adam optimizer
    optimizer = torch.optim.Adam(bc_model.parameters(), lr=0.001)
    # iter loop
    bc_model.train()
    epochs = 500
    log_file = open(r'param/log_{}.txt'.format(idx), 'w')
    for epoch in range(epochs):
        err = 0
        for i, (data_x, data_y) in enumerate(data_loader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred = bc_model(data_x)
            loss = criterion(pred, data_y)
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            err = err + loss
        print('[idx', idx, 'epoch:', epoch, '],loss:', err.item())
        log_file.write('[idx{},epoch:{},loss:{}]\n'.format(idx, epoch, err.item()))

    torch.save(bc_model.state_dict(), r'param/bc_model_{}.pth'.format(idx))
    log_file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for idx in range(8, 17):
        train_bc(idx)

    # a = torch.rand((10, 1), dtype=torch.float32)
    # b = a + 0.5
    # # a = a.view(-1, 1)
    # loss = torch.nn.MSELoss()
    # error = loss(a, b)
    # # show error
    # print('error is', error.item())
