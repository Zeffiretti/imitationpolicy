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
    file = r'dual-result.txt'
    device = torch.device('cuda')
    data_set = dataset.DemonstrationData(file, index=idx, time_scale=1000)
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


def train_policy():
    file = r'dual-result.txt'
    # log file
    log_file = open(r'param/log_policy.txt', 'w')
    device = torch.device('cuda')
    data_set = dataset.ImitationData(file, demo_no=1, time_scale=1000)
    # data_set = dataset.DemonstrationData(file, index=0, time_scale=1000)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True)
    policy_model = model.PolicyNet(input=1000).to(device)
    # mse loss
    criterion = torch.nn.MSELoss().to(device)
    # adam optimizer
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.000001)
    # train policy model
    policy_model.load_state_dict(torch.load(r'param/policy_model_19.pth'))
    policy_model.train()
    bc_models = []
    for i in range(16):
        bc_model = model.BehaviorCost(input=1000)
        bc_model = bc_model.to(device)
        bc_model.load_state_dict(torch.load(r'param/bc_model_{}.pth'.format(i)))
        bc_model.eval()
        bc_models.append(bc_model)

    for epoch in range(20, 30):
        policy_stack = data_set.init_policy().to(device)
        err = 0
        for i, (demo_stack, demo_y, policy_y) in enumerate(data_loader):
            # move to device
            demo_stack, demo_new, policy_y = demo_stack.to(device), demo_y.to(device), policy_y.to(device)
            # combine two stacks
            training_data = torch.cat((demo_stack, policy_stack), dim=2)
            # calculate ideal policy (maybe)
            with torch.set_grad_enabled(False):
                bc_stack = torch.cat([bc_model(training_data) for bc_model in bc_models]).mean(dim=1)
                bc_stack = bc_stack.view(1, -1)
            # print bc_stack shape
            # print('bc_stack shape', bc_stack.shape)

            with torch.set_grad_enabled(True):
                policy_new = policy_model(training_data)
                # print policy_new shape
                # print('policy_new shape', policy_new.shape)
                state_new = torch.cat((demo_new, policy_new), dim=-1)
                # print state_new shape
                # print('state_new shape', state_new.shape)
                # calculate policy loss
                policy_loss = criterion(state_new, bc_stack)
                # backward propagation
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
                policy_new.detach_()
                state_new.detach_()
                # update policy stack
                policy_stack = torch.cat((policy_stack, policy_new.view(1, 1, -1)), dim=1)
                policy_stack = policy_stack[:, 1:, :]
                err = err * i / (i + 1) + policy_loss / (i + 1)
                if i % 1000 == 0:
                    print('[epoch:', epoch, ',i:', i, '],loss:', err.item())
                    log_file.write('[epoch:{},loss:{}]\n'.format(epoch, err.item()))
                if i % 10000 == 0:
                    torch.save(policy_model.state_dict(), r'param/policy_temp/policy_model_{}_{}.pth'.format(epoch, i))
        print('[epoch:', epoch, '],loss:', err.item())
        log_file.write('[epoch:{},loss:{}]\n'.format(epoch, err.item()))
        torch.save(policy_model.state_dict(), r'param/policy_model_{}.pth'.format(epoch))
    log_file.close()


def generate_data():
    """
    generate data using policy model
    :return: write data into file 'policy.txt'
    """
    file = r'dual-result.txt'
    # log file
    # policy_file = open(r'policy.txt', 'w')
    device = torch.device('cuda')
    data_set = dataset.ImitationData(file, demo_no=1, time_scale=1000)
    # data_set = dataset.DemonstrationData(file, index=0, time_scale=1000)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True)
    policy_model = model.PolicyNet(input=1000).to(device)
    policy_model.load_state_dict(torch.load(r'param/policy_model_29.pth'))
    policy_model.eval()
    policy_stack = data_set.init_policy().to(device)
    with torch.no_grad():
        for i, (demo_stack, demo_y, policy_y) in enumerate(data_loader):
            # move to device
            demo_stack, demo_new, policy_y = demo_stack.to(device), demo_y.to(device), policy_y.to(device)
            # combine two stacks
            training_data = torch.cat((demo_stack, policy_stack[:, -1000:, :]), dim=2)
            # calculate ideal policy (maybe)
            policy_new = policy_model(training_data)
            policy_new = policy_new.view(1, -1)
            # print policy_new shape
            # print('policy_new shape', policy_new.shape)
            # update policy stack
            policy_stack = torch.cat((policy_stack, policy_new.view(1, 1, -1)), dim=1)
            # policy_stack = policy_stack[:, 1:, :]
            if i % 100 == 0:
                # print policy new
                print('policy_new at', i, ':', policy_new.item())
    np.savetxt(r'policy.txt', policy_stack.view(-1, 8).cpu().detach().numpy() / 30, fmt='%f')

    # policy_file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1st step: train behavior cost net for idx=0:16
    for idx in range(16):
        train_bc(idx)
    # 2nd step: train policy net
    train_policy()
    # 3rd step: generate data
    generate_data()

