#! /home/ruiruizhong/miniconda3/envs/pytorch/bin/python

import random
import torch
import numpy as np
import tqdm as tqdm
import os
import time
from utils.params import configs
from utils.util import set_global_seed, train_data_loader, test_data_loader, train_val_sample, weights_init
from model import AutoEncoder


str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id

def main():
    print('Configs:', configs)

    device = torch.device(configs.device)

    # set global seed
    # set_global_seed(configs.seed)

    # load test data
    all_source = [3, 9, 11, 12, 13, 14, 19, 20, 25]
    test_loader = test_data_loader(configs.test_source)
    train_source = list(set(all_source) - set([configs.test_source]))
    print(f'Test source: {configs.test_source}, Train source: {train_source}')
    train_dataset, val_dataset = train_data_loader(train_source, configs)

    # load model
    model = AutoEncoder().to(device)
    model.apply(weights_init)

    # print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    # define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    # check if folders exist
    if not os.path.exists(f'./save_model/{configs.test_source}'):
        os.makedirs(f'./save_model/{configs.test_source}')
    if not os.path.exists(f'./log/{configs.test_source}'):
        os.makedirs(f'./log/{configs.test_source}')


    training_log = []
    validation_log = []
    best_val = float('inf')
    

    # training
    for epoch in range(1, configs.num_epochs + 1):
        t_start = time.time()
        # load data
        set_global_seed(epoch)
        train_loader, val_loader = train_val_sample(train_dataset, val_dataset, configs)
        model.train()
        loss_t = 0
        for i, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            loss_t += loss.item()
        
        scheduler.step()
        training_log.append(loss_t / len(train_loader))
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, lr: {lr} Training Loss: {loss_t / len(train_loader)}, Time: {time.time() - t_start}')

        # validation
        if epoch % configs.log_interval == 0 or epoch == 1:
            model.eval()
            loss_v = 0
            for i, (data, _) in enumerate(val_loader):
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                loss_v += loss.item()

            loss_v /= len(val_loader)
            validation_log.append(loss_v)
            print(f'Validation Loss: {loss_v}')

            if loss_v < best_val:
                best_val = loss_v
                torch.save(model.state_dict(), f'./save_model/{configs.test_source}/best_model.pth')

    # save training log
    np.save(f'./log/{configs.test_source}/training_log.npy', np.array(training_log))
    np.save(f'./log/{configs.test_source}/validation_log.npy', np.array(validation_log))

    # save the last model
    torch.save(model.state_dict(), f'./save_model/{configs.test_source}/last_model.pth')


if __name__ == '__main__':
    main()