import numpy as np
import os
import glob
import random
import torch
import rasterio
import torch.nn as nn

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # WARNING: if cudnn.enabled=False => greatly reduces training/inference speed.
    torch.backends.cudnn.enabled = True

def load_source(source):
    x_list = []
    y_list = []
    
    for i in source:
        x =np.load(f'./dataset/A{i:003}/x.npy')
        y =np.load(f'./dataset/A{i:003}/y.npy')
        x_list.append(x)
        y_list.append(y)

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    x = x / 255.0
        
    return x, y


def train_data_loader(train_source, config):
    """
    Load data from the given path
    :param train_source: the source of training data
    """
    # Load data
    train_x, train_y = load_source(train_source)

    total_train_samples = len(train_x)
    val_size = int(total_train_samples * config.val_ratio)
    indices = list(range(total_train_samples))
    val_indices, train_indices = indices[:val_size], indices[val_size:]

    val_x, val_y = train_x[val_indices], train_y[val_indices]
    train_x, train_y = train_x[train_indices], train_y[train_indices]

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_x, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32))

    return train_dataset, val_dataset

def train_val_sample(train_dataset, val_dataset, config):
    # Split train data into train and validation sets

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader

def test_data_loader(test_source):
    """
    Load data from the given path
    :param test_source: the source of testing data
    :param batch_size: the batch size for data loading
    :return: the loaded data
    """
    # Load data
    test_source = [test_source]
    test_x, test_y = load_source(test_source)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)