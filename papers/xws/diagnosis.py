# Data science libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
# Pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

# Others
from IPython.core.debugger import set_trace
from pathlib import Path
import os

import sys

sys.path.append(r"D:\本科项目资料\故障诊断科研")

from helper import get_df_all, download
from train_helper import get_dataloader, fit, validate
import nn_model
from data_urls import URLS


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


setup_seed(123)

working_dir = Path(r"D:\本科项目资料\故障诊断科研")
DATA_PATH = Path(working_dir/'Data')
save_model_path = Path(working_dir/'model')
DE_path = DATA_PATH / '12k_DE'

for path in [DATA_PATH, save_model_path]:
    if not path.exists():
        path.mkdir(parents=True)

bs = 64
lr = 0.001
wd = 1e-5
betas=(0.99, 0.999)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random_seed = 42
df_all = get_df_all(DE_path, segment_length=500, normalize=True)
features = df_all.columns[2:]
target = 'label'
len(features)
print(torch.cuda.device_count())
print(device)
X_train, X_valid, y_train, y_valid = train_test_split(df_all[features],
                                                      df_all[target],
                                                      test_size=0.50, random_state=random_seed, shuffle=True
                                                     )
## Create DataLoader of train and validation set
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_valid = torch.tensor(y_valid.values, dtype=torch.long)
X_train=X_train.to(device)
X_valid=X_valid.to(device)
y_train=y_train.to(device)
y_valid=y_valid.to(device)

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)
train_dl, valid_dl = get_dataloader(train_ds, valid_ds, bs)
## Instantiate model, optimizer and loss function
model = nn_model.CNN_1D_2L_new(len(features))
model.to(device)
opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
loss_func = CrossEntropyLoss().to(device)


## Train
epochs = 20
model, metrics = fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False)


## Train
epochs = 20
model, metrics = fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False)

torch.save(model.state_dict(), "model.pth")
model2 = nn_model.CNN_1D_2L_new(len(features))
model2.load_state_dict(torch.load("model.pth"))
model2.eval()


print(validate(model, valid_dl, loss_func))

