
import torch
from torch import nn
from torch.nn import functional as F
# 2 Layers CNN
class CNN_1D_2L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool1d(2,stride=2)
        )
        
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.AvgPool1d(2,stride=2)
        )
        
        self.linear1 = nn.Linear(self.n_in*128 //4, 4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.n_in*128//4)
        return self.linear1(x)


# 3 Layers CNN
class CNN_1D_3L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
                
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, (5,), stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
        
        self.linear1 = nn.Linear(self.n_in*128 //8, 4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.n_in*128//8)
        return self.linear1(x)
        
        
class CNN_1D_2L_new(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            #nn.MaxPool1d(2,stride=2)
        )
        
        self.linear1 = nn.Linear(458, 458)
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            #nn.AvgPool1d(2,stride=2)
        )
        
        self.linear2 = nn.Linear(64*419, 512)
        self.linear3=nn.Linear(512,4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        #print('===debug===',x.shape)
        x = self.layer1(x)
        #print('===debug===x==',x.shape)
        x1=nn.AvgPool1d(2,stride=2)(x)      
        x2=nn.AvgPool1d(4,stride=4)(x)
        x3=nn.AvgPool1d(6,stride=6)(x)
        x4=torch.cat([x1,x2,x3],dim=-1)
        #print('===debug===x4==',x4.shape)
        x4=self.linear1(x4)
        x = self.layer2(x4)
        #print('===debug===x.layer2==',x.shape)
        x5=nn.AvgPool1d(2,stride=2)(x4)
        x6=nn.AvgPool1d(4,stride=4)(x4)
        x7=nn.AvgPool1d(6,stride=6)(x4)
        x8=torch.cat([x5,x6,x7],dim=-1)
        #print('===debug===x8==',x8.shape)
        
        x = x8.view(-1, 64*419)
        x=self.linear2(x)
        x = nn.ReLU()(x)
        
        return self.linear3(x)
    
    
class CNN_1D_2L_wind(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer11 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool1d(2,stride=2)
        )
        self.layer12 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool1d(2,stride=2)
        )
        self.layer13 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool1d(2,stride=2)
        )
        self.linear1 = nn.Linear(458, 458)
        self.layer21 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.AvgPool1d(2,stride=2)
        )
        self.layer22 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.AvgPool1d(2,stride=2)
        )
        self.layer23 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.AvgPool1d(2,stride=2)
        )
        
        self.linear2 = nn.Linear(128*113, 512)
        self.linear3=nn.Linear(512,4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x1=nn.AvgPool1d(2,stride=2)(x)
        x2=nn.AvgPool1d(4,stride=4)(x)
        x3=nn.AvgPool1d(6,stride=6)(x)
        #print('===debug===',x.shape)
        x1 = self.layer11(x1)
        x1 = self.layer21(x1)
        x2 = self.layer12(x2)
        x2 = self.layer22(x2)
        x3 = self.layer13(x3)
        x3 = self.layer23(x3)
        
        x4=torch.cat([x1,x2,x3],dim=-1)
        #print('===debug===x4==',x4.shape)
        
        #print('===debug===x.layer2==',x.shape)
        
        #print('===debug===x8==',x8.shape)
        
        x = x4.view(-1, 128*113)
        x=self.linear2(x)
        x = nn.ReLU()(x)
        return self.linear3(x)