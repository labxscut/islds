# 数据加载与预处理
import torch
import numpy as np
from scipy import io
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
import torchaudio
from pathlib import Path
import wfdb
import os
import math
from torchvision import transforms
import ast

'''
定义 PTBXLDatasetReport 类:
这是一个继承自 PyTorch Dataset 的自定义数据集类，用于加载和处理 PTB-XL 数据集，并生成报告文本
'''
class PTBXLDatasetReport(Dataset):
    def __init__(self, path,sampling_rate=100, train_set =True, transform=None):     # 初始化
        # 根据采样率加载原始ECG信号数据，并将读取的数据转换为np数组
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        # 加载标注数据Y，信号数据X
        path = 'path/to/ptbxl/'
        sampling_rate=100
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        X = load_raw_data(Y, sampling_rate, path)
        # 将存储为字符串的字典（scp_codes）转换为实际字典格式
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # 加载并聚合诊断信息
        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        # 定义诊断信息聚合函数
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key]['SCP-ECG Statement Description'])
            return ','.join(list(set(tmp)))

        # 应用诊断聚合函数
        Y['SCP-ECG Statement Description'] = Y.scp_codes.apply(aggregate_diagnostic)
        
        # 分割数据集
        test_fold = 10

        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)]['SCP-ECG Statement Description']

        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold]['SCP-ECG Statement Description']

        if train_set:
            self.data = np.array(X_train)
            self.labels = list(y_train)
        else:
            self.data = np.array(X_test)
            self.labels = list(y_test)

        self.text = []
        for i in range(len(labels)):
            if len(labels[i]) > 1:
                sentence = f'An ECG of {", ".join(labels[i][:-1])}, and {labels[i][-1]}'
            elif len(labels[i]) ==1:
                sentence = f'An ECG of {labels[i][0]}'
            else:
                sentence = 'An unlabeled ECG'
            self.text.append(sentence)
        
        """
        # 假设 self.label 是一个包含多个标签的列表
        label_mapping = {"NORM": 0, "CD": 1, "HYP": 2, "MI": 3, "STTC": 4}


        # 对每个标签进行编码
        # print(labels[50])
        # print(labels[51])
        
        self.label = [[label_mapping[label] for label in sample] for sample in labels]

        mlb = MultiLabelBinarizer()
        self.onehot_label = mlb.fit_transform(self.label)
        self.onehot_label = torch.tensor(self.onehot_label, dtype=torch.float32)
        """
        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0,2,1)
        self.data = torch.nan_to_num(self.data, nan=0.0)
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] 
        text = self.text[idx]
        #label = self.label[idx]
        #label = self.label[idx]
        #onehot_label = self.onehot_label[idx]
        return data,text
dataset = PTBXLDatasetReport('path/to/PTB-XL',sampling_rate=500)