import torch
import numpy as np
from scipy import io #用于输入输出
import pandas as pd 
from torch.utils.data import Dataset, DataLoader #处理数据
from sklearn import preprocessing #进行数据的预处理
from sklearn.preprocessing import MultiLabelBinarizer #处理多标签分类任务
import wfdb #加载和处理 ECG 信号数据
import ast #用于可视化 ECG 信号或结果
import matplotlib.pyplot as plt
class PTBXLDataset(Dataset): #用以加载和处理 PTB-XL 数据集，这样处理的数据可以被直接用来训练 PyTorch 模型或者进行测试
    def __init__(self, path,sampling_rate = 500, train_set =True, transform=None): #transform: 可选的信号变换函数，用于对数据进行预处理
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+'/'+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+'/'+f) for f in df.filename_hr] #wfdb.rdsamp 函数用于读取心电信号数据
            data = np.array([signal for signal, meta in data])
            return data


        Y = pd.read_csv(path+'\ptbxl_database.csv', index_col='ecg_id') #标签数据从 ptbxl_database.csv 文件中加载，并解析为字典
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x)) #返回一个集合，包含与当前记录相关的诊断类别

        # 导入原始数据
        X = load_raw_data(Y, sampling_rate, path)
        agg_df = pd.read_csv(path+'\scp_statements.csv', index_col=0) #index_col=0 表示使用文件的第一列作为 DataFrame 的索引。
        agg_df = agg_df[agg_df.diagnostic == 1] #只记录有限的数据

        def aggregate_diagnostic(y_dic): #用于将每个 ECG 记录的详细诊断代码（y_dic）转换为诊断超类。
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # 应用诊断超类
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic) #存储每个记录的诊断超类

        # 拆分成训练集与测试集
        test_fold = 10
        # 训练
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # 测试
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        if train_set:
            self.data = np.array(X_train)
            labels = list(y_train)
        else:
            self.data = np.array(X_test)
            labels = list(y_test)
        
        label_mapping = {"NORM": 0, "CD": 1, "HYP": 2, "MI": 3, "STTC": 4}    
        self.label = [[label_mapping[label] for label in sample] for sample in labels] #将诊断超类转换为数字标签。

        mlb = MultiLabelBinarizer()
        self.onehot_label = mlb.fit_transform(self.label)
        self.onehot_label = torch.tensor(self.onehot_label, dtype=torch.float32)

        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0,2,1)
        self.data = torch.nan_to_num(self.data, nan=0.0)
        self.X=X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] 
        onehot_label = self.onehot_label[idx]
        return data,onehot_label
    
#读取数据
path = r"c:\Users\TJ\Desktop\统计软件课程代码\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
sampling_rate = 100
#创建数据集
train_dataset = PTBXLDataset(path,sampling_rate=100,train_set=True) 
test_dataset = PTBXLDataset(path,sampling_rate=100,train_set=False)

#创建数据加载器
fine_tune_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #用于训练的 DataLoader，设置了批量大小为 64，并且打乱了数据
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#ECG 绘制第五千个信号波形图
#plt.rcParams['figure.figsize'] = (20.0, 10.0) 
#plt.figure()
#plt.plot(train_dataset.X[5000][:,0],color='red', linewidth=1.1)
#plt.grid(linestyle='--')
# plt.yticks([])
#plt.show()

import torch.nn as nn
# 五层经典卷积网络
class CNN_1D_5L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 16, (3,), stride=1, padding=4),
            nn.BatchNorm1d(16),#批量归一化
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.AvgPool1d(2,stride=2)#每次池化操作在输入中选择2个值，计算它们的平均值
        )
                
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 16, (5,), stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),#丢弃部分神经元，防止过拟合
            nn.AvgPool1d(2,stride=2)#池化操作
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, (5,), stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.AvgPool1d(2,stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, (5,), stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.AvgPool1d(2,stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.AvgPool1d(2,stride=2)
        )

    
        self.linear1 = nn.Linear(128, 5) #全连接层，128个输入，5个输出

        
    def forward(self, x):#神经网络的前向传播过程 
        x = x.view(-1, 12, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x=torch.mean(x,dim=-1)#将特征图从一个二维的形状（[batch_size, 128, 特征图长度]）转化为一维的形状（[batch_size, 128]）
        return self.linear1(x)#每行包含一个样本在5个类别上的预测分数
        
#评价函数和训练函数
from torch import nn as nn#专门用于构建神经网络的模块
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
import datetime
import torch.nn.functional as F


#评价函数
def evaluate_multilabel_classification(model, dataloader, device, criterion):
    model.eval()
    all_labels = []
    all_predictions = []# 用于存储所有的真实标签和预测结果
    total_loss = 0.0#记录累积损失
    total_samples = 0#记录处理的样本总数

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)#将原始输出转化为 [0, 1] 之间的概率

            loss = criterion(outputs, labels.float())#计算当前批次的损失

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    average_loss = total_loss / total_samples#通过累加的损失和总样本数计算平均损失

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    # 计算每一个标签的AUROC
    auroc_list = []
    for i in range(all_labels.shape[1]):
        auroc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
        auroc_list.append(auroc)

    # 计算平均的AUROC
    auroc_macro = np.mean(auroc_list)

    #计算每一个标签的F1分数
    f1_list = []
    for i in range(all_labels.shape[1]):
        f1 = f1_score(all_labels[:, i], (all_predictions[:, i] > 0.5).astype(int))
        f1_list.append(f1)

    #所有类别的平均 F1 分数 
    f1_macro = np.mean(f1_list)

    return {"Loss": average_loss, "AUROC per class": auroc_list, "AUROC Macro": auroc_macro, "F1 per class": f1_list, "F1 Macro": f1_macro}
#训练函数
def train_model_ptbxl(train_loader, test_loader, model, criterion, optimizer, num_epochs, device='cpu',evaluate=True):
    model.to(device) 
    train_loss = [] 
    test_loss=[]
    macro_auc=[]
    f1_auc=[]
    for epoch in range(num_epochs):
        model.train()#启用训练模式
        total_loss = 0.0
        start_time = datetime.datetime.now() 
        for inputs, labels in train_loader:#从 train_loader 中获取批次数据，并将数据和标签移动到指定设备上
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))#将输入数据传入模型，计算输出，并使用损失函数计算损失
            loss.backward()
            optimizer.step()#计算梯度并更新模型参数
            total_loss += loss.item()#叠加损失
        end_time = datetime.datetime.now() 
        if evaluate:
            model.eval()
            result = evaluate_multilabel_classification(model, test_loader, device, criterion)
            formatted_result = {
                "Loss": "{:.4f}".format(result["Loss"]),
                "AUROC per class": ["{:.4f}".format(auroc) for auroc in result["AUROC per class"]],
                "AUROC Macro": "{:.4f}".format(result["AUROC Macro"]),
                "F1 per class": ["{:.4f}".format(f1) for f1 in result["F1 per class"]],
                "F1 Macro": "{:.4f}".format(result["F1 Macro"])
            }#格式化评估结果
            train_loss.append("{:.4f}".format(total_loss))
            test_loss.append(formatted_result["Loss"])
            macro_auc.append(formatted_result["AUROC Macro"])
            f1_auc.append(formatted_result["F1 Macro"])#记录当前 epoch 的结果
            print(f"Epoch [{epoch + 1}/{num_epochs}]","train_loss:", "{:.4f}".format(total_loss),"test_Loss:", formatted_result["Loss"],
                "AUROC Macro:", formatted_result["AUROC Macro"],
                "F1 Macro:", formatted_result["F1 Macro"], "Time:", end_time - start_time)  

    return train_loss,test_loss,macro_auc,f1_auc
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#检测是否有可用的 GPU
device

#加载模型以及训练参数
model = CNN_1D_5L(1000)
model.to(device)

lr = 0.001#决定了每次迭代时，模型参数沿着梯度方向的调整幅度
optimizer = torch.optim.AdamW(model.parameters(), lr = lr,weight_decay = 1e-2)# 使用 AdamW 优化器，并设置权重衰减（L2正则化）为 0.01
criterion = nn.MultiLabelSoftMarginLoss()
epochs = 20
#训练和评价模型
train_loss,test_loss,macro_auc,f1_auc=train_model_ptbxl(fine_tune_loader, test_loader, model, criterion, optimizer,epochs, device=device)
model.eval()

torch.save(model.state_dict(), 'model_weights_ptbxl.pth')#训练完成后，保存模型的权重到 model_weights_ptbxl.pth 文件中

result = evaluate_multilabel_classification(model, test_loader, device, criterion)
formatted_result = {
    "Loss": "{:.4f}".format(result["Loss"]),
    "AUROC per class": ["{:.4f}".format(auroc) for auroc in result["AUROC per class"]],
    "AUROC Macro": "{:.4f}".format(result["AUROC Macro"]),
    "F1 per class": ["{:.4f}".format(f1) for f1 in result["F1 per class"]],
    "F1 Macro": "{:.4f}".format(result["F1 Macro"])
}

# 输出结果
print("Formatted Result:")
print("Loss:", formatted_result["Loss"])
print("AUROC per class:", formatted_result["AUROC per class"])
print("AUROC Macro:", formatted_result["AUROC Macro"])
print("F1 per class:", formatted_result["F1 per class"])
print("F1 Macro:", formatted_result["F1 Macro"])