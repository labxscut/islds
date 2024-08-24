from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np
import random
from DataLoader import PTBXLDatasetReport
from MSDNN import MSDNN

'''
定义clinicalBERT类：
这是一个自定义的神经网络模型类，基于预训练的BERT模型来处理文本数据;
BERT模型的参数被冻结，以避免在训练过程中更新权重
'''
class clinicalBERT(nn.Module):
    # 初始化
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将文本转换为模型可处理的张量格式
        self.tokenizer = AutoTokenizer.from_pretrained("path/to/Bio_ClinicalBERT")    
        self.bert = BertModel.from_pretrained("path/to/Bio_ClinicalBERT").to(self.device)
        # 一个线性层，将BERT模型的输出（768维）转换为128维的特征表示
        self.predictor = nn.Linear(768, 128).to(self.device)
        for param in self.bert.parameters():         # 冻结参数，即参数在训练过程中不会被更新
                param.requires_grad = False
    
    # 定义了模型的前向传播过程，首先通过BERT提取特征，然后通过线性层进一步转换
    def forward(self, text):
        with torch.no_grad():       # 关闭梯度计算
            # 张量化
            x = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device) 
            # 使用在初始化时加载的BERT模型处理输入的文本数据     
            outputs = self.bert(**x)
            # 提取池化输出
            x = outputs['pooler_output']
        # 通过自定义线性层进行预测
        x = self.predictor(x)
        return x

# 创建实例化对象
model = clinicalBERT()

'''
定义ConvBlock类：
定义了一个卷积块，包含一个1D卷积层、批归一化层和激活层，用于处理ECG信号数据。
'''
class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel, stride = 1):      # (self, 输入通道数，输出通道数，卷积操作的步幅)
        super().__init__()
        # 定义一维卷积层，卷积核大小为3
        # padding表示在输入的每一端添加一个像素的填充，使输出的宽度与输入一致。避免缩小数据尺寸
        self.conv = nn.Conv1d(channel_in, channel, kernel_size=3, padding=1, stride=stride)
        # 定义批归一化层
        self.norm = nn.BatchNorm1d(channel)
        # 定义激活层
        self.activate = nn.ReLU()
    def forward(self, x):               # 定义前向传播方法
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)   
        return x

'''
定义simple_CNN类：
一个简单的卷积神经网络，用于ECG数据分类。由多个卷积块组成，并最终通过一个线性层输出分类结果。
'''
class simple_CNN(nn.Module):
    def __init__(self, num_classes=5, init_channels=12, channel = 16 ,layers = 3 ,drop_out_rate=0.2):  #初始化
        super(simple_CNN, self).__init__()
        # 定义了一个由多个卷积块组成的层序列
        self.layers = nn.Sequential(
                ConvBlock(init_channels, channel, stride = 2)
            )
        # 循环添加额外的卷积块，直到达到指定的层数
        for _ in range(layers - 1):
            self.layers.append(ConvBlock(channel, channel, stride = 2))
        # 定义全连接层
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x):    # 定义前向传播方法
        x = self.layers(x) 
        x = torch.mean(x, dim=-1)  
        x = self.fc(x)  

        return x


'''
定义CLIP_ECG类：
将心电图（ECG）和文本数据结合起来进行处理和比较
''' 
class CLIP_ECG(nn.Module):
    def __init__(self,device=None,ecg_encoder = simple_CNN):    # 初始化
        super(CLIP_ECG, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 文本编码器
        self.text_encoder = clinicalBERT(device=device)
        # ECG编码器
        self.ecg_encoder = ecg_encoder(num_classes=128).to(self.device)
        # dropout层
        self.uma = nn.Dropout(0.1)

    def train_forward(self, ecg, text):     # 定义训练前向传播方法
        # 使用ECG编码器处理输入的ECG数据，提取其特征
        ecg_feature = self.ecg_encoder(ecg)
        # 对提取的ECG特征应用 Dropout，生成一个新的特征，用于稍后的比较
        ecg_feature2 = self.uma(ecg_feature)
        # 对ECG特征进行L2归一化，将其转换为单位向量，以便于特征间的比较。
        ecg_feature = F.normalize(ecg_feature, p=2, dim=-1)
        ecg_feature2 = F.normalize(ecg_feature2, p=2, dim=-1)
        # 使用文本编码器处理输入的文本数据，提取其特征，然后进行L2归一化
        text_feature = self.text_encoder(text)
        text_feature = F.normalize(text_feature, p=2, dim=-1)
        # 返回两个相关性矩阵，一个是ECG和文本特征之间的，另一个是ECG特征与其经过Dropout后的版本之间的。
        return ecg_feature@text_feature.T, ecg_feature@ecg_feature2.T

    def forward(self, ecg, text):     # 定义前向传播方法
        ecg_feature = self.ecg_encoder(ecg)
        ecg_feature = F.normalize(ecg_feature, p=2, dim=-1)
        text_feature = self.text_encoder(text)
        text_feature = F.normalize(text_feature, p=2, dim=-1)
        # 只返回ECG和文本特征的相关性矩阵，用于模型的推理输出。
        return ecg_feature@text_feature.T


        

        # 转换数据格式并处理缺失值
        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0,2,1)
        self.data = torch.nan_to_num(self.data, nan=0.0)
        
    # 获取数据集长度
    def __len__(self):
        return len(self.data)

    # 获取数据集样本
    def __getitem__(self, idx):
        data = self.data[idx] 
        text = 'The report of the ECG is that {' + self.labels[idx] + '}'

        return data,text




# 数据加载与模型训练
import torch.nn.functional as F

# 实例化对象，创建编码器，加载数据
ecg_encoder = simple_CNN(num_classes=128)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 设置训练参数，开始训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIP_ECG(device=None)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001,weight_decay = 1e-3)
temperature = 0.07

epochs = 50
for epoch in range(epochs):
    for ecg,text in data_loader:
        num = len(text)
        logits1 , logits2 = model.train_forward(ecg.to(device),text)
        labels = torch.arange(0,num).to(device)
        # 计算交叉熵损失（ECG 与文本）
        loss_i = F.cross_entropy(logits1 / temperature,labels)
        # 计算交叉熵损失（文本与 ECG）
        loss_t = F.cross_entropy(logits1.T / temperature,labels)
        # 计算交叉熵损失（ECG 与经过 dropout 的 ECG）
        loss_uma = F.cross_entropy(logits2 /temperature,labels)
        # 合并损失
        loss = (loss_i+loss_t)/2 + loss_uma
        # 清除梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        #print(outputs.shape)
    print('epoch:',epoch+1)


# 定义测试数据集类，并进行测试
from sklearn.metrics import roc_auc_score, accuracy_score
class PTBXLDatasetTest(Dataset):
    def __init__(self, path,sampling_rate = 100, train_set =False, transform=None):
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        path = 'path/to/ptbxl/'

        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        X = load_raw_data(Y, sampling_rate, path)

        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # 超类诊断
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        test_fold = 10

        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        
        if train_set:
            self.data = np.array(X_train)
            labels = list(y_train)
        else:
            self.data = np.array(X_test)
            labels = list(y_test)
        # 假设 self.label 是一个包含多个标签的列表
        label_mapping = {"NORM": 0, "CD": 1, "MI": 2,"HYP": 3, "STTC": 4}
        # 对每个标签进行编码  
        self.label = [[label_mapping[label] for label in sample] for sample in labels]

        mlb = MultiLabelBinarizer()
        self.onehot_label = mlb.fit_transform(self.label)
        self.onehot_label = torch.tensor(self.onehot_label, dtype=torch.float32)

        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0,2,1)
        self.data = torch.nan_to_num(self.data, nan=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] 

        onehot_label = self.onehot_label[idx]
        return data,onehot_label
    
# predict2
def predict(model,dataset,labels):
    pred_probs = []
    true_labels = []
    model.eval()
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for ecg,onehot_labels in data_loader:
            logits = model(ecg.to(device),labels)
            prob = F.softmax(logits, dim=-1)
            # prob = F.sigmoid(logits)
            pred_probs.append(prob.cpu().numpy())
            true_labels.append(onehot_labels.cpu().numpy())

    pred_probs = np.vstack(pred_probs)
    true_labels = np.vstack(true_labels)
    
    # 计算 AUC
    auc_scores = []
    for i in range(true_labels.shape[1]):
        auc = roc_auc_score(true_labels[:, i], pred_probs[:, i])
        auc_scores.append(auc)
    
    # 计算准确率
    pred_labels = (pred_probs > 0.5).astype(int)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print("AUC scores:", auc_scores)
    print("Accuracy:", accuracy)

labels = ['The ECG of normal ECG, a type of diagnostic',
          'The ECG of Conduction Disturbance, a type of diagnostic',
          'The ECG of Myocardial Infarction, a type of diagnostic',
          'The ECG of Hypertrophy, a type of diagnostic',
          'The ECG of ST/T change, a type of diagnostic']

test_dataset = PTBXLDatasetTest('path/to/PTBXL',sampling_rate=500)
predict(model,test_dataset,labels)


def predict2(model,dataset,labels):
    pred_probs = []
    true_labels = []
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for ecg,onehot_labels in data_loader:
            logits = model(ecg.to(device),labels)
            prob = F.softmax(logits, dim=-1)
            #prob = F.sigmoid(logits)
            pred_probs.append(prob.cpu().numpy())
            true_labels.append(onehot_labels.cpu().numpy())
            #print(onehot_labels)
            #print(prob)
    pred_probs = np.vstack(pred_probs)
    true_labels = np.vstack(true_labels)
    
    acc_num = 0
    # 计算 AUC
    for i in range(len(pred_probs)):
        pred =  np.argmax(pred_probs[i])
        if true_labels[i][pred]==1:
            acc_num  =  acc_num  + 1
    print(round(acc_num/len(true_labels),4))


predict2(model,test_dataset,labels)

# predict 3
import random
class randomTextSet(Dataset):
    def __init__(self, path,sampling_rate = 100, train_set =False, transform=None):
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        path = 'path/to/ptbxl/'

        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        X = load_raw_data(Y, sampling_rate, path)

        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # 超类诊断
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        test_fold = 10 
        y = Y.diagnostic_superclass
        
        self.data = np.array(X)
        labels = list(y)
        # 假设 self.label 是一个包含多个标签的列表
        label_mapping = {"NORM": 0, "CD": 1, "MI": 2,"HYP": 3, "STTC": 4}
        # 对每个标签进行编码
              
        self.label = [[label_mapping[label] for label in sample] for sample in labels]

        single_label_index = [[] for _ in range(5)] 
        for i in range(len(self.label)):
            if len(self.label[i])== 1:
                single_label_index[self.label[i][0]].append(i)
        selected_samples = []
        self.selected_labels = []
        for i, lst in enumerate(single_label_index):
            # 随机选择200个元素，可以使用random.sample方法
            selected = random.sample(lst, 200)
            selected_samples.extend(selected)
            self.selected_labels.extend([i]*200)


        self.data = X[selected_samples]

        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0,2,1)
        self.data = torch.nan_to_num(self.data, nan=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] 

        label = self.selected_labels[idx]
        return data,label

random_test_dataset = randomTextSet('path/to/',sampling_rate=500)
print(len(random_test_dataset))
from sklearn.metrics import classification_report
def predict3(model,dataset,labels):
    pred_probs = []
    true_labels = []
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for ecg,ecg_labels in data_loader:
            logits = model(ecg.to(device),labels)
            prob = F.softmax(logits, dim=-1)
            #prob = F.sigmoid(logits)
            pred_probs.append(prob.cpu().numpy())
            true_labels.append(ecg_labels.cpu().numpy())

    pred_probs = np.vstack(pred_probs)
    true_labels = np.vstack(true_labels)
    y_pred = pred_probs.argmax(axis=1)
    report = classification_report(true_labels, y_pred)

    # 打印分类报告
    print(report)

predict3(model,random_test_dataset,labels)