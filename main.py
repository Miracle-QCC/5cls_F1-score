import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import Module
import torch.optim as optim
from torch import nn

def eval_model(all_labels, all_predictions):
    f1_macro = f1_score(all_labels, all_predictions, average='weighted')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')

    print(f"F1 Score: {f1_macro:.4f}")
    print(f"Micro F1 Score: {f1_micro:.4f}")

def get_mean_std(data):
    # 计算每一列的均值和标准差，同时处理NaN值
    mean_values = np.nanmean(data, axis=0)
    std_values = np.nanstd(data, axis=0)
    return mean_values,std_values

def norm_data(data, m, s):
    normalized_data = (data - m) / s
    return normalized_data

def replace_nan(data, column_means):
    for i in range(data.shape[1]):
        # 获取当前列的均值
        mean_value = column_means[i]

        # 找到当前列中为 NaN 的位置
        nan_positions = np.isnan(data[:, i])

        # 用均值替换 NaN
        data[nan_positions, i] = mean_value
    return data

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x,y

if __name__ == '__main__':
    train_file = 'train.csv'
    test_file = 'test.csv'
    val_file = 'val.csv'
    # 打开CSV文件
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    val_data = pd.read_csv(val_file)

    train_data_np = train_data.T[1:].values.transpose()
    test_data_np = test_data.T[1:].values.transpose()
    val_data_np = val_data.T[1:].values.transpose()

    ###################### 处理缺失值 ##############

    train_data_np_clear = np.where(train_data_np != '?',train_data_np,np.NAN).astype(np.float32)
    train_X = train_data_np_clear[:,:-1]
    # 得到均值和方差
    m, s = get_mean_std(train_X)

    ### 用均值填充
    train_X = replace_nan(train_X, m)
    train_Y = train_data_np_clear[:,-1].astype(np.int64) - 1 ### 原标签是[1,2,3,4,5]，现改为[0,1,2,3,4]
    # 对预测输入进行标准化
    train_X_norm = norm_data(train_X, m, s)

    test_data_np_clear = np.where(test_data_np != '?',test_data_np, np.NAN).astype(np.float32)
    test_X = test_data_np_clear[:, :-1]
    ### 用均值填充
    test_X = replace_nan(test_X, m)
    # 对预测输入进行标准化
    test_X_norm = norm_data(test_X, m, s)
    # test_Y = test_data_np_clear[:, -1]

    val_data_np_clear = np.where(val_data_np != '?', val_data_np, np.NAN).astype(np.float32)
    # 对预测输入进行标准化
    val_X = val_data_np_clear[:, :-1]
    ### 用均值填充
    val_X = replace_nan(val_X, m)
    val_X_norm = norm_data(val_X, m, s)
    val_Y = val_data_np_clear[:, -1].astype(np.int64) - 1

    ### 创建dataloader
    train_dataset = CustomDataset(train_X,train_Y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = CustomDataset(val_X, val_Y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


    ### 创建模型
    net = Module(train_X.shape[1], 5).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    #
    ### 训练
    epochs = 300
    # for e in tqdm(range(epochs)):
    for e in range(epochs):
        total_train_loss = []
        net.train()
        for x,y in train_loader:
            # print(x)
            x = x.cuda()
            y = y.squeeze().cuda()

            y_pred = net(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())

        ## 验证loss是否下降
        with torch.no_grad():
            total_val_loss = []
            net.eval()
            for x, y in val_loader:
                # print(x)
                x = x.cuda()
                y = y.squeeze().cuda()

                y_pred = net(x)
                loss = criterion(y_pred, y)

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                total_val_loss.append(loss.item())
        train_m_loss = np.mean(total_train_loss)
        val_m_loss = np.mean(total_val_loss)
        print(f"{e}/{epochs} train loss:{train_m_loss}, val loss:{val_m_loss}")


    torch.save(net.state_dict(), 'model.pth')
    net.eval()

    ### 计算F1值
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        total_val_loss = []
        net.eval()
        for x, y in val_loader:
            # print(x)
            x = x.cuda()
            y = y.squeeze().cuda()

            y_pred = net(x)
            predictions = torch.argmax(y_pred, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    eval_model(all_labels, all_predictions)
    labels_pred = []
    ### 对test进行预测
    for idx, x in enumerate(test_X):
        x_input = torch.from_numpy(x).cuda()
        y_pred = torch.argmax(net(x_input))
        # test_data[idx,'class label'] = y_pred.item() + 1
        labels_pred.append(y_pred.item() + 1) ## +1是还原标签
    ### 修改回原来的csv，并保存为新的csv
    test_data = pd.concat([test_data, pd.Series(labels_pred, name='class label')], axis=1)

    test_data.to_csv('modified_test1.csv', index=False)
