import numpy as np # for numerical computing
import pandas as pd # for manipulation and analysis
import os # for interacting with the operating system 
import matplotlib.pyplot as plt # for creating visualizations
from PIL import Image # for working with images
import torch # for machine learning
import torch.nn as nn # for building neural networks in PyTorch
import torch.optim as optim # for optimizing neural networks in PyTorch
import torchvision # for computer vision tasks in PyTorch
from torchvision import models, transforms # for loading and preprocessing image data
import torchvision.models as models # for using pre-trained models in PyTorch
from torch.utils.data import DataLoader, Dataset # for working with datasets in PyTorch
import torchview

# # 导入数据
# train = pd.read_csv('./data/train.csv')
# test = pd.read_csv('./data/test.csv')

# # 查看数据
# print(train.head())
# print(test.head())

# 配置文件地址（父文件夹两个点；同文件夹一个点；字文件夹加目录）
train_csv = './data/train.csv'
test_csv = './data/test.csv'

#如果有GPU可用，使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义用于读取数据的类，继承PyTorch的Dataset类
class manualDataset(Dataset):
    # 初始化方法，split用于标识当前是训练集还是测试集，val_split用于标识是否划分验证集
    def __init__(self, file_path, split='train', val_split=False):
        df = pd.read_csv(file_path)
        self.split = split
        self.val_split = val_split
        if self.split=='test' and val_split==False:
            # 如果是测试集且不划分验证集，则直接读取所有测试数据，将原来的一张图片一行的形式转换为28*28的矩阵
            # 最后增加一个维度，变成1*28*28的形式
            self.raw_files = df.values.reshape(-1, 28, 28).astype(np.uint8)[:,:,:,None]
            self.labels = None
        else:
            # 如果是训练集且不划分验证集
            # 将原来的一张图片一行的形式转换为28*28的矩阵，最后增加一个维度，变成1*28*28的形式，然后转换为torch.Tensor
            # 数据从第1列开始读，标签在第0列
            self.raw_files = df.iloc[:,1:].values.reshape(-1, 28, 28).astype(np.uint8)[:,:,:,None]
            self.labels = torch.from_numpy(df.iloc[:,0].values)

        # 如果要划分验证集，选择前10个数据作为验证集
        if(val_split):
            self.raw_files = self.raw_files[:10]

        # 定义图像的预处理转换操作，采用PyTorch提供的transforms
        # ToPIlImage将数组转换为PIL图片，RandomHorizontalFlip随机翻转图片，ToTensor将PIL图片转换为Tensor
        # PIL 是 Python 的一个图像处理库，提供了许多图像处理相关的功能
        # 这些转换操作用于数据增强，transforms.Compose将图像变换操作组合在一起
        self.train_transform = transforms.Compose([ transforms.ToPILImage(),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        self.test_transform = transforms.Compose([ transforms.ToTensor()])

        # 返回数据集中的元素数量
    def __len__(self):
        return len(self.raw_files)
    
    # 按索引读取数据集中的元素
    def __getitem__(self, idx):
        raw = self.raw_files[idx]
        
        # 如果是训练集或者验证集，返回图像和标签
        if self.split == 'train' or self.val_split:
            raw = self.train_transform(raw)
            label = self.labels[idx]
            return raw, label
        
        # 如果是测试集，返回图像
        elif self.split == 'test':
            raw = self.test_transform(raw)
            return raw

# 创建训练集、测试集和验证集
train_dataset = manualDataset(train_csv, split='train', val_split=False)
test_dataset = manualDataset(test_csv, split='test', val_split=False)
val_dataset = manualDataset(train_csv, split='test', val_split=True)    # 根据manualDataset的代码，这里的split参数不影响结果

# 定义训练数据和测试数据batch_size
train_batch_size = 32
test_batch_size = 1

# 创建训练集和测试集的DataLoader
# DataLoader是 PyTorch 中的一个数据加载器类，它可以将manualDataset中的数据以所需的批量大小、随机顺序等方式进行加载
# shuffle=True表示打乱数据顺序，增强模型的泛化能力
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size)

# 创建一个卷积神经网络
# 继承自它继承自nn.Module(troch.nn)类，可以使用 PyTorch 提供的自动求导机制进行训练和优化
class CNN(nn.Module):
    def __init__(self):
        super().init
