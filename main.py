
import torch
import torch.nn as nn

from network.models import LR, FM, Wide_Deep, DeepFM
from preprocess.data_process import process, load_data
from utils.train import train

# 采用 Kaggle 上的 Criteo_1M_with_nans.csv 数据集
data_path = "data/Criteo_1M_with_nans.csv"
batch_size = 256
num_workers = 4 # 使用子进程数
dim = 8 # Embedding 层的维度
lr = 5e-4
weight_decay = 5e-3
epochs = 20
early_stop_patience = 3

if __name__ == "__main__":

    # 加载数据
    train_df, valid_df, test_df, all_cat_cols, dense_cols, encoders = process(data_path)
    train_iter, valid_iter, test_iter = load_data(train_df, valid_df, test_df, all_cat_cols, dense_cols, batch_size, num_workers)
    emb_size_of_every_col = [len(encoders[col]) + 1 for col in all_cat_cols] # 每个特征列特征值的最大值（这一列 Embedding 的大小）

    '''# LR 定义
    model = LR(emb_size_of_every_col, len(all_cat_cols), len(dense_cols))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=1,
        threshold=1e-4,
        min_lr=1e-6
    )
    train(model, criterion, optimizer, scheduler, epochs, train_iter, valid_iter, test_iter, early_stop_patience)'''

    # FM 定义
    model = FM(emb_size_of_every_col, len(dense_cols), dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=1,
        threshold=1e-4,
        min_lr=1e-6
    )
    train(model, criterion, optimizer, scheduler, epochs, train_iter, valid_iter, test_iter, early_stop_patience)

    '''# Wide&Deep 定义
    model = Wide_Deep(emb_size_of_every_col, len(all_cat_cols), len(dense_cols), dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=1,
        threshold=1e-4,
        min_lr=1e-6
    )
    train(model, criterion, optimizer, scheduler, epochs, train_iter, valid_iter, test_iter, early_stop_patience)'''

    '''# DeepFM 定义
    model = DeepFM(emb_size_of_every_col, len(all_cat_cols), len(dense_cols), dim=dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',          # 监控指标越大越好（AUC）
        factor=0.5,          # lr 每次乘以 0.5
        patience=1,          # 1 个 epoch 没提升就衰减
        threshold=1e-4,      # 小于 threshold 不算提升
        min_lr=1e-6          # 学习率最小值
    )
    train(model, criterion, optimizer, scheduler, epochs, train_iter, valid_iter, test_iter, early_stop_patience)'''
