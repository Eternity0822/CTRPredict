
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# 所有整数特征的列名
int_cols = ['intCol_0', 'intCol_1', 'intCol_2', 'intCol_3', 'intCol_4',
            'intCol_5', 'intCol_6', 'intCol_7', 'intCol_8', 'intCol_9', 
            'intCol_10', 'intCol_11', 'intCol_12']
# 类别特征的列名
cat_cols = ['catCol_0', 'catCol_1', 'catCol_2', 'catCol_3', 'catCol_4', 
            'catCol_5', 'catCol_6', 'catCol_7', 'catCol_8', 'catCol_9', 
            'catCol_10', 'catCol_11', 'catCol_12', 'catCol_13', 'catCol_14', 
            'catCol_15', 'catCol_16', 'catCol_17', 'catCol_18', 'catCol_19', 
            'catCol_20', 'catCol_21', 'catCol_22', 'catCol_23', 'catCol_24', 'catCol_25']
# 标签
label_col = 'target'

def process(path):

    df = pd.read_csv(path)
    # 划分 80% 训练集 10% 验证集 10% 测试集，random_state 为随机种子，数字相同则划分相同
    # stratify 传入标签那一列，保证训练集、测试集中标签占比，和原始数据集一致
    train_df, tmp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    # tmp_df 再对半划分，得到 10% 验证集 10% 测试集
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42, stratify=tmp_df[label_col])

    # ----------------- 填充整数特征列的 nan -----------------
    int_median = {} # 保存每列中位数
    for col in int_cols: # 整数特征用中位数来填充缺失值
        median = train_df[col].median() # 只用训练集计算中位数
        int_median[col] = median
        train_df[col] = train_df[col].fillna(median).astype(int)
        valid_df[col] = valid_df[col].fillna(median).astype(int)
        test_df[col] = test_df[col].fillna(median).astype(int)
    
    # ----------------- 对整数列取 log1p 并将所有整数列标准化 -----------------
    log_mean, log_std = {}, {} # 保存每列的 mean 以及 std
    for col in int_cols:
        # 对数变换 log1p: log(1 + x)
        train_df[col + '_log'] = np.log1p(train_df[col].clip(lower=0).astype(np.float32))
        valid_df[col + '_log'] = np.log1p(valid_df[col].clip(lower=0).astype(np.float32))
        test_df[col + '_log'] = np.log1p(test_df[col].clip(lower=0).astype(np.float32))
        # clip(lower=0) 把所有负值压到 0

        # 用训练集计算 mean 和 std
        mean = train_df[col + '_log'].mean()
        std = train_df[col + '_log'].std()
        if std == 0: std = 1.0 
        # 保存 mean 和 std
        log_mean[col + '_log'] = mean
        log_std[col + '_log'] = std

        # 用训练集 mean 和 std 做标准化，防止数据泄露
        train_df[col + '_log'] = (train_df[col + '_log'] - mean) / std
        valid_df[col + '_log'] = (valid_df[col + '_log'] - mean) / std
        test_df[col + '_log'] = (test_df[col + '_log'] - mean) / std
    dense_cols = [f'{col}_log' for col in int_cols] # 处理后的稠密特征列列名

    bin_cols = [f'{col}_bin' for col in int_cols] # 分桶生成新列
    for col in int_cols:
        # 如果 x <= 2，值设为 1 否则为 floor(log2(x))
        train_df[col + '_bin'] = train_df[col].apply(lambda x : int(np.log2(x)) if x > 2 else 1)
        valid_df[col + '_bin'] = valid_df[col].apply(lambda x : int(np.log2(x)) if x > 2 else 1)
        test_df[col + '_bin'] = test_df[col].apply(lambda x : int(np.log2(x)) if x > 2 else 1)

    for col in cat_cols: # 类别特征的缺失值填充为新的类别 missing
        train_df[col] = train_df[col].fillna('missing').astype(str)
        valid_df[col] = valid_df[col].fillna('missing').astype(str)
        test_df[col] = test_df[col].fillna('missing').astype(str)

    # 分桶后的整数列变为新的类别特征, 加入到 cat_cols 中
    all_cat_cols = cat_cols + bin_cols

    # 将类别特征的哈希值映射为整数索引（每列的含义不同，映射也不同）
    encoders = {} # 保存每列的映射
    for col in all_cat_cols:
        unique_vals = train_df[col].unique() # 获取不重复的值
        mapping = {v: i for i, v in enumerate(unique_vals)} # 建立字典映射
        encoders[col] = mapping
        train_df[col] = train_df[col].map(mapping) # 对该列采用映射

        # 测试集与验证集：复用映射，未见类别映射为 unknown_idx, 因为测试集（验证集）可能有训练集未出现的类别
        unknown_idx = len(mapping)
        test_df[col] = test_df[col].map(mapping).fillna(unknown_idx).astype(int)
        valid_df[col] = valid_df[col].map(mapping).fillna(unknown_idx).astype(int)

    artifact = { # 记录所有统计量和特征列名
        "int_median": int_median,
        "log_mean": log_mean,
        "log_std": log_std,
        "encoders": encoders,
        "int_cols": int_cols,
        "cat_cols": cat_cols,
        "bin_cols": bin_cols,
        "all_cat_cols": all_cat_cols,
        "dense_cols": dense_cols,
        "label_col": label_col,
    }
        
    return train_df, valid_df, test_df, all_cat_cols, dense_cols, encoders, artifact

def load_data(train_df, valid_df, test_df, all_cat_cols, dense_cols, batch_size, num_workers = 4):
    # num_workers: 开启的子进程数

    # 稠密特征转换为 float32
    X_dense_train = torch.tensor(train_df[dense_cols].values, dtype=torch.float32)
    X_dense_valid = torch.tensor(valid_df[dense_cols].values, dtype=torch.float32)
    X_dense_test = torch.tensor(test_df[dense_cols].values, dtype=torch.float32)

    # 类别特征转换为 long 的 tensor
    X_cat_train = torch.tensor(train_df[all_cat_cols].values, dtype=torch.long)
    X_cat_valid = torch.tensor(valid_df[all_cat_cols].values, dtype=torch.long)
    X_cat_test = torch.tensor(test_df[all_cat_cols].values, dtype=torch.long)

    # 标签转为 float32 的 tensor, 因为后面要用 BCEWithLogitsLoss 损失函数
    Y_train = torch.tensor(train_df[label_col].values, dtype=torch.float32)
    Y_valid = torch.tensor(valid_df[label_col].values, dtype=torch.float32)
    Y_test = torch.tensor(test_df[label_col].values, dtype=torch.float32)

    # 数据组织成 TensorDataset
    train_dataset = data.TensorDataset(X_cat_train, X_dense_train, Y_train)
    valid_dataset = data.TensorDataset(X_cat_valid, X_dense_valid, Y_valid)
    test_dataset = data.TensorDataset(X_cat_test, X_dense_test, Y_test)

    train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_iter = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, valid_iter, test_iter
