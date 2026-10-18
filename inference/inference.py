
import numpy as np
import torch
import torch.utils.data as data

def process_and_pack(df, artifact, batch_size, num_workers):
    # df 列名：target intCol_0 ~ intCol_12 catCol_0 ~ catCol_25
    df = df.copy()

    int_cols = artifact["int_cols"] # 列名：intCol_0 ~ intCol_12
    cat_cols = artifact["cat_cols"] # 列名：catCol_0 ~ catCol_25
    dense_cols = artifact["dense_cols"]
    all_cat_cols = artifact["all_cat_cols"]
    log_mean = artifact["log_mean"] # 每列的 mean
    log_std = artifact["log_std"] # 每列的 std

    # ---------- 处理数值特征 ----------
    for col in int_cols:
        median = artifact["int_median"][col]
        df[col] = df[col].fillna(median).astype(int) # 填充缺失值为中位数
        # log1p 变换，并写入新的列 intCol_i_log（dense_cols）
        df[col + '_log'] = np.log1p(df[col].clip(lower=0).astype(np.float32))
        # 取出对应列的 mean 和 std，做标准化
        mean = log_mean[col + '_log']
        std = log_std[col + '_log']
        df[col + '_log'] = (df[col + '_log'] - mean) / std
        # 对整数分桶，并写入新的列 intCol_i_bin
        df[col + '_bin'] = df[col].apply(lambda x : int(np.log2(x)) if x > 2 else 1)

    # 填充类别特征缺失值
    for col in cat_cols:
        df[col] = df[col].fillna('missing').astype(str)

    # ---------- 处理类别特征 ----------
    for col in all_cat_cols:
        # 类别特征重新映射
        mapping = artifact["encoders"][col]
        unknown_idx = len(mapping)
        df[col] = df[col].map(mapping).fillna(unknown_idx).astype(int)

    X_cat = torch.tensor(df[all_cat_cols].values, dtype=torch.long)
    X_dense = torch.tensor(df[dense_cols].values, dtype=torch.float32)

    dataset = data.TensorDataset(X_cat, X_dense)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def predict(dataloader, model):
    model.eval()
    device = next(model.parameters()).device
    probs = []
    with torch.no_grad():
        for X_cat, X_dense in dataloader:
            logits = model(X_cat.to(device), X_dense.to(device)) # [batch_size]
            probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy().reshape(-1) # [seq_lens]

