import argparse
import yaml
import joblib
import torch
import torch.nn as nn

from network.models import LR, FM, Wide_Deep, DeepFM
from preprocess.data_process import process, load_data
from utils.train import train

def build_model(name, params):
    name = name.lower()
    if name == "lr":
        return LR(**params)
    elif name == "fm":
        return FM(**params)
    elif name == "wide_deep":
        return Wide_Deep(**params)
    elif name == "deepfm":
        return DeepFM(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

if __name__ == "__main__":

    # 命令行解析器初始化
    parser = argparse.ArgumentParser() # 实例化 ArgumentParser 接收命令行传来的参数
    # 加入命令行可以解析的参数名称
    parser.add_argument("--config", type=str, required=True) # required=True 表示必须传入 config 参数
    parser.add_argument("--dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--early_stop_patience", type=int)
    args = parser.parse_args() # 读取命令行参数，传给 args

    # 处理 --config
    with open(args.config, 'r', encoding="utf-8") as f:
        # 使用 safe_load 解析 yaml 文件，yaml 里的参数转换成字典存在 config 里
        config = yaml.safe_load(f)

    # 处理命令行传入的参数，如果在命令行中传入了相应参数，则覆盖 config 里的内容（yaml 里的内容）
    for key in ["dim", "lr", "epochs", "weight_decay", "early_stop_patience"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    # 加载数据
    train_df, valid_df, test_df, all_cat_cols, dense_cols, encoders, artifact = process(config["data_path"])
    joblib.dump(artifact, "save/artifact.pkl") # 将统计量保存起来，推理用
    train_iter, valid_iter, test_iter = load_data(train_df, valid_df, test_df, all_cat_cols, dense_cols, config["batch_size"], config["num_workers"])
    emb_size_of_every_col = [len(encoders[col]) + 1 for col in all_cat_cols] # 每个特征列特征值的最大值（这一列 Embedding 的大小）

    # 模型初始化准备
    name = config["model_name"]
    if name == "lr":
        params = { # lr 的参数
            "emb_size_of_every_col": emb_size_of_every_col,
            "cat_feature_cols": len(all_cat_cols),
            "dense_feature_cols": len(dense_cols)
        }
    elif name == "fm":
        params = { # fm 的参数
            "emb_size_of_every_col": emb_size_of_every_col,
            "dense_feature_cols": len(dense_cols),
            "dim": config["dim"]
        }
    else:
        params = { # wide_deep 和 deepfm 的参数
            "emb_size_of_every_col": emb_size_of_every_col,
            "cat_feature_cols": len(all_cat_cols),
            "dense_feature_cols": len(dense_cols),
            "dim": config["dim"]
        }
    
    # 根据 name 和 params 初始化模型
    try:
        model = build_model(name, params)
    except ValueError as ve:
        print(f"Model input erroe: {str(ve)}")
    except Exception as e:
        print(f"Unknown error occurred when initializing model：{str(e)}")
        
    # 损失函数，优化器和调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6
    )
    
    # 训练模型
    train(model, criterion, optimizer, scheduler, config["epochs"], train_iter, valid_iter, test_iter, config["early_stop_patience"])