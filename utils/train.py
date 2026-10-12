
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, valid_iter, criterion):

    model.eval()
    all_y_pred = []
    all_y_label = []
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for X_cat, X_dense, y_label in valid_iter:
            logits = model(X_cat, X_dense)
            loss = criterion(logits, y_label)
            y_pred = torch.sigmoid(logits) # 模型输出的是 logits，未经过 sigmoid
            # 计算总损失与样本数，后面会计算 avg_loss
            total_loss += loss.item() * len(y_label) # BCEWithLogitsLoss 返回的是误差均值，别忘了乘该批次样本数量
            num_samples += len(y_label)
            # 将 y_pred 与 y_label 转为 numpy 数组并加入列表，后面方便计算 auc
            all_y_pred.append(y_pred.cpu().numpy())
            all_y_label.append(y_label.cpu().numpy())

    # 计算损失
    avg_loss = total_loss / num_samples
    # 计算 AUC
    y_pred = np.concatenate(all_y_pred).reshape(-1)
    y_label = np.concatenate(all_y_label).reshape(-1)
    auc = roc_auc_score(y_label, y_pred)

    return avg_loss, auc

def train(model, criterion, optimizer, scheduler, epochs, train_iter, valid_iter, test_iter, early_stop_patience):
    
    best_auc = 0.0
    bad_epochs = 0
    save_path = "save/best_" + type(model).__name__ + ".pt"
    save_loss = "save/loss_curve_" + type(model).__name__ + ".png"
    save_auc = "save/auc_curve_" + type(model).__name__ + ".png"

    # 绘图用
    train_losses = []
    valid_losses = []
    valid_aucs = []

    for e in range(epochs):
        # 模型训练
        model.train()
        total_train_loss = 0.0
        train_samples = 0

        for X_cat, X_dense, y in train_iter:
            optimizer.zero_grad()
            loss = criterion(model(X_cat, X_dense), y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y)
            train_samples += len(y)

        train_loss = total_train_loss / train_samples
        # 每一个 epoch 结束后，在验证集上计算 avg_loss 以及 AUC
        valid_loss, auc = evaluate(model, valid_iter, criterion)

        # --- 调度器：根据验证集 AUC 决定是否衰减学习率 ---
        scheduler.step(auc)   # 注意：mode='max' 时传 AUC
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {e + 1}: \ntrain loss: {train_loss:.4f}\nvalid loss: {valid_loss:.4f}\nvalid AUC: {auc:.4f}\nlr: {cur_lr:.2e}\n")

        # 绘图用
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_aucs.append(auc)

        if auc > best_auc: # 更新最佳模型
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            bad_epochs = 0
        else: # 如果 AUC 没有上涨，则早停计数加一
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"Training early stopped at epoch {e + 1}")
                break

    # 训练结束，加载最佳模型，在测试集上计算 AUC
    model.load_state_dict(torch.load(save_path))
    _, test_auc = evaluate(model, test_iter, criterion)
    print(f"Test AUC: {test_auc:.4f}")

    # 绘图部分
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_loss)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, valid_aucs, label='Valid AUC', marker='o')
    plt.axhline(y=test_auc, color='r', linestyle='--', label=f'Test AUC = {test_auc:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_auc)
    plt.close()
