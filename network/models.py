
import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, emb_size_of_every_col, cat_feature_cols, dense_feature_cols):
        super(LR, self).__init__()
        # LR 没有交叉项，高维 Embedding 之间不会产生交互，维度为 1 即可
        self.embeddings = nn.ModuleList([nn.Embedding(size, 1) for size in emb_size_of_every_col])
        self.linear = nn.Linear(cat_feature_cols + dense_feature_cols, 1)

    def forward(self, X_cat, X_dense):
        # X_cat: [batch_size, cat_feature_cols] 
        # X_dense: [batch_size, dense_feature_cols]

        # ------------- 取出类别特征向量 -------------
        num_cols = X_cat.shape[1]
        cat_features = []
        # 取 X 的每一列特征，送入该列的 Embedding
        for i in range(num_cols):
            feature = self.embeddings[i](X_cat[:, i]) # [batch_size, 1, 1]
            cat_features.append(feature)
        cat_features = torch.stack(cat_features, dim=1) # [batch_size, cat_feature_cols, 1]
        cat_features = cat_features.reshape(cat_features.shape[0], -1) # [batch_size, cat_feature_cols]

        # 与稠密特征连接
        cat_dense_features = torch.cat([cat_features, X_dense], dim=1) # [batch_size, cat_feature_cols + dense_feature_cols]
        
        return torch.squeeze(self.linear(cat_dense_features), dim=1) # [batch_size]

class FM(nn.Module):
    def __init__(self, emb_size_of_every_col, dense_feature_cols, dim):
        super(FM, self).__init__()
        # cat_feature_cols 为类别特征列的列数，包括分桶后的整数类别特征
        # dense_feature_cols 为稠密特征列数

        self.two_square_embs = nn.ModuleList([nn.Embedding(size, dim, max_norm=1) for size in emb_size_of_every_col]) # 二阶交叉特征的 Embedding
        # 二阶 Embedding 初始化为小随机值
        for emb in self.two_square_embs:
            nn.init.normal_(emb.weight, mean=0, std=0.01)
        # 一阶 Embedding 初始化为 0
        self.one_square_embs = nn.ModuleList([nn.Embedding(size, 1) for size in emb_size_of_every_col]) # 一阶特征的 Embedding
        for emb in self.one_square_embs:
            nn.init.zeros_(emb.weight)

        self.w0 = nn.Parameter(torch.zeros(1)) # 全局偏置项
        self.dense_linear = nn.Linear(dense_feature_cols, 1) # 稠密特征经过线性层处理
        self.dropout = nn.Dropout(0.2) # 对 Embedding 做 Dropout

    def FMCross(self, features_embs):
        # features_embs: [batch_size, cat_feature_cols, dim]
        square_of_sum = torch.sum(features_embs, dim = 1) ** 2 # [batch_size, dim]
        sum_of_square = torch.sum(features_embs ** 2, dim = 1) # [batch_size, dim]
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1) # [batch_size]

    def forward(self, X_cat, X_dense):
        # X_cat: [batch_size, cat_feature_cols] 
        # X_dense: [batch_size, dense_feature_cols]
        
        # ----------- 类别特征处理（二阶交叉以及一阶线性，不共享 Embedding）-----------
        # 取出二阶项特征向量
        num_cols = X_cat.shape[1]
        two_square_features = []
        # 取 X 的每一列特征，送入该列的 Embedding
        for i in range(num_cols):
            feature = self.two_square_embs[i](X_cat[:, i]) # [batch_size, 1, dim] 第 i 列特征的 Embedding 向量
            two_square_features.append(feature)
        two_square_features = torch.stack(two_square_features, dim=1) # [batch_size, cat_feature_cols, dim]
        two_square_features = self.dropout(two_square_features)

        # 取出一阶项特征向量
        num_cols = X_cat.shape[1]
        one_square_features = []
        # 取 X 的每一列特征，送入该列的 Embedding
        for i in range(num_cols):
            feature = self.one_square_embs[i](X_cat[:, i]) # [batch_size, 1, 1] 第 i 列特征的 Embedding 向量
            one_square_features.append(feature)
        one_square_features = torch.squeeze(torch.stack(one_square_features, dim=1), dim=-1) # [batch_size, cat_feature_cols]
        one_square_features = self.dropout(one_square_features)

        # ---------- 稠密特征处理（直接作为一阶项加入模型）----------
        dense = torch.squeeze(self.dense_linear(X_dense), dim=-1) # [batch_size]

        # 一阶项部分（dim=1 的 Embedding 直接求和等价于线性映射）
        one = torch.sum(one_square_features, dim=1) # [batch_size]
        # 交叉特征部分
        two = self.FMCross(two_square_features) # [batch_size]

        return self.w0 + one + two + dense # 一阶项，二阶项和全局偏置以及稠密特征部分

class Wide_Deep(nn.Module):
    def __init__(self, emb_size_of_every_col, cat_feature_cols, dense_feature_cols, dim):
        super(Wide_Deep, self).__init__()
        # cat_feature_cols 为类别特征列的列数，包括分桶后的整数类别特征
        # dense_feature_cols 为稠密特征列的列数
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, max_norm=1) for size in emb_size_of_every_col])
        self.linear = nn.Linear(cat_feature_cols * dim + dense_feature_cols, 1)
        self.mlp = self._mlp(cat_feature_cols * dim + dense_feature_cols)
        self.dropout = nn.Dropout(0.3)

    def _mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 4, 1) # 不需要 sigmoid
        )

    def Deep(self, features_embs):
        # features_embs: [batch_size, cat_feature_cols * dim + dense_feature_cols]
        return torch.squeeze(self.mlp(features_embs), dim=1)

    def Wide(self, features_embs):
        # features_embs: [batch_size, cat_feature_cols * dim + dense_feature_cols]
        return torch.squeeze(self.linear(features_embs), dim=1)

    def forward(self, X_cat, X_dense):
        # X_cat: [batch_size, cat_feature_cols] 
        # X_dense: [batch_size, dense_feature_cols]
        
        # ------------- 取出类别特征向量 -------------
        num_cols = X_cat.shape[1]
        cat_features = []
        # 取 X 的每一列特征，送入该列的 Embedding
        for i in range(num_cols):
            feature = self.embeddings[i](X_cat[:, i]) # [batch_size, 1, dim] 第 i 列特征的 Embedding 向量
            cat_features.append(feature)
        cat_features = torch.stack(cat_features, dim=1) # [batch_size, cat_feature_cols, dim]
        cat_features = self.dropout(cat_features)
        # 特征拼接
        flatten_cat_features = cat_features.reshape(cat_features.shape[0], -1) # [batch_size, cat_feature_cols * dim]
        cat_dense_features = torch.cat([flatten_cat_features, X_dense], dim=1) # [batch_size, cat_feature_cols * dim + dense_feature_cols]
        
        # Wide 部分
        wide = self.Wide(cat_dense_features) # [batch_size]
        # Deep 部分
        deep = self.Deep(cat_dense_features) # [batch_size]
        
        return deep + wide

class DeepFM(nn.Module):
    def __init__(self, emb_size_of_every_col, cat_feature_cols, dense_feature_cols, dim):
        super(DeepFM, self).__init__()
        # cat_feature_cols 为类别特征列的列数，包括分桶后的整数类别特征
        # dense_feature_cols 为稠密特征列的列数
        # 每一列一个 Embedding 列的先后顺序与 all_cat_cols 中一致

        # FM 的参数
        self.two_square_embs = nn.ModuleList([nn.Embedding(size, dim, max_norm=1) for size in emb_size_of_every_col]) # 这部分参数与 Deep 部分的类别特征共享
        for emb in self.two_square_embs:
            nn.init.normal_(emb.weight, mean=0, std=0.01)
        self.one_square_embs = nn.ModuleList([nn.Embedding(size, 1) for size in emb_size_of_every_col])
        for emb in self.one_square_embs:
            nn.init.zeros_(emb.weight)

        self.w0 = nn.Parameter(torch.zeros(1))
        self.dense_linear = nn.Linear(dense_feature_cols, 1)

        # Deep 的参数
        self.mlp = self._mlp(cat_feature_cols * dim + dense_feature_cols)

    def _mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 4, 1) # 不需要 sigmoid
        )

    def Deep(self, features_embs):
        # features_embs: [batch_size, cat_feature_cols * dim + dense_feature_cols]
        return torch.squeeze(self.mlp(features_embs), dim=1)

    def FMCross(self, features_embs):
        # features_embs: [batch_size, cat_feature_cols, dim]
        square_of_sum = torch.sum(features_embs, dim = 1) ** 2 # [batch_size, dim]
        sum_of_square = torch.sum(features_embs ** 2, dim = 1) # [batch_size, dim]
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1) # [batch_size]

    def forward(self, X_cat, X_dense):
        # X_cat: [batch_size, cat_feature_cols] 
        # X_dense: [batch_size, dense_feature_cols]
        
        # ----------- 类别特征处理（二阶交叉以及一阶线性，不共享 Embedding）-----------
        # 取出二阶项特征向量
        num_cols = X_cat.shape[1]
        two_square_features = []
        # 取 X 的每一列特征，送入该列的 Embedding
        for i in range(num_cols):
            feature = self.two_square_embs[i](X_cat[:, i]) # [batch_size, 1, dim] 第 i 列特征的 Embedding 向量
            two_square_features.append(feature)
        two_square_features = torch.stack(two_square_features, dim=1) # [batch_size, cat_feature_cols, dim]

        # 取出一阶项特征向量
        num_cols = X_cat.shape[1]
        one_square_features = []
        # 取 X 的每一列特征，送入该列的 Embedding
        for i in range(num_cols):
            feature = self.one_square_embs[i](X_cat[:, i]) # [batch_size, 1, 1]
            one_square_features.append(feature)
        one_square_features = torch.squeeze(torch.stack(one_square_features, dim=1), dim=-1) # [batch_size, cat_feature_cols]
        
        # FM 部分
        two = self.FMCross(two_square_features) # [batch_size]
        one = torch.sum(one_square_features, dim=1) # [batch_size]
        dense = torch.squeeze(self.dense_linear(X_dense), dim=-1) # [batch_size]
        fm = one + two + dense + self.w0

        # Deep 部分
        flatten_cat_features = two_square_features.reshape(two_square_features.shape[0], -1) # [batch_size, cat_feature_cols * dim]
        cat_dense_features = torch.cat([flatten_cat_features, X_dense], dim=1) # [batch_size, cat_feature_cols * dim + dense_feature_cols]
        deep = self.Deep(cat_dense_features) # [batch_size]
        
        return deep + fm # 由于损失函数采用 BCEWithLogitsLoss 所以不需要取 sigmoid
