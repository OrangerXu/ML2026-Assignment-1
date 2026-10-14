import pandas as pd
import numpy as np
from model import Model


class Solution:
    def __init__(self):
        """
        初始化随机权重示例，供你检查接口；它不是教师评分基线。
        """
        self.model = Model()
        np.random.seed(42)
        n_features = 15

        self.model.weights = np.random.randn(n_features) * 0.01
        self.model.bias = np.random.randn() * 0.01

        self.X_mean = np.zeros(n_features)
        self.X_std = np.ones(n_features)

    def forward(self, sample: dict) -> dict:
        """
        单样本推理接口

        Args:
            sample: 包含单个样本特征的字典（不包含age列）

        Returns:
            dict: 包含'prediction'键的字典，值为预测的年龄
        """
        sample_df = pd.DataFrame([sample])

        # The evaluator supplies the 15 features; id and age are excluded.
        X = sample_df.drop(columns=['id', 'age'], errors='ignore')

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        X = X.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)

        X_scaled = (X - self.X_mean) / self.X_std

        prediction = self.model.predict(X_scaled)[0]

        return {'prediction': float(prediction)}
