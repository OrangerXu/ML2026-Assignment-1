<div align="center">

# Assignment 1: 年龄预测回归任务

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

**🖥️ 本地自评测** | 克隆仓库，写代码，`python evaluate.py` 立刻看到你的分数

---

</div>

## 📋 任务概述

> 在 `train.csv`（60 万行）上进行年龄预测回归任务：基于客户的职业、教育、信贷等特征预测客户年龄（`age` 列）。`test_public.csv`（5 万行，含真实标签）供你本地自评。

### 🎓 作业要求

```
✨ 1. 使用 SHAP (SHapley Additive exPlanations) 方法分析各特征对年龄预测的贡献度
✨ 2. 基于SHAP分析结果实施特征工程，优化模型性能
✨ 3. 独立实现机器学习模型的核心算法逻辑，不得调用任何外部机器学习库
```

<details>
<summary>❌ 禁止使用的库（评测脚本会扫描源码）</summary>

- sklearn / scikit-learn
- tensorflow
- torch / pytorch
- keras
- xgboost
- lightgbm
- catboost
- statsmodels

</details>

> ⚠️ 推理阶段（`forward`）只允许从 `train.csv` 读取数据；不允许加载任何外部训练产物（pickle / joblib / npy 等模型文件）。

---

## 📊 字段说明

- age: Age of the client (numeric, 预测目标)
- job: Type of job (categorical: "admin.", "blue-collar", "entrepreneur", etc.)
- marital: Marital status (categorical: "married", "single", "divorced")
- education: Level of education (categorical: "primary", "secondary", "tertiary", "unknown")
- default: Has credit in default? (categorical: "yes", "no")
- balance: Average yearly balance in euros (numeric)
- housing: Has a housing loan? (categorical: "yes", "no")
- loan: Has a personal loan? (categorical: "yes", "no")
- contact: Type of communication contact (categorical: "unknown", "telephone", "cellular")
- day: Last contact day of the month (numeric, 1-31)
- month: Last contact month of the year (categorical: "jan", "feb", …, "dec")
- duration: Last contact duration in seconds (numeric)
- campaign: Number of contacts performed during this campaign (numeric)
- pdays: Number of days since last contact from a previous campaign (numeric; -1 = never contacted)
- previous: Number of contacts performed before this campaign (numeric)
- poutcome: Outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")

---

## 📈 评测指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **MSE** (Mean Squared Error): 均方误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **Prediction_Time**: 预测耗时

> ⚡ 评测使用 **10 个并发线程** 逐样本调用你的 `forward`，请保证线程安全

---

## 📂 项目结构

```
📦 project/
 ┣ 📄 train.csv              # 训练数据（60万行，带标签）
 ┣ 📄 test_public.csv        # 自评数据（5万行，带标签）
 ┣ 📄 evaluate.py            # 本地评测脚本
 ┣ 🔧 model.py               # 模型实现（你要改的）
 ┣ 🚀 solution.py            # 推理接口（你要改的）
 ┗ 📋 requirements.txt       # 依赖库
```

> 💡 **Baseline**: 初始 `solution.py` 使用随机权重，你需要实现自己的训练逻辑来提升性能

---

## 💻 模型实现

### 🎨 1. `model.py` - 模型类

```python
import numpy as np

class Model:
    def __init__(self):
        """初始化模型参数"""
        self.weights = None
        self.bias = None

    def predict(self, X):
        """
        Args:
            X: numpy数组, shape (n_samples, n_features)
        Returns:
            numpy数组, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
```

### 🚀 2. `solution.py` - 推理接口

```python
class Solution:
    def forward(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """接收单条样本特征字典，返回预测结果

        Args:
            sample: 不含 age 列的单条样本，例如:
                {'id': 123, 'job': 'blue-collar', 'marital': 'married',
                 'education': 'secondary', 'default': 'no', 'balance': 3595,
                 'housing': 'no', 'loan': 'yes', 'contact': 'unknown',
                 'day': 3, 'month': 'jul', 'duration': 198, 'campaign': 2,
                 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'}

        Returns:
            {'prediction': 预测的年龄(float)}
        """
```

> 💡 推荐模式：在 `Solution.__init__` 中读取 `train.csv` 完成训练并保存参数，`forward` 中只做特征处理和预测。

---

## ⚙️ 环境

```bash
conda create -n ML python=3.8
conda activate ML
pip install -r requirements.txt
```

---

## 🚀 运行评测

```bash
# 在仓库根目录运行（与 solution.py、model.py 同级）
python evaluate.py

# 可选参数
python evaluate.py --rows 5000                 # 只评测前 5000 行（快速试跑）
python evaluate.py --student-id 2026001 --name 张三   # 学号姓名写入结果文件
```

评测完成后：

1. 终端打印 MAE / MSE / RMSE / 预测耗时
2. 生成 `result.json`（含指标、代码文件 MD5、环境信息）

**提交作业时**：将 `solution.py`、`model.py`、`result.json` 一起上交。

---

## ⚖️ 评分说明（请仔细阅读）

- 自评分数（`test_public.csv` 上的结果）**仅供参考**；
- **最终成绩以教师在保留数据上的复测为准**——教师会在与公开数据不同分布的另一份私有数据上重新运行你提交的代码；
- 总分 = 代码效果分 + AI 代码评价分；
- **作弊零分**：包括但不限于——直接读取/硬编码测试集答案、加载外部模型文件、抄袭网络流传的旧版本解答。你的代码会被静态扫描、复测和 AI 评审三重核查。

---

Made with ❤️ for Machine Learning Education
