# Assignment 1：年龄预测回归任务

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

本作业要求使用客户的职业、教育、信贷及营销记录等 15 个特征预测年龄，完成模型核心算法实现、SHAP 特征贡献分析和特征工程实验。

## 1. 任务概述与作业要求

预测目标为 `age`，任务类型为监督学习回归。模型训练数据为 `train.csv`，公开验证数据为 `test_public.csv`。

| 数据文件 | 样本数量 | 标签 | 用途 |
|---|---:|---|---|
| `train.csv` | 600000 | 包含 `age` | 模型训练与训练集内验证 |
| `test_public.csv` | 50000 | 包含 `age` | 本地自评、方案比较与误差分析 |

需完成以下内容：

1. **模型实现。** 在 `model.py` 和 `solution.py` 中独立实现模型核心算法，包括训练逻辑、数据预处理和预测过程。
2. **SHAP 分析。** 使用 SHAP（SHapley Additive exPlanations）分析提交模型的特征贡献。分析代码保存在 `analysis.py`，报告中说明背景数据、解释样本、主要参数及分析结论。
3. **特征工程。** 根据分析结果开展特征工程实验，比较修改前后的模型表现，并说明实验依据。未取得改善的实验也应如实记录。
4. **结果复现。** 固定训练与分析中的随机种子，提交完整自评结果及相应源码，保证教师能够重新训练和评测。

模型核心算法可以使用 NumPy、pandas 和 Python 标准库。禁止直接导入或调用以下机器学习库及其现成模型：

`sklearn` / `scikit-learn`、`tensorflow`、`torch` / `pytorch`、`keras`、`xgboost`、`lightgbm`、`catboost`、`statsmodels`。

`analysis.py` 可以使用 `shap` 和绘图库。SHAP 自身的间接依赖不视为模型核心禁库违规，但解释对象必须是提交的模型，不得另用现成模型替代作业核心算法。SHAP 分析方法不要求手写实现。

## 2. 数据字段说明

`age` 为预测目标，`id` 为记录编号。评测器向模型提供以下 15 个特征，不传入 `id` 或 `age`。

| 字段 | 类型 | 含义与取值说明 |
|---|---|---|
| `job` | 类别 | 职业，如 `admin.`、`blue-collar`、`entrepreneur` |
| `marital` | 类别 | 婚姻状况，如 `married`、`single`、`divorced` |
| `education` | 类别 | 教育程度，如 `primary`、`secondary`、`tertiary`、`unknown` |
| `default` | 类别 | 是否有信贷违约记录，`yes` / `no` |
| `balance` | 数值 | 年平均余额，单位为欧元 |
| `housing` | 类别 | 是否有住房贷款，`yes` / `no` |
| `loan` | 类别 | 是否有个人贷款，`yes` / `no` |
| `contact` | 类别 | 联系方式，如 `unknown`、`telephone`、`cellular` |
| `day` | 数值 | 最近联系的日期，取值为 1 至 31 |
| `month` | 类别 | 最近联系的月份，如 `jan`、`feb`、`dec` |
| `duration` | 数值 | 最近联系的持续时间，单位为秒 |
| `campaign` | 数值 | 本次营销中的联系次数 |
| `pdays` | 数值 | 距上次营销联系的天数；`-1` 表示此前未联系过 |
| `previous` | 数值 | 本次营销之前的联系次数 |
| `poutcome` | 类别 | 上次营销结果，如 `unknown`、`other`、`failure`、`success` |

## 3. 评测指标与评分方式

### 3.1 评测指标

| 指标 | 含义 | 作用 |
|---|---|---|
| MAE | 平均绝对误差 | 衡量预测值与标签的平均绝对偏差 |
| MSE | 均方误差 | 衡量预测误差的平方均值 |
| RMSE | 均方根误差 | 效果评分的主要指标，数值越低表示误差越小 |
| 耗时 | 训练与准备、预测、工作进程总耗时 | 检查实现的运行成本 |

公开验证集指标用于本地检查与方案比较，不直接决定最终效果分。

### 3.2 评分方式

参考基线成绩：RMSE 7.7398（MAE 6.2973 / MSE 59.9042，test_public.csv）, 所有指标越低越好。

## 4. 项目结构

```text
ML2026-Assignment-1/
├── train.csv          # 训练数据
├── test_public.csv    # 公开验证数据
├── model.py           # 模型核心算法，需修改
├── solution.py        # 训练入口与推理接口，需修改
├── evaluate.py        # 本地自评脚本
├── requirements.txt   # 本地实验环境依赖
└── README.md          # 作业说明
```

提交时还需提供自行编写的 `analysis.py`、`report.md` 及评测生成的 `result.json`。

## 5. 模型实现与接口规范

### 5.1 模型训练

建议在 `Solution.__init__` 中读取 `train.csv`，完成特征编码、数值处理和模型训练，将参数保存在内存中。`model.py` 用于组织模型核心算法，`solution.py` 用于连接训练流程与评测接口。

模型的数据来源仅限 `train.csv`。公开验证标签不得并入训练数据，不得读取其他测试文件、联网获取额外数据、加载外部训练产物或在源码中硬编码预训练权重和答案映射。

### 5.2 单样本推理接口

评测器调用 `Solution.forward(sample)`。输入为包含 15 个特征的字典，输出必须包含 `prediction`，其值为有限的年龄预测数值。

```python
class Solution:
    def forward(self, sample: dict) -> dict:
        """接收单条样本特征，返回年龄预测结果。"""
        prediction = ...  # 替换为模型预测逻辑
        return {"prediction": float(prediction)}
```

输入样本示例：

```python
{
    "job": "blue-collar", "marital": "married",
    "education": "secondary", "default": "no", "balance": 3595,
    "housing": "yes", "loan": "no", "contact": "unknown",
    "day": 3, "month": "jul", "duration": 198,
    "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown"
}
```

评测器使用 10 个线程并发调用同一个 `Solution` 实例。`forward` 不应重新训练模型或修改共享权重，训练与推理的特征处理规则应保持一致。

教师运行模型时仅加载 `solution.py` 和 `model.py`，运行环境提供 NumPy、pandas 和标准库。运行所需源码应放在这两个文件中，不得依赖额外模块或个人计算机上的绝对路径。每次公开或私有评测的预算为 1800 秒，包含训练与预测；内存上限为 2 GiB。

## 6. 环境配置

建议使用 Python 3.11。教师模型运行环境采用 Python 3.11.11、NumPy 1.24.3 和 pandas 2.0.3。

```bash
conda create -n ML2026 python=3.11
conda activate ML2026
pip install -r requirements.txt
```

`requirements.txt` 包含本地分析与可视化所需依赖，不表示其中所有库均可用于模型核心实现；库使用范围以第 1 节为准。

## 7. 本地评测

在项目根目录执行：

```bash
# 完整评测公开验证集
python evaluate.py

# 快速检查前 5000 行
python evaluate.py --rows 5000

# 在结果文件中记录学号和姓名
python evaluate.py --student-id 20260001 --name 张三
```

评测结束后，脚本输出指标并生成 `result.json`，记录评测器版本、数据摘要、样本数量、源码摘要、环境信息及耗时。`--rows` 用于调试；提交前应对最终源码执行完整评测。

修改源码后必须重新生成结果文件。本地评测器用于学生自评，不构成操作系统安全沙箱；教师采用独立的容器核验器执行提交，运行器不提供测试标签或网络访问。

## 8. 提交要求

提交以下五个文件，保持文件名不变并采用 UTF-8 编码：

| 文件 | 提交内容 |
|---|---|
| `solution.py` | 模型训练入口与单样本预测接口 |
| `model.py` | 模型核心算法 |
| `analysis.py` | SHAP 分析代码 |
| `report.md` | 模型说明、分析结论、特征实验及参考声明 |
| `result.json` | 最终源码的完整自评结果 |

报告应说明模型与训练方法、预处理步骤、SHAP 分析设置和结论、特征工程对照结果，以及参考资料和 AI 工具的使用情况。实验结果应与提交源码对应。

本仓库不规定截止日期。提交方式、成绩结算与补交安排以课程通知为准。

## 9. 学术诚信与参考资料

允许查阅算法资料，使用情况应在报告中注明；使用 AI 工具时，应说明辅助范围。

复制历年解答、窃取或硬编码测试标签、探测教师评测数据、伪造实验结果以及其他违反数据与库使用规则的行为，将按课程规则核查处理。
