# DOE Master Pro V6

DOE Master Pro V6 是一款基于Python和Qt开发的智能实验设计与优化系统，用于帮助科研人员和工程师进行实验设计、模型训练、结果分析和参数优化。

## 功能特性

### 1. 实验设计
- 支持拉丁超立方抽样 (LHS)
- 可自定义变量范围和样本数量
- 实时查看抽样分布
- 支持手动输入实验数据

### 2. 模型训练
- 支持 Kriging (高斯过程) 模型
- 支持 Random Forest 模型
- 自动推荐合适的算法
- 交叉验证评估模型性能
- 实时显示训练进度和结果

### 3. 结果分析
- 关键因子分析 (Pareto图)
- SHAP蜂群图 (Random Forest模型)
- 主效应图
- 交互效应图
- 3D响应面图
- 模型诊断图

### 4. 预测与优化
- 自动参数寻优
- 支持最大化/最小化目标
- 多方案预测推演
- 实时显示优化结果

## 安装依赖

```bash
# 安装基本依赖
pip install numpy pandas scipy matplotlib scikit-learn

# 安装可选依赖
pip install shap  # 用于SHAP蜂群图

# 安装Qt库 (选择其中一个)
pip install PyQt6
# 或
pip install PySide6
```

## 如何运行

### 方法1：直接运行

```bash
python main.py
```

### 方法2：使用IDE

1. 用Python IDE打开项目目录
2. 安装所需依赖
3. 运行`main.py`文件

## 项目结构

```
xdoeV8/
├── main.py              # 程序入口
├── app.py               # 主应用类
├── analysis_engine.py   # 核心数学引擎
├── workers.py           # 工作线程
├── plotting.py          # 绘图组件
├── components.py        # 自定义UI组件
├── utils.py             # 工具函数
└── README.md            # 项目说明
```

## 使用指南

### 1. 实验设计

1. 在"1. 实验设计"页面，点击"➕ 添加变量"添加实验变量
2. 设置每个变量的名称、最小值和最大值
3. 设置样本数量和小数位数
4. 点击"🎲 生成实验设计 (LHS)"生成实验方案
5. 在数据表格中填入实验结果 (Response_Y列)

### 2. 模型训练

1. 切换到"2. 模型训练"页面
2. 选择合适的算法 (或使用推荐算法)
3. 点击"🚀 开始训练模型"开始训练
4. 查看训练日志和模型性能指标
5. 点击"🔍 模型诊断"查看模型拟合情况

### 3. 结果分析

1. 切换到"3. 结果分析"页面
2. 从下拉菜单选择要查看的图表类型
3. 根据需要调整图表参数
4. 点击"💾 保存图片"保存图表

### 4. 预测与优化

1. 切换到"4. 预测与优化"页面
2. 选择优化目标 (最大化或最小化)
3. 点击"⚡ 寻找最优解"开始自动优化
4. 在"4.2 多方案预测推演"区域添加预测场景并计算结果

## 技术栈

- **编程语言**: Python 3.7+
- **GUI框架**: PyQt6 / PySide6
- **科学计算**: NumPy, Pandas, SciPy
- **机器学习**: scikit-learn
- **可视化**: Matplotlib
- **实验设计**: scipy.stats.qmc

## 许可证

本项目采用 MIT 许可证，详情请见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱：example@example.com
- GitHub：https://github.com/example/doe-master-pro

## 更新日志

### v6.0
- 全新的模块化架构
- 支持SHAP蜂群图
- 优化了模型训练速度
- 改进了用户界面
- 增加了自动算法推荐功能
