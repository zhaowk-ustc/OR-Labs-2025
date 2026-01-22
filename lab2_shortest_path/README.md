# Shortest Path (Lab 2)

## 环境配置

推荐使用 conda

```powershell
conda env create -f environment.yml
conda activate or25
```

## 运行实验

在本目录下运行：

```powershell
python lab2.py
```

脚本会自动完成：
- 生成随机强连通有向图（Erdős–Rényi），边权为 1~10 的整数
- 用堆优化 Dijkstra 计算从源点 `0` 到汇点 `n-1` 的最短路长度
- 将最短路建模为 LP（最小费用流形式），调用 lab1 的自研求解器 `lp_solver` 求解
- 对不同规模做多次重复实验，输出均值与标准差
- 在当前目录生成性能对比图 `performance_comparison.png`（对数坐标）

## 输出说明

- 控制台：打印一张表格，列为 `Node Count / Trials / Dijkstra (Mean ± Std) / LP Solver (Mean ± Std)`
- 文件：生成 `performance_comparison.png`

## 参数说明（可在 `lab2.py` 中修改）

`run_experiment()` 中的关键参数：
- `NODE_COUNTS = [10, 20, 30, 40, 50]`：节点规模
- `NUM_TRIALS = 20`：每个规模重复次数

## 注意事项

- Dijkstra 要求边权非负；本实验生成的边权为正整数，满足要求。
- `lp_solver` 仅接受不等式约束 `A_ub x <= b_ub`，因此代码将等式流量守恒约束 `A_eq x = b_eq` 转换为一对不等式（见实验报告）。
