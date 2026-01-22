import networkx as nx
import numpy as np
import heapq
import time
import math
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. 导入 LP Solver
# ==========================================
try:
    from lp_solver import solver, LPProblem
    print("成功导入 lp_solver 模块！")
except ImportError:
    print("错误：未找到 lp_solver 模块。请确保代码与 lp_solver 文件夹在同一目录下。")
    exit()

# ==========================================
# 2. 数据生成 (NetworkX)
# ==========================================
def generate_connected_graph(n):
    """
    生成一个具有 n 个节点的 Erdős–Rényi 随机连通图。
    根据 source 148-149 要求，设置 p > (1 + epsilon) * ln(n) / n 以确保连通性。
    """
    if n <= 1:
        return nx.erdos_renyi_graph(n, 1.0, directed=True)

    epsilon = 0.5
    # 计算保证连通性的概率阈值
    if n > 1:
        threshold = (1 + epsilon) * math.log(n) / n
    else:
        threshold = 1.0
        
    p = max(threshold, 0.2) # 至少给 0.2 的概率，防止 N 很大时边太稀疏导致图不连通概率过高
    if p > 1: p = 1

    while True:
        # 生成有向图
        G = nx.erdos_renyi_graph(n, p, directed=True)
        
        # 检查连通性 (强连通)
        if nx.is_strongly_connected(G):
            # 为边赋予随机权重 (1 到 10)，均为正数
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = np.random.randint(1, 11)
            return G

# ==========================================
# 3. Dijkstra 算法
# ==========================================
def dijkstra_shortest_path(G, source, target):
    """
    Dijkstra 算法实现。
    约束：不直接使用 PriorityQueue，使用 heapq。
    """
    # 前置检查：负权边
    if any(d.get('weight', 1) < 0 for u, v, d in G.edges(data=True)):
        raise ValueError("图中存在负权边，Dijkstra 无法处理。")
    
    dist = {node: float('inf') for node in G.nodes()}
    dist[source] = 0
    
    # 优先队列 (distance, node)
    pq = [(0, source)]
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if u == target:
            return d
        
        if d > dist[u]:
            continue
        
        for v in G.neighbors(u):
            weight = G.edges[u, v].get('weight', 1)
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
                
    return float('inf')

# ==========================================
# 4. LP 建模与求解
# ==========================================
def solve_shortest_path_with_my_lp(G, source, target):
    """
    将最短路建模为 LP 并求解。
    Constraints: Flow conservation constraints.
    """
    nodes = list(G.nodes())
    edges = list(G.edges())
    num_edges = len(edges)
    num_nodes = len(nodes)
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # --- 1. 构建目标函数系数 c (边权重) ---
    c = np.array([G.edges[u, v]['weight'] for u, v in edges], dtype=float)
    
    # --- 2. 构建流量守恒等式约束 A_eq * x = b_eq ---
    # source: +1, target: -1, others: 0
    A_eq = np.zeros((num_nodes, num_edges))
    b_eq = np.zeros(num_nodes)
    
    b_eq[node_to_idx[source]] = 1.0
    b_eq[node_to_idx[target]] = -1.0
    
    for j, (u, v) in enumerate(edges):
        # 流出 u (+1), 流入 v (-1)
        A_eq[node_to_idx[u], j] = 1.0
        A_eq[node_to_idx[v], j] = -1.0

    # --- 3. 转换为不等式以适配 Solver ---
    # Solver 接受 A_ub * x <= b_ub
    # 等式 Ax = b 转化为: Ax <= b 且 -Ax <= -b
    A_ub = np.vstack([A_eq, -A_eq])
    b_ub = np.concatenate([b_eq, -b_eq])
    
    # --- 4. 调用求解器 ---
    lp = LPProblem(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0.0, None))
    
    try:
        res = solver.solve(lp)
        if res.success:
            return res.fun
        else:
            return float('inf')
    except Exception:
        return float('inf')

# ==========================================
# 5. 实验主程序
# ==========================================
def run_experiment():
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    NODE_COUNTS = [10, 20, 30, 40, 50]
    NUM_TRIALS = 20
    
    print(f"{'Node Count':<10} | {'Trials':<6} | {'Dijkstra (Mean ± Std)':<25} | {'LP Solver (Mean ± Std)':<25}")
    print("-" * 85)
    
    avg_times_dijkstra = []
    avg_times_lp = []
    std_times_dijkstra = []
    std_times_lp = []

    for n in NODE_COUNTS:
        trial_times_d = []
        trial_times_l = []
        
        for _ in range(NUM_TRIALS):
            # 1. 每次生成新的随机图
            G = generate_connected_graph(n)
            source, target = 0, n - 1
            
            # 2. Dijkstra
            t_start = time.perf_counter()
            d_res = dijkstra_shortest_path(G, source, target)
            trial_times_d.append(time.perf_counter() - t_start)
            
            # 3. LP Solver
            t_start = time.perf_counter()
            l_res = solve_shortest_path_with_my_lp(G, source, target)
            trial_times_l.append(time.perf_counter() - t_start)
            
        # 计算统计量
        d_mean = np.mean(trial_times_d)
        d_std = np.std(trial_times_d)
        l_mean = np.mean(trial_times_l)
        l_std = np.std(trial_times_l)
        
        # 记录数据
        avg_times_dijkstra.append(d_mean)
        avg_times_lp.append(l_mean)
        std_times_dijkstra.append(d_std)
        std_times_lp.append(l_std)
        
        print(f"{n:<10} | {NUM_TRIALS:<6} | {d_mean:.6f} ± {d_std:.6f} s      | {l_mean:.6f} ± {l_std:.6f} s")

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(NODE_COUNTS, avg_times_dijkstra, yerr=std_times_dijkstra, 
                 fmt='-o', capsize=5, label='Dijkstra')
    
    plt.errorbar(NODE_COUNTS, avg_times_lp, yerr=std_times_lp, 
                 fmt='-s', capsize=5, label='LP Solver (Simplex)')
    
    # 设置 Y 轴为对数刻度
    plt.yscale('log')
    
    plt.title(f'Performance Comparison: Dijkstra vs LP (Log Scale, Avg of {NUM_TRIALS} Runs)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds) - Log Scale')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    run_experiment()