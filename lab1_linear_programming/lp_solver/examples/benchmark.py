"""
性能评测与可视化脚本

功能:
- 遍历一系列预设的线性规划问题规模。
- 对每个规模，重复运行求解器20次。
- 收集成功运行的结果，并计算平均耗时。
- 使用matplotlib和seaborn生成性能对比图表。
- 最终将详细数据以表格形式打印。

- 在项目根目录运行本示例：
    python -m lp_solver.examples.benchmark_plot
"""

import time
import numpy as np
from typing import Dict, Any, List

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from lp_solver.types import LPProblem
from lp_solver.module0_standardize import to_standard_form
from lp_solver.module1_redundancy import prune_redundant_rows
from lp_solver.module2_bigm import init_with_big_m
from lp_solver.module3_simplex import iterate_simplex
from lp_solver.module4_postprocess import recover_solution
from lp_solver.exceptions import UnboundedError, InfeasibleError

def generate_feasible_lp(m_eq: int, m_ub: int, n: int, seed: int = 42) -> LPProblem:
    """生成一个具有给定规模且保证有可行解的随机线性规划问题。"""
    np.random.seed(seed)
    x_sol = np.random.rand(n) * 10
    c = -np.random.rand(n)
    A_eq = np.random.rand(m_eq, n) * 2 - 1
    A_ub = np.random.rand(m_ub, n) * 2 - 1
    b_eq = A_eq @ x_sol
    b_ub = (A_ub @ x_sol + np.random.rand(m_ub) * 2)
    bounds = (0., None)
    return LPProblem(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

def run_single_benchmark(lp: LPProblem) -> Dict[str, Any]:
    """对单个LP问题运行一次评测，返回包含所有性能数据的字典。"""
    results: Dict[str, Any] = {'custom': {}, 'scipy': {}}
    custom_times: Dict[str, float] = {}
    t_start: float=0.0
    # --- 1. 运行自定义求解器 (分步计时) ---
    try:
        t_start = time.perf_counter()
        std, xfm = to_standard_form(lp)
        custom_times['mod0'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        std_pruned = prune_redundant_rows(std)
        custom_times['mod1'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        tab = init_with_big_m(std_pruned)
        custom_times['mod2'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        sol = iterate_simplex(tab, rule="bland")
        custom_times['mod3'] = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        sol_recovered = recover_solution(sol, xfm, sense="min")
        custom_times['mod4'] = time.perf_counter() - t_start

        results['custom']['success'] = True
        results['custom']['total_time'] = sum(custom_times.values())
        results['custom']['times'] = custom_times

    except Exception:
        # 其他所有异常均视为失败
        results['custom']['success'] = False

    # --- 2. 运行 SciPy (HiGHS) 求解器 ---
    try:
        from scipy.optimize import linprog
        t_start = time.perf_counter()
        res_scipy = linprog(c=lp.c, A_ub=lp.A_ub, b_ub=lp.b_ub, A_eq=lp.A_eq, b_eq=lp.b_eq, bounds=lp.bounds, method="highs")
        results['scipy']['total_time'] = time.perf_counter() - t_start
        results['scipy']['success'] = res_scipy.success
    except Exception:
        results['scipy']['success'] = False
            
    return results

def aggregate_and_plot_results(all_run_data: Dict, num_runs: int):
    """对所有运行数据进行聚合、打印表格并生成图表。"""
    
    summary_data = []
    plot_data = []

    for size, runs in all_run_data.items():
        size_str = f"({size[0]},{size[1]},{size[2]})"
        
        # 聚合自定义求解器结果
        custom_successful_runs = [r['custom'] for r in runs if r['custom']['success']]
        if custom_successful_runs:
            custom_times = [r['total_time'] for r in custom_successful_runs]
            avg_custom_time = np.mean(custom_times)
            std_custom_time = np.std(custom_times)
            avg_mod0 = np.mean([r['times'].get('mod0', 0) for r in custom_successful_runs])
            avg_mod1 = np.mean([r['times'].get('mod1', 0) for r in custom_successful_runs])
            avg_mod2 = np.mean([r['times'].get('mod2', 0) for r in custom_successful_runs])
            avg_mod3 = np.mean([r['times'].get('mod3', 0) for r in custom_successful_runs])
            avg_mod4 = np.mean([r['times'].get('mod4', 0) for r in custom_successful_runs])
            custom_success_rate = len(custom_successful_runs) / num_runs
        else:
            avg_custom_time = std_custom_time = avg_mod0 = avg_mod1 = avg_mod2 = avg_mod3 = avg_mod4 = 0
            custom_success_rate = 0

        # 聚合SciPy结果
        scipy_successful_runs = [r['scipy'] for r in runs if r['scipy']['success']]
        if scipy_successful_runs:
            scipy_times = [r['total_time'] for r in scipy_successful_runs]
            avg_scipy_time = np.mean(scipy_times)
            std_scipy_time = np.std(scipy_times)
            scipy_success_rate = len(scipy_successful_runs) / num_runs
        else:
            avg_scipy_time = std_scipy_time = 0
            scipy_success_rate = 0

        summary_data.append({
            'Size': size_str,
            'Custom Time (ms)': f"{avg_custom_time * 1000:.2f} ± {std_custom_time * 1000:.2f}",
            'SciPy Time (ms)': f"{avg_scipy_time * 1000:.2f} ± {std_scipy_time * 1000:.2f}",
            'M0 (ms)': avg_mod0 * 1000, 'M1 (ms)': avg_mod1 * 1000,
            'M2 (ms)': avg_mod2 * 1000, 'M3 (ms)': avg_mod3 * 1000,
            'M4 (ms)': avg_mod4 * 1000,
            'Custom Success Rate': f"{custom_success_rate:.0%}",
            'SciPy Success Rate': f"{scipy_success_rate:.0%}",
        })
        
        plot_data.append({
            'Size': size_str,
            'avg_custom_time': avg_custom_time, 'std_custom_time': std_custom_time,
            'avg_scipy_time': avg_scipy_time, 'std_scipy_time': std_scipy_time,
            'modules': [avg_mod0, avg_mod1, avg_mod2, avg_mod3, avg_mod4]
        })

    if not summary_data:
        print("No successful runs to analyze.")
        return

    df_summary = pd.DataFrame(summary_data)
    df_plot = pd.DataFrame(plot_data)
    
    # --- 打印结果表格 ---
    print("\n\n" + "="*30 + " Benchmark Summary " + "="*30)
    # 扩展列宽以显示完整的 ± 信息
    with pd.option_context('display.max_colwidth', 100):
        print(df_summary.drop(columns=[col for col in df_summary if str(col).startswith('M')]).to_string(index=False))
    
    # --- 绘制图表 ---
    sns.set_theme(style="whitegrid")
    
    # 图1: 总耗时对比
    plt.figure(figsize=(12, 7))
    plt.errorbar(df_plot['Size'], df_plot['avg_custom_time'] * 1000, yerr=df_plot['std_custom_time'] * 1000, 
                 marker='o', linestyle='-', label='Custom Solver', capsize=5)
    plt.errorbar(df_plot['Size'], df_plot['avg_scipy_time'] * 1000, yerr=df_plot['std_scipy_time'] * 1000, 
                 marker='x', linestyle='--', label='SciPy (HiGHS)', capsize=5)
    plt.title(f'Solver Performance vs. Problem Size (Avg. of {num_runs} runs)', fontsize=16)
    plt.xlabel('Problem Size (Constraints_eq, Constraints_ub, Variables)', fontsize=12)
    plt.ylabel('Average Total Time (ms)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("benchmark_total_time.png")
    print("\nSaved total time comparison chart to 'benchmark_total_time.png'")

    # 图2: 自定义求解器模块耗时分解
    df_modules = pd.DataFrame(df_plot['modules'].to_list(), index=df_plot['Size'], 
                              columns=['M0', 'M1', 'M2', 'M3', 'M4'])
    df_modules_ms = df_modules * 1000

    plt.figure(figsize=(12, 7))
    ax = df_modules_ms.plot(kind='bar', stacked=True, ax=plt.gca())
    
    total_avg_ms = df_plot['avg_custom_time'] * 1000
    total_std_ms = df_plot['std_custom_time'] * 1000
    ax.errorbar(df_plot.index, total_avg_ms, yerr=total_std_ms, fmt='none', c='black', capsize=5)

    plt.title(f'Custom Solver Module-wise Time Breakdown', fontsize=16)
    plt.xlabel('Problem Size (Constraints_eq, Constraints_ub, Variables)', fontsize=12)
    plt.ylabel('Average Time (ms)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Modules', labels=['Module 0: Standardize', 'Module 1: Prune', 'Module 2: BigM', 'Module 3: Simplex', 'Module 4: Recover'])
    plt.tight_layout()
    plt.savefig("benchmark_module_time.png")
    print("Saved module-wise time breakdown chart to 'benchmark_module_time.png'")
    
    plt.show()


if __name__ == '__main__':
    problem_sizes = [
        (5, 10, 15),       # 小型问题
        (20, 30, 50),      # 中型问题
        (50, 80, 120),     # 较大型问题
        (100, 150, 250),   # 更大型问题
        # (150, 250, 400),   # 大型问题
    ]
    NUM_RUNS_PER_SIZE = 20

    print("==========================================================")
    print("=  LP Solver Benchmark (Module-wise) vs. SciPy (HiGHS)   =")
    print("==========================================================")

    all_runs_data = {size: [] for size in problem_sizes}
    
    for size in problem_sizes:
        size_str = f"({size[0]},{size[1]},{size[2]})"
        with tqdm(total=NUM_RUNS_PER_SIZE, desc=f"Benchmarking size {size_str}") as pbar:
            for i in range(NUM_RUNS_PER_SIZE):
                lp_problem = generate_feasible_lp(size[0], size[1], size[2], seed=42 + i)
                run_results = run_single_benchmark(lp_problem)
                all_runs_data[size].append(run_results)
                pbar.update(1)

    aggregate_and_plot_results(all_runs_data, NUM_RUNS_PER_SIZE)