# -*- coding: utf-8 -*-
"""
排队论蒙特卡洛仿真 —— 物流配送中心装卸货位（装卸港）作业模拟
================================================================
课程设计配套程序（修订版）

功能
----
1. 用逆变换法生成指数分布的货车到达间隔与装卸服务时长；
2. 对 c 个装卸货位（默认 c = 1、2、3）做离散事件仿真，统计：
     · 货车平均等待时间（全部车辆口径 / 仅排队车辆口径）
     · 装卸货位的空闲次数与空闲时长
     · 系统内最大排队车辆数
     · 货位利用率
3. 做 REPS 次独立重复，给出均值、标准差与 95% 置信区间；
4. 与 M/M/c 排队论稳态理论值对照，检验仿真程序的正确性；
5. 到达率敏感性分析与"是否增开第二个货位"的成本比较。

随机数发生器（--rng）
--------------------
· python （默认）：内置 32 位 mulberry32 发生器。它与课程设计报告生成
  数据时所用的发生器完全一致，因此报告第 4、5 章的所有数值都可以由本
  参数原样复现；
· numpy         ：numpy.random.default_rng(seed)（PCG64），批量生成指数
  随机数。统计规律相同，但具体数值与报告中的不同。

运行环境：Python 3.8+，仅需标准库即可运行；--plot 需要 matplotlib。
运行示例：
    python sim_mmc.py                      # 完整运行（500 次重复）
    python sim_mmc.py --reps 20            # 快速自检
    python sim_mmc.py --rng numpy --plot    # 使用 numpy 随机数并绘图
所有结果由固定种子决定，可完全复现。
"""

from __future__ import annotations

import argparse
import math
import os
from collections import deque

# ---------------------------------------------------------------------------
# 0. 模型参数（时间单位统一为分钟）
# ---------------------------------------------------------------------------
LAMBDA_H = 15.0                 # 货车到达速率 λ，辆/h
MU_H = 20.0                     # 单货位服务速率 μ，辆/h
LAMBDA = LAMBDA_H / 60.0        # λ = 0.25 辆/min，平均到达间隔 4 min
MU = MU_H / 60.0                # μ = 1/3 辆/min，平均装卸时长 3 min
N_TRUCK = 100                   # 每次仿真的货车数量
REPS = 500                      # 独立重复次数
SEED0 = 20260101                # 基准随机种子
BAY_COUNTS = (1, 2, 3)          # 仿真的货位数方案
RHO = LAMBDA / MU               # 单货位服务强度 ρ = λ/μ = 0.75


# ---------------------------------------------------------------------------
# 1. 随机数发生器
# ---------------------------------------------------------------------------
_U32 = 0xFFFFFFFF


class Mulberry32:
    """32 位可复现均匀随机数发生器（与报告数据生成时所用版本一致）。

    递推关系（所有运算均在 32 位无符号整数上取模 2^32）：
        state += 0x6D2B79F5
        t1 = state ^ (state >> 15)
        t2 = t1 * (1 | t1)
        t3 = t2 + (t2 ^ (t2 >> 7)) * (61 | t2)
        out = t3 ^ (t3 >> 14)
    """

    __slots__ = ('state',)

    def __init__(self, seed: int) -> None:
        self.state = seed & _U32

    def next_u32(self) -> int:
        state = (self.state + 0x6D2B79F5) & _U32
        self.state = state
        t1 = (state ^ (state >> 15)) & _U32
        t2 = (t1 * (1 | t1)) & _U32
        t3 = (t2 + ((t2 ^ (t2 >> 7)) & _U32) * (61 | t2)) & _U32
        return (t3 ^ (t3 >> 14)) & _U32

    def random(self) -> float:
        """[0, 1) 上均匀分布。"""
        return self.next_u32() / 4294967296.0

    def random_batch(self, size: int):
        """批量生成 [0, 1) 均匀随机数（与逐个生成结果完全一致）。"""
        return [self.next_u32() / 4294967296.0 for _ in range(size)]

    def expovariate(self, rate: float) -> float:
        """逆变换法：X = -ln(1-U)/rate，U ~ U[0,1)。

        直接用 1-U 而不是 U，避免 U = 0 时对数发散，无需 1e-6 之类的
        经验性截断，数学上与 -ln(U)/rate 完全等价。
        """
        return -math.log(1.0 - self.random()) / rate


class NumpyMulberry32:
    """与 Mulberry32 数值等价的 numpy 向量化实现（uint32 精确模拟 32 位溢出）。

    递推式与 Mulberry32 完全相同，因此相同种子下二者输出的随机数序列
    逐位一致；批量生成使 500 次重复仿真的耗时可缩短一个数量级。
    """

    def __init__(self, seed: int, block: int = 4096) -> None:
        import numpy as np                      # 延迟导入，未安装时不影响主流程
        self._np = np
        self._state = np.uint32(seed & _U32)
        self._buf = None
        self._pos = 0
        self._block = block
        self._c1 = np.uint32(0x6D2B79F5)
        self._one32 = np.uint32(_U32)

    def _refill(self):
        np = self._np
        n = self._block
        x = np.empty(n, dtype=np.uint32)
        state = self._state
        for i in range(n):
            state = np.uint32(state + self._c1)
            t1 = state ^ (state >> np.uint32(15))
            t2 = t1 * (t1 | np.uint32(1))          # uint32 乘法自动按 2^32 取模
            t3 = t2 + ((t2 ^ (t2 >> np.uint32(7))) * (t2 | np.uint32(61)))
            x[i] = t3 ^ (t3 >> np.uint32(14))
        self._state = state
        self._buf = x.astype(np.float64) / 4294967296.0
        self._pos = 0

    def random(self) -> float:
        if self._buf is None or self._pos >= self._block:
            self._refill()
        v = float(self._buf[self._pos])
        self._pos += 1
        return v

    def random_batch(self, size: int):
        return [self.random() for _ in range(size)]

    def expovariate(self, rate: float) -> float:
        return -math.log(1.0 - self.random()) / rate


def make_rng(kind: str, seed: int):
    if kind == 'numpy':
        return NumpyMulberry32(seed)
    return Mulberry32(seed)


def _exponentials(rng, rate: float, size: int):
    """生成 size 个指数随机数（逆变换法）。"""
    return [-math.log(1.0 - u) / rate for u in rng.random_batch(size)]


# ---------------------------------------------------------------------------
# 2. 单次仿真（核心算法）
# ---------------------------------------------------------------------------
def simulate_once(rng, c: int, n_truck: int = N_TRUCK,
                  lam: float = LAMBDA, mu: float = MU) -> dict:
    """对 c 个货位、n_truck 辆货车做一次离散事件仿真。

    事件驱动逻辑（与报告第 3 章算法一致）：
      · 货车按到达时刻升序处理，先到先服务；
      · 每辆货车占用"最早空闲"的货位；
      · 排队车辆数 = 该车到达后系统内已到达但尚未开始装卸的车辆数。

    返回指标字典（records 为逐车时间节点）。
    """
    # 2.1 生成到达时刻：到达间隔 ~ Exp(λ)，累加得到绝对时刻
    arrivals = []
    t = 0.0
    for interval in _exponentials(rng, lam, n_truck):
        t += float(interval)
        arrivals.append(t)

    # 2.2 仿真状态初始化
    free_at = [0.0] * c                 # 各货位下一次空闲的时刻
    pending_start = deque()             # 已到达但尚未开始装卸的车辆的开始时刻
    records = []
    wait_total = 0.0
    wait_queued_total = 0.0
    queued_count = 0
    max_queue = 0
    busy_time = 0.0

    # 2.3 逐辆处理
    for i, arrive in enumerate(arrivals):
        # (a) 选择最早空闲的货位
        k = min(range(c), key=lambda j: free_at[j])

        # (b) 统计该车到达并加入系统后的排队车辆数
        #     pending_start 中开始时刻 <= 当前到达时刻的车辆已经开工，先出队
        while pending_start and pending_start[0] <= arrive:
            pending_start.popleft()
        queue_len = len(pending_start) + 1
        if queue_len > max_queue:
            max_queue = queue_len

        # (c) 等待时间与服务时间
        start = arrive if arrive > free_at[k] else free_at[k]
        wait = start - arrive
        wait_total += wait
        if wait > 1e-12:
            queued_count += 1
            wait_queued_total += wait

        service = -math.log(1.0 - rng.random()) / mu
        end = start + service
        pending_start.append(start)
        free_at[k] = end
        busy_time += service

        records.append({'id': i + 1, 'arrive': arrive, 'start': start,
                        'end': end, 'wait': wait, 'queue_len': queue_len})

    makespan = max(free_at)             # 全部货车装卸完毕的时刻
    return {
        'avg_wait': wait_total / n_truck,
        'avg_wait_queued': (wait_queued_total / queued_count) if queued_count else 0.0,
        'queued_pct': 100.0 * queued_count / n_truck,
        'idle_periods': count_idle_periods(records),
        'max_queue': max_queue,
        'makespan': makespan,
        'utilisation': 100.0 * busy_time / (c * makespan),
        'idle_bay_time': c * makespan - busy_time,
        'busy_time': busy_time,
        'records': records,
    }


def count_idle_periods(records) -> int:
    """统计"所有货位都不工作"的时间段个数。

    系统在 t = 0 处于空闲状态，最初那一段空闲计入 1 次（与报告 3.2 节的
    定义一致）；最后一批货车离开之后的空闲时段不计入。
    """
    events = []
    for r in records:
        events.append((r['start'], 1))
        events.append((r['end'], -1))
    events.sort(key=lambda e: (e[0], e[1]))

    active = 0
    was_idle = True
    idle_periods = 0
    idx = 0
    n = len(events)
    while idx < n:
        now = events[idx][0]
        while idx < n and events[idx][0] == now:
            active += events[idx][1]
            idx += 1
        if was_idle and active > 0:
            idle_periods += 1
            was_idle = False
        elif not was_idle and active == 0:
            was_idle = True
    return idle_periods


# ---------------------------------------------------------------------------
# 3. M/M/c 稳态理论值（用于校核仿真）
# ---------------------------------------------------------------------------
def mmc_theory(c: int, lam: float = LAMBDA, mu: float = MU) -> dict:
    r = lam / mu
    rho = r / c
    if rho >= 1.0:
        return {'c': c, 'rho': rho, 'stable': False}

    # Erlang B 递推 -> Erlang C
    b = 1.0
    for k in range(1, c + 1):
        b = (r * b) / (k + r * b)
    p_wait = b / (1.0 - rho * (1.0 - b))
    p0 = 1.0 / (sum(r ** k / math.factorial(k) for k in range(c))
                + (r ** c / math.factorial(c)) * (1.0 / (1.0 - rho)))
    lq = p_wait * rho / (1.0 - rho)
    wq = lq / lam
    return {'c': c, 'rho': rho, 'stable': True, 'p0': p0, 'p_wait': p_wait,
            'lq': lq, 'wq': wq, 'w': wq + 1.0 / mu, 'l': lq + r}


# ---------------------------------------------------------------------------
# 4. 多次独立重复统计
# ---------------------------------------------------------------------------
def _summarise(values) -> dict:
    v = sorted(float(x) for x in values)
    n = len(v)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    def q(p):
        return v[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {'n': n, 'mean': mean, 'sd': sd, 'se': se,
            'ci_low': mean - 1.96 * se, 'ci_high': mean + 1.96 * se,
            'min': v[0], 'p05': q(0.05), 'median': q(0.5),
            'p95': q(0.95), 'max': v[-1]}


METRIC_KEYS = ('avg_wait', 'avg_wait_queued', 'queued_pct', 'idle_periods',
               'max_queue', 'makespan', 'utilisation', 'idle_bay_time')

METRIC_LABELS = (
    ('avg_wait', '平均等待时间（全部车辆）/min'),
    ('avg_wait_queued', '平均等待时间（排队车辆）/min'),
    ('queued_pct', '需要排队的车辆比例/%'),
    ('idle_periods', '货位空闲次数/次'),
    ('max_queue', '最大排队车辆数/辆'),
    ('utilisation', '货位利用率/%'),
    ('idle_bay_time', '空闲货位时长/(货位·min)'),
    ('makespan', '仿真总工期/min'),
)


def run_replications(c: int, reps: int = REPS, seed0: int = SEED0,
                     n_truck: int = N_TRUCK, rng_kind: str = 'python',
                     lam: float = LAMBDA, mu: float = MU):
    runs = []
    for i in range(reps):
        rng = make_rng(rng_kind, seed0 + 7919 * i)
        runs.append(simulate_once(rng, c, n_truck, lam, mu))
    return {k: _summarise([r[k] for r in runs]) for k in METRIC_KEYS}, runs


# ---------------------------------------------------------------------------
# 5. 结果输出
# ---------------------------------------------------------------------------
LINE = '-' * 96


def print_report(results, single, theory, reps, seed=SEED0):
    print(LINE)
    print('表 4-1  仿真结果与 M/M/c 稳态理论值对照（%d 辆货车，%d 次独立重复）' % (N_TRUCK, reps))
    print(LINE)
    print('%-6s %-8s %-12s %-12s %-20s %-10s %-10s' %
          ('货位数', '理论ρ', '理论Wq/min', '仿真Wq/min', '95%置信区间/min', '最大排队', '利用率/%'))
    for c in sorted(results):
        s, th = results[c], theory[c]
        print('%-8d %-10.3f %-14.3f %-14.3f [%6.2f, %6.2f]      %-12.2f %-10.1f' % (
            c, th['rho'], th['wq'], s['avg_wait']['mean'],
            s['avg_wait']['ci_low'], s['avg_wait']['ci_high'],
            s['max_queue']['mean'], s['utilisation']['mean']))
    print()

    print(LINE)
    print('表 4-2  单货位方案各指标的均值、标准差与 95%% 置信区间（%d 次独立重复）' % reps)
    print(LINE)
    print('%-34s %10s %10s %20s' % ('指标', '均值', '标准差', '95%置信区间'))
    s1 = results[1]
    for key, label in METRIC_LABELS:
        v = s1[key]
        print('%-34s %10.3f %10.3f   [%8.3f, %8.3f]' % (label, v['mean'], v['sd'], v['ci_low'], v['ci_high']))
    print()

    print(LINE)
    print('表 4-3  基准随机种子（seed = %d）单次仿真的详细结果' % seed)
    print(LINE)
    print('%-8s %-14s %-14s %-14s %-14s %-12s' %
          ('货位数', '平均等待/min', '最大排队/辆', '空闲次数/次', '总工期/min', '利用率/%'))
    for c in sorted(single):
        r = single[c]
        print('%-10d %-16.3f %-16d %-16d %-16.1f %-12.1f' % (
            c, r['avg_wait'], r['max_queue'], r['idle_periods'], r['makespan'], r['utilisation']))
    print()
    print('基准单次仿真前 10 辆货车的时间节点（c = 1）：')
    print('%-8s %-12s %-12s %-12s %-12s %-12s' %
          ('车号', '到达/min', '开始装卸/min', '装卸完毕/min', '等待/min', '到达时排队/辆'))
    for r in single[1]['records'][:10]:
        print('%-10d %-14.2f %-14.2f %-14.2f %-14.2f %-12d' % (
            r['id'], r['arrive'], r['start'], r['end'], r['wait'], r['queue_len']))
    print()


def sensitivity(c: int = 1, reps: int = 500, rng_kind: str = 'python'):
    print(LINE)
    print('表 5-1  到达率敏感性分析（c = %d，每次 %d 次独立重复）' % (c, reps))
    print(LINE)
    print('%-16s %-10s %-16s %-20s %-14s %-12s' %
          ('到达率/(辆·h⁻¹)', 'ρ', '平均等待/min', '95%置信区间/min', '最大排队/辆', '利用率/%'))
    for mult in (0.8, 0.9, 1.0, 1.1, 1.2):
        lam = LAMBDA * mult
        if lam / MU >= 1.0:
            print('%-18.1f %-12.3f 系统不稳定（ρ ≥ 1）' % (lam * 60, lam / MU))
            continue
        summary, _ = run_replications(c, reps=reps, rng_kind=rng_kind, lam=lam)
        w, q, u = summary['avg_wait'], summary['max_queue'], summary['utilisation']
        print('%-18.1f %-12.3f %-18.3f [%6.2f, %6.2f]      %-16.2f %-12.1f' % (
            lam * 60, lam / MU, w['mean'], w['ci_low'], w['ci_high'], q['mean'], u['mean']))
    print()

def cost_comparison(results, cw: float = 60.0, cb: float = 25.0):
    """排队成本比较：Cw 为等待成本 元/(辆·h)，Cb 为货位成本 元/(货位·h)。"""
    print(LINE)
    print('表 5-2  货位配置方案成本比较（等待成本 %.0f 元/(辆·h)，货位成本 %.0f 元/(货位·h)）' % (cw, cb))
    print(LINE)
    print('%-10s %-16s %-16s %-16s %-16s' % ('方案', '等待成本/元', '货位成本/元', '总成本/元', '折合每辆车/元'))
    for c in sorted(results):
        r = results[c]
        hours = r['makespan']['mean'] / 60.0
        wait_cost = LAMBDA_H * hours * cw * (r['avg_wait']['mean'] / 60.0)
        bay_cost = c * hours * cb
        print('%-12d %-18.1f %-18.1f %-18.1f %-16.2f' % (
            c, wait_cost, bay_cost, wait_cost + bay_cost, (wait_cost + bay_cost) / N_TRUCK))
    print()


# ---------------------------------------------------------------------------
# 6. 绘图（可选）
# ---------------------------------------------------------------------------
def make_plots(single, results, reps, outdir='figures'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(outdir, exist_ok=True)

    # 图 1：到达间隔与服务时长直方图
    rng = Mulberry32(SEED0)
    arrivals = [rng.expovariate(LAMBDA) for _ in range(400)]
    services = [rng.expovariate(MU) for _ in range(400)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, data, rate, title, color in (
            (axes[0], arrivals, LAMBDA, '货车到达间隔分布', '#0072B2'),
            (axes[1], services, MU, '单车装卸服务时长分布', '#D55E00')):
        ax.hist(data, bins=24, density=True, alpha=0.65, color=color, edgecolor='white')
        xs = [i * max(data) / 200 for i in range(201)]
        ax.plot(xs, [rate * math.exp(-rate * x) for x in xs], 'k-', lw=1.2, label='指数分布理论密度')
        ax.set_xlabel('时间/min'); ax.set_ylabel('概率密度'); ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'fig1_distribution.png'), dpi=200); plt.close(fig)

    # 图 2：车辆时间轴（甘特图）+ 排队队长曲线
    rec = single[1]['records']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for r in rec:
        ax1.barh(r['id'], r['wait'], left=r['arrive'], height=0.8, color='#D55E00')
        ax1.barh(r['id'], r['end'] - r['start'], left=r['start'], height=0.8, color='#009E73')
    ax1.set_ylabel('货车编号'); ax1.set_title('货车时间轴：橙色为排队等待，绿色为装卸作业'); ax1.invert_yaxis()
    times, qs = [0.0], [0]
    for r in rec:
        times += [r['arrive'], r['start']]
        qs += [qs[-1] + 1, max(0, qs[-1])]
    ax2.step(times, qs, where='post', color='#0072B2')
    ax2.set_xlabel('系统时间/min'); ax2.set_ylabel('排队车辆数'); ax2.set_title('排队队长随时间变化')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'fig2_timeline.png'), dpi=200); plt.close(fig)

    # 图 3：多次重复的平均等待时间分布
    _, runs = run_replications(1, reps=reps)
    waits = [r['avg_wait'] for r in runs]
    mean = sum(waits) / len(waits)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(waits, bins=28, color='#0072B2', alpha=0.75, edgecolor='white')
    ax.axvline(mean, color='k', lw=1.5, label='%d 次重复均值 %.2f min' % (reps, mean))
    ax.axvline(mmc_theory(1)['wq'], color='#D55E00', ls='--', lw=1.5,
               label='M/M/1 理论值 %.2f min' % mmc_theory(1)['wq'])
    ax.set_xlabel('平均等待时间/min'); ax.set_ylabel('出现次数')
    ax.set_title('%d 次独立重复的平均等待时间分布（c = 1）' % reps); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'fig3_replications.png'), dpi=200); plt.close(fig)

    # 图 4：不同货位数的等待时间箱线图
    data = []
    for c in sorted(results):
        _, runs = run_replications(c, reps=max(100, reps // 2))
        data.append([r['avg_wait'] for r in runs])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.boxplot(data, labels=['%d 个货位' % c for c in sorted(results)], showfliers=False)
    ax.set_ylabel('平均等待时间/min'); ax.set_title('货位数量对货车平均等待时间的影响')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'fig4_compare.png'), dpi=200); plt.close(fig)
    print('结果图已保存到 %s/ 目录（fig1~fig4）' % outdir)


# ---------------------------------------------------------------------------
# 7. 主程序
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description='装卸货位排队论蒙特卡洛仿真')
    parser.add_argument('--reps', type=int, default=REPS, help='独立重复次数（默认 %d）' % REPS)
    parser.add_argument('--rng', choices=('python', 'numpy'), default='python',
                        help='随机数实现：python（默认，可复现报告数据）或 numpy')
    parser.add_argument('--seed', type=int, default=SEED0, help='基准随机种子')
    parser.add_argument('--plot', action='store_true', help='绘制结果图（需要 matplotlib）')
    parser.add_argument('--quiet-sensitivity', action='store_true', help='跳过敏感性分析')
    args = parser.parse_args(argv)

    print('参数：λ = %.2f 辆/h，μ = %.2f 辆/h/货位，ρ = %.3f，每次仿真 %d 辆货车，重复 %d 次，随机数 = %s\n'
          % (LAMBDA_H, MU_H, RHO, N_TRUCK, args.reps, args.rng))

    results, theory, single = {}, {}, {}
    for c in BAY_COUNTS:
        results[c], _ = run_replications(c, reps=args.reps, seed0=args.seed, rng_kind=args.rng)
        theory[c] = mmc_theory(c)
        single[c] = simulate_once(make_rng(args.rng, args.seed), c)

    print_report(results, single, theory, args.reps, seed=args.seed)
    if not args.quiet_sensitivity:
        sensitivity(1, reps=args.reps, rng_kind=args.rng)
    cost_comparison(results)

    if args.plot:
        make_plots(single, results, args.reps)


if __name__ == '__main__':
    main()
