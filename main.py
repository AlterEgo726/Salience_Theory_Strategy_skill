"""
main.py - ST因子策略回测主入口

用法：
    python main.py -s equal_weight -c False    # (1) 等权无成本
    python main.py -s equal_weight -c True     # (2) 等权有成本
    python main.py -s markowitz -c False       # (3) 马科维茨无成本
    python main.py -s markowitz -c True        # (4) 马科维茨有成本
"""

import argparse
import json
import os
import sys
import time
import pandas as pd
import numpy as np

# 确保能找到本项目的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from st_calculator import calculate_st_simple
from st_backtest import run_backtest, daily_to_monthly_return


def load_config(config_path: str) -> dict:
    """加载配置文件，不存在时使用默认值。"""
    default = {
        "theta": 0.1,
        "delta": 0.7,
        "lookback_months": 36,
        "min_history_days": 504,
        "max_weight": 0.10,
        "shrink": 0.20,
        "cost_rate": 0.002,
        "decile_count": 10,
    }
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user = json.load(f)
            default.update(user)
    return default


def load_data(data_dir: str) -> tuple:
    """
    从 data_dir 加载原始数据文件。

    Returns
    -------
    stock_ret_df : pd.DataFrame  [Stkcd, date, Ri]
    market_ret_df : pd.DataFrame [date, Rm]
    """
    stock_path = os.path.join(data_dir, 'stock_ret_df.csv')
    market_path = os.path.join(data_dir, 'market_ret_df.csv')

    if not os.path.exists(stock_path):
        raise FileNotFoundError(f"找不到个股日收益率文件: {stock_path}")
    if not os.path.exists(market_path):
        raise FileNotFoundError(f"找不到市场日收益率文件: {market_path}")

    print(f"[读取] 个股日收益率: {stock_path}")
    stock_ret = pd.read_csv(stock_path, dtype={'Stkcd': str}, parse_dates=['date'])

    print(f"[读取] 市场日收益率: {market_path}")
    market_ret = pd.read_csv(market_path, parse_dates=['date'])

    return stock_ret, market_ret


def print_results(pivot: pd.DataFrame, strategy: str, cost: bool):
    """打印结果摘要。"""
    mean_returns = pivot.mean(numeric_only=True)

    print()
    print("=" * 60)
    strategy_label = {
        'equal_weight': '等权组合',
        'markowitz': '马科维茨最小方差组合',
    }.get(strategy, strategy)
    cost_label = '考虑交易成本' if cost else '不考虑交易成本'
    print(f"[结果] 策略: {strategy_label} | {cost_label}")
    print("=" * 60)

    for col, val in mean_returns.items():
        print(f"  {col:>8}: {val:>8.4%}")

    print("=" * 60)

    # 年化夏普比
    for col in [1, 10, 'L-H', 'H-L']:
        if col in mean_returns.index:
            col_series = pivot[col]
            sharpe = (
                col_series.mean() / col_series.std() * np.sqrt(12)
                if col_series.std() > 0 else 0.0
            )
            print(f"  {col:>8} 年化夏普: {sharpe:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='ST因子（Salience Theory Value）策略回测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # (1) 等权无成本（最快）
  python main.py -s equal_weight -c False

  # (2) 等权有成本
  python main.py -s equal_weight -c True

  # (3) 马科维茨无成本（运行时间长！）
  python main.py -s markowitz -c False

  # (4) 马科维茨有成本（运行时间长！）
  python main.py -s markowitz -c True
        """,
    )

    parser.add_argument(
        '-s', '--strategy', type=str, default='equal_weight',
        choices=['equal_weight', 'markowitz'],
        help="策略类型: equal_weight | markowitz（默认: equal_weight）",
    )
    parser.add_argument(
        '-c', '--cost', type=str, default='False',
        choices=['True', 'False', 'true', 'false'],
        help="是否考虑交易成本: True | False（默认: False）",
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help="数据目录（默认: 本项目下的 data/input）",
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="输出目录（默认: 本项目下的 output）",
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="配置文件路径（默认: 本项目下的 config.json）",
    )

    args = parser.parse_args()
    cost = args.cost.lower() == 'true'

    # 路径
    base_dir = SCRIPT_DIR
    data_dir = args.data_dir or os.path.join(base_dir, 'data', 'input')
    output_dir = args.output_dir or os.path.join(base_dir, 'output')
    config_path = args.config or os.path.join(base_dir, 'config.json')

    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    cfg = load_config(config_path)

    print("=" * 60)
    print("[ST因子策略回测工具]")
    print("=" * 60)
    print(f"  策略: {args.strategy}")
    print(f"  成本: {'考虑' if cost else '不考虑'}")
    print(f"  参数: theta={cfg['theta']}, delta={cfg['delta']}, "
          f"分组={cfg['decile_count']}")
    print(f"  单边成本率: {cfg['cost_rate']*100:.2f}%")
    if args.strategy == 'markowitz':
        print(f"  马科维茨: 回溯{cfg['lookback_months']}月, "
              f"权重上限{cfg['max_weight']*100:.0f}%, "
              f"最少历史{cfg['min_history_days']}天, "
              f"shrink={cfg['shrink']}")
        print("  [提示] 马科维茨运行时间可能较长，请耐心等待...")
    print("-" * 60)

    # 1. 加载数据
    stock_ret, market_ret = load_data(data_dir)

    # 2. 计算ST因子
    print("[计算] ST因子值...")
    t0 = time.time()
    st_df = calculate_st_simple(
        stock_ret, market_ret,
        theta=cfg['theta'], delta=cfg['delta'],
    )
    print(f"  [完成] ST因子 ({len(st_df)} 条记录, {time.time()-t0:.1f}秒)")

    # 3. 计算月度收益率（从日度复利连乘）
    print("[计算] 月度收益率（日度复利连乘）...")
    monthly_ret = daily_to_monthly_return(stock_ret)
    print(f"  [完成] 月度收益率 ({len(monthly_ret)} 条记录)")

    # 4. 回测
    print("[回测] 运行中...")
    t0 = time.time()
    pivot = run_backtest(
        st_df=st_df,
        monthly_ret=monthly_ret,
        daily_ret=stock_ret if args.strategy == 'markowitz' else None,
        strategy=args.strategy,
        cost=cost,
        decile_count=cfg['decile_count'],
        cost_rate=cfg['cost_rate'],
        lookback_months=cfg['lookback_months'],
        min_history_days=cfg['min_history_days'],
        max_weight=cfg['max_weight'],
        shrink=cfg['shrink'],
    )
    print(f"  [完成] 回测 ({time.time()-t0:.1f}秒)")

    # 5. 输出
    output_path = os.path.join(output_dir, 'pivot.csv')
    pivot.to_csv(output_path)
    print(f"[输出] 结果已保存: {output_path}")

    # 6. 打印结果
    print_results(pivot, args.strategy, cost)


if __name__ == '__main__':
    main()
