"""
st_backtest.py — ST因子策略回测模块

支持四种策略组合：
    ① 等权无成本    ② 等权有成本
    ③ 马科维茨无成本  ④ 马科维茨有成本

核心流程：
    数据准备 → 分10组 → 按策略算收益 → 扣除成本(可选) → 输出pivot
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize as _minimize
from typing import Optional, Dict, Tuple

# =====================================================================
# 全局缓存（避免马科维茨权重重复计算）
# =====================================================================
_WEIGHT_CACHE: Dict[Tuple, dict] = {}


# =====================================================================
# 数据准备
# =====================================================================

def daily_to_monthly_return(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """
    将日度个股收益率复利连乘为月度收益率。

    Parameters
    ----------
    daily_ret : pd.DataFrame
        需包含列: ['Stkcd', 'date', 'Ri']

    Returns
    -------
    pd.DataFrame
        ['Stkcd', 'month', 'ret'] — 月度收益率（小数形式）
    """
    df = daily_ret.copy()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['ret_m'] = df.groupby(['Stkcd', 'month'])['Ri'].transform(
        lambda x: (1 + x).prod() - 1
    )
    monthly = df[['Stkcd', 'month', 'ret_m']].drop_duplicates(
        subset=['Stkcd', 'month']
    ).reset_index(drop=True)
    return monthly.rename(columns={'ret_m': 'ret'})


# =====================================================================
# 分组
# =====================================================================

def group_deciles(
    st_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    decile_count: int = 10,
) -> pd.DataFrame:
    """
    合并ST因子和月度收益，构造下一月收益，并按ST分组。

    Parameters
    ----------
    st_df : pd.DataFrame
        ST因子值 [Stkcd, month, ST]
    ret_df : pd.DataFrame
        月度收益率 [Stkcd, month, ret]
    decile_count : int
        每月分组数，默认10

    Returns
    -------
    pd.DataFrame
        [Stkcd, month, ST, ret, ret_next, decile]
    """
    # 合并
    df = pd.merge(st_df, ret_df, on=['Stkcd', 'month'])

    # 构造下一月收益
    df = df.sort_values(['Stkcd', 'month'])
    df['ret_next'] = df.groupby('Stkcd')['ret'].shift(-1)
    df = df.dropna(subset=['ret_next', 'ST'])

    # 分组
    df['decile'] = df.groupby('month')['ST'].transform(
        lambda x: pd.qcut(x, decile_count, labels=False, duplicates='drop') + 1
    )

    return df


# =====================================================================
# 等权组合
# =====================================================================

def equal_weight_portfolio_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每月每个分组的等权平均收益。

    Parameters
    ----------
    df : pd.DataFrame
        包含 [month, decile, ret_next]

    Returns
    -------
    pd.DataFrame
        [month, decile, ret_next]
    """
    return df.groupby(['month', 'decile'])['ret_next'].mean().reset_index()


def _equal_weight_turnover(df: pd.DataFrame, decile: int) -> pd.Series:
    """
    计算等权组合的月度调仓换手率（双边）。

    Parameters
    ----------
    df : pd.DataFrame
        包含 [Stkcd, month, decile, ret]
    decile : int
        目标分组编号

    Returns
    -------
    pd.Series
        index=month, value=双边换手率
    """
    sub = df[df['decile'] == decile].copy()
    prev = {}
    out = {}
    for m, g in sub.groupby('month'):
        s = g.set_index('Stkcd')['ret'].to_dict()
        n = len(s)
        if n == 0:
            out[m], prev = 0.0, {}
            continue
        tgt = {k: 1.0 / n for k in s}
        if not prev:
            out[m] = 1.0
        else:
            all_keys = set(prev) | set(s)
            gross = {k: prev.get(k, 0.0) * (1 + s.get(k, 0.0)) for k in all_keys}
            denom = sum(gross.values()) or 1.0
            drft = {k: v / denom for k, v in gross.items()}
            out[m] = sum(
                abs(tgt.get(k, 0.0) - drft.get(k, 0.0)) for k in all_keys
            )
        prev = tgt
    return pd.Series(out, name=decile)


# =====================================================================
# 马科维茨最小方差组合
# =====================================================================

def _minvar_weights_from_daily(
    daily_pivot: pd.DataFrame,
    max_weight: float = 0.10,
    shrink: float = 0.20,
) -> dict:
    """
    从日度收益率宽表计算最小方差权重。

    Parameters
    ----------
    daily_pivot : pd.DataFrame
        行=日期, 列=股票代码, 值为日收益率
    max_weight : float
        单股权重上限
    shrink : float
        对角收缩系数

    Returns
    -------
    dict
        {股票代码: 权重}
    """
    daily_cov = daily_pivot.cov().to_numpy(dtype=float)
    if daily_cov.ndim == 0 or daily_cov.shape == (1, 1):
        return {daily_pivot.columns[0]: 1.0}

    cov = daily_cov * 21  # 月度化协方差
    diag = np.diag(np.diag(cov))
    sigma = (1.0 - shrink) * cov + shrink * diag + np.eye(cov.shape[0]) * 1e-8
    n = sigma.shape[0]
    cap = max(max_weight, 1.0 / n)

    def obj(w):
        return float(w @ sigma @ w)

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, cap) for _ in range(n)]
    x0 = np.repeat(1.0 / n, n)
    res = _minimize(
        obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        return {daily_pivot.columns[i]: 1.0 / n for i in range(n)}
    w = np.clip(res.x, 0.0, cap)
    w = w / w.sum()
    return {daily_pivot.columns[i]: w[i] for i in range(n)}


def _compute_leg_weights(
    df: pd.DataFrame,
    daily_ret: pd.DataFrame,
    month: pd.Timestamp,
    decile: int,
    lookback_months: int = 36,
    min_history_days: int = 504,
    max_weight: float = 0.10,
    shrink: float = 0.20,
) -> dict:
    """
    计算某个月份×分组的马科维茨最小方差权重。

    Parameters
    ----------
    df : pd.DataFrame
        包含 [Stkcd, month, decile]
    daily_ret : pd.DataFrame
        日度收益率 [Stkcd, Trddt, daily_return]
    month : pd.Timestamp
        目标月份
    decile : int
        目标分组
    lookback_months : int
        回溯月数
    min_history_days : int
        最少历史交易日数（不足则剔除）
    max_weight, shrink : float
        传递给 _minvar_weights_from_daily

    Returns
    -------
    dict
        {Stkcd: weight} 或 {}（无合格股票时）
    """
    codes = df[(df['month'] == month) & (df['decile'] == decile)][
        'Stkcd'
    ].unique().tolist()
    if not codes:
        return {}

    start = month - pd.DateOffset(months=lookback_months)
    hist = daily_ret[
        (daily_ret['Stkcd'].isin(codes))
        & (daily_ret['Trddt'] >= start)
        & (daily_ret['Trddt'] < month)
    ]
    if hist.empty:
        return {}

    wide = hist.pivot(
        index='Trddt', columns='Stkcd', values='daily_return'
    ).sort_index()
    counts = wide.notna().sum()
    eligible = counts[counts >= min_history_days].index.tolist()
    if len(eligible) < 2:
        return {}

    return _minvar_weights_from_daily(
        wide[eligible], max_weight=max_weight, shrink=shrink
    )


def _get_leg_weights(
    df: pd.DataFrame,
    daily_ret: pd.DataFrame,
    month: pd.Timestamp,
    decile: int,
    lookback_months: int = 36,
    min_history_days: int = 504,
    max_weight: float = 0.10,
    shrink: float = 0.20,
) -> dict:
    """带缓存的权重获取（避免重复计算相同的 month×decile）。"""
    key = (month, decile, lookback_months, min_history_days, max_weight, shrink)
    if key not in _WEIGHT_CACHE:
        _WEIGHT_CACHE[key] = _compute_leg_weights(
            df, daily_ret, month, decile,
            lookback_months, min_history_days, max_weight, shrink,
        )
    return _WEIGHT_CACHE[key]


def markowitz_portfolio_returns(
    df: pd.DataFrame,
    daily_ret: pd.DataFrame,
    lookback_months: int = 36,
    min_history_days: int = 504,
    max_weight: float = 0.10,
    shrink: float = 0.20,
) -> pd.DataFrame:
    """
    对每个 month×decile 计算马科维茨最小方差组合收益。

    Parameters
    ----------
    df : pd.DataFrame
        包含 [Stkcd, month, decile, ret_next]
    daily_ret : pd.DataFrame
        日度收益率 [Stkcd, Trddt, daily_return]
    lookback_months, min_history_days, max_weight, shrink
        同 _get_leg_weights

    Returns
    -------
    pd.DataFrame
        [month, decile, ret_next]
    """
    rows = []
    pairs = df[['month', 'decile']].drop_duplicates()
    for _, row in pairs.iterrows():
        weights = _get_leg_weights(
            df, daily_ret, row['month'], row['decile'],
            lookback_months, min_history_days, max_weight, shrink,
        )
        if not weights:
            r = np.nan
        else:
            ret_dict = df[
                (df['month'] == row['month']) & (df['decile'] == row['decile'])
            ].set_index('Stkcd')['ret_next'].to_dict()
            r = float(
                sum(w * ret_dict.get(stkcd, 0.0) for stkcd, w in weights.items())
            )
        rows.append({'month': row['month'], 'decile': row['decile'], 'ret_next': r})
    return pd.DataFrame(rows)


def markowitz_turnover(
    df: pd.DataFrame,
    daily_ret: pd.DataFrame,
    decile: int,
    lookback_months: int = 36,
    min_history_days: int = 504,
    max_weight: float = 0.10,
    shrink: float = 0.20,
) -> pd.Series:
    """
    计算马科维茨组合的月度调仓换手率（双边）。

    Parameters
    ----------
    df : pd.DataFrame
        包含 [Stkcd, month, decile]
    daily_ret : pd.DataFrame
        日度收益率
    decile : int
        目标分组
    其他参数同 _get_leg_weights

    Returns
    -------
    pd.Series
        index=month, value=双边换手率（NaN表示当月无合格权重）
    """
    months = sorted(df[df['decile'] == decile]['month'].unique())
    prev = {}
    out = {}
    for m in months:
        weights = _get_leg_weights(
            df, daily_ret, m, decile,
            lookback_months, min_history_days, max_weight, shrink,
        )
        if not weights:
            out[m] = np.nan
            prev = {}
            continue
        if not prev:
            out[m] = 1.0
        else:
            all_keys = set(prev) | set(weights)
            out[m] = float(
                sum(abs(weights.get(k, 0.0) - prev.get(k, 0.0)) for k in all_keys)
            )
        prev = weights
    return pd.Series(out, name=decile)


# =====================================================================
# 成本计算
# =====================================================================

def apply_cost(
    pivot: pd.DataFrame,
    df: pd.DataFrame,
    strategy: str,
    daily_ret: Optional[pd.DataFrame] = None,
    cost_rate: float = 0.002,
    **kwargs,
) -> pd.DataFrame:
    """
    扣除交易成本，返回包含成本后的 pivot。

    Parameters
    ----------
    pivot : pd.DataFrame
        [month, 1..10, L-H, H-L]
    df : pd.DataFrame
        分组数据 [Stkcd, month, decile, ret]
    strategy : str
        'equal_weight' 或 'markowitz'
    daily_ret : pd.DataFrame or None
        马科维茨策略需要
    cost_rate : float
        单边交易成本
    **kwargs
        马科维茨其他参数

    Returns
    -------
    pd.DataFrame
        考虑成本后的 pivot（1, 10, L-H, H-L 已减成本）
    """
    result = pivot.copy()

    if strategy == 'equal_weight':
        to1 = _equal_weight_turnover(df, 1)
        to10 = _equal_weight_turnover(df, 10)
    else:
        to1 = markowitz_turnover(df, daily_ret, 1, **kwargs)
        to10 = markowitz_turnover(df, daily_ret, 10, **kwargs)
        to1 = to1.fillna(1.0)
        to10 = to10.fillna(1.0)

    # 对齐月份
    common_months = result.index.intersection(to1.index).intersection(to10.index)

    for col in [1, 10, 'L-H', 'H-L']:
        if col in result.columns:
            result.loc[common_months, col] = (
                result.loc[common_months, col]
                - cost_rate * (to1.reindex(common_months) + to10.reindex(common_months))
            )

    return result


# =====================================================================
# 主回测函数
# =====================================================================

def build_pivot(
    portfolio_ret: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    strategy: str = 'equal_weight',
    daily_ret: Optional[pd.DataFrame] = None,
    cost: bool = False,
    cost_rate: float = 0.002,
    **kwargs,
) -> pd.DataFrame:
    """
    从组合收益构建完整的pivot DataFrame，包含L-H/H-L及可选成本。

    Parameters
    ----------
    portfolio_ret : pd.DataFrame
        含 [month, decile, ret_next]
    df : pd.DataFrame or None
        分组数据（成本计算需要）
    strategy : str
        'equal_weight' 或 'markowitz'
    daily_ret : pd.DataFrame or None
        马科维茨需要
    cost : bool
        是否考虑交易成本
    cost_rate : float
        单边交易成本比率
    **kwargs
        传递给成本计算/马科维茨的其他参数

    Returns
    -------
    pd.DataFrame
        index=month, columns=[1..10, 'L-H', 'H-L']
    """
    pivot = portfolio_ret.pivot(index='month', columns='decile', values='ret_next')

    # 补全缺失列
    for col in range(1, 11):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot = pivot[sorted(pivot.columns)]

    # L-H / H-L
    pivot['L-H'] = pivot[1] - pivot[10]
    pivot['H-L'] = pivot[10] - pivot[1]

    # 成本
    if cost:
        if df is None:
            raise ValueError("cost=True 时需提供 df 参数")
        pivot = apply_cost(
            pivot, df, strategy, daily_ret, cost_rate, **kwargs
        )
        # 确保1、10的列也已扣除成本（apply_cost中已处理）

    return pivot


# =====================================================================
# 完整回测流程
# =====================================================================

def run_backtest(
    st_df: pd.DataFrame,
    monthly_ret: pd.DataFrame,
    daily_ret: Optional[pd.DataFrame] = None,
    strategy: str = 'equal_weight',
    cost: bool = False,
    decile_count: int = 10,
    cost_rate: float = 0.002,
    lookback_months: int = 36,
    min_history_days: int = 504,
    max_weight: float = 0.10,
    shrink: float = 0.20,
) -> pd.DataFrame:
    """
    一键运行完整回测流程。

    Parameters
    ----------
    st_df : pd.DataFrame
        ST因子值 [Stkcd, month, ST]
    monthly_ret : pd.DataFrame
        月度收益率 [Stkcd, month, ret]
    daily_ret : pd.DataFrame or None
        日度收益率 [Stkcd, date, Ri]（马科维茨需要）
    strategy : str
        'equal_weight' 或 'markowitz'
    cost : bool
        是否考虑交易成本
    decile_count : int
        分组数
    cost_rate : float
        单边交易成本
    lookback_months : int
        马科维茨回溯月数
    min_history_days : int
        最少历史交易日数
    max_weight : float
        单股权重上限
    shrink : float
        协方差对角收缩系数

    Returns
    -------
    pd.DataFrame
        pivot结果 + 控制台打印平均收益
    """
    _WEIGHT_CACHE.clear()

    # 1. 分组
    df = group_deciles(st_df, monthly_ret, decile_count=decile_count)

    # 2. 计算组合收益
    if strategy == 'equal_weight':
        portfolio_ret = equal_weight_portfolio_returns(df)
        kwargs = {}
    else:
        if daily_ret is None:
            raise ValueError("马科维茨策略需要提供 daily_ret 数据")
        # 需将daily_ret转为[Stkcd, Trddt, daily_return]格式
        dr = daily_ret.rename(columns={'date': 'Trddt', 'Ri': 'daily_return'})
        portfolio_ret = markowitz_portfolio_returns(
            df, dr,
            lookback_months=lookback_months,
            min_history_days=min_history_days,
            max_weight=max_weight,
            shrink=shrink,
        )
        kwargs = {
            'lookback_months': lookback_months,
            'min_history_days': min_history_days,
            'max_weight': max_weight,
            'shrink': shrink,
        }

    # 3. 构建pivot
    pivot = build_pivot(
        portfolio_ret,
        df=df if cost else None,
        strategy=strategy,
        daily_ret=dr if strategy == 'markowitz' and cost else None,
        cost=cost,
        cost_rate=cost_rate,
        **kwargs,
    )

    # 4. 填充NaN（当月无组合时视为0收益）
    pivot = pivot.fillna(0.0)

    return pivot
