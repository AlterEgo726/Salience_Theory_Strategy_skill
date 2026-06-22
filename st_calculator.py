"""
st_calculator.py — ST因子（Salience Theory Value）计算模块

将原始的 ST因子计算.py 重构为可导入的函数模块。

输入：
    stock_ret_df: 个股日收益率DataFrame [Stkcd, date, Ri]
    market_ret_df: 市场日收益率DataFrame [date, Rm]

输出：
    st_df: 个股月度ST因子值 [Stkcd, month, ST]
"""

import numpy as np
import pandas as pd


def calculate_st_simple(
    stock_ret_df: pd.DataFrame,
    market_ret_df: pd.DataFrame,
    theta: float = 0.1,
    delta: float = 0.7,
) -> pd.DataFrame:
    """
    计算个股月度ST因子值（Salience Theory Value）。

    Parameters
    ----------
    stock_ret_df : pd.DataFrame
        个股日收益率，包含列：['Stkcd', 'date', 'Ri']
        - Stkcd: 股票代码（字符串）
        - date: 交易日（datetime）
        - Ri: 个股日收益率（小数形式，如 0.01 = 1%）
    market_ret_df : pd.DataFrame
        市场日收益率，包含列：['date', 'Rm']
        - date: 交易日（datetime）
        - Rm: 市场等权日收益率（小数形式）
    theta : float, default=0.1
        凸显函数中的平滑参数。theta 越小，对小幅偏离的区分越敏感。
    delta : float, default=0.7
        凸显权重衰减参数（0<δ<1）。δ 越小，高凸显度的权重衰减越快。

    Returns
    -------
    pd.DataFrame
        包含列 ['Stkcd', 'month', 'ST']，每行代表一只股票在一个月内的ST因子值。
    """
    # 1. 合并数据
    merged = pd.merge(stock_ret_df, market_ret_df, on='date')
    merged['month'] = merged['date'].dt.to_period('M').dt.to_timestamp()

    # 2. 计算凸显度
    merged['凸显度'] = np.abs(merged['Ri'] - merged['Rm']) / (
        np.abs(merged['Ri']) + np.abs(merged['Rm']) + theta
    )

    # 3. 按凸显度排序（method='dense'，凸显相同时同排名）
    merged = merged.sort_values(
        ['Stkcd', 'month', '凸显度'], ascending=[True, True, False]
    )
    merged['k'] = merged.groupby(['Stkcd', 'month'])['凸显度'].rank(
        method='dense', ascending=False
    )

    # 4. 计算权重
    merged['day_count'] = merged.groupby(['Stkcd', 'month'])['date'].transform('count')
    merged['pi'] = 1 / merged['day_count']
    merged['delta_k'] = delta ** merged['k']
    merged['p_delta'] = merged['pi'] * merged['delta_k']

    merged['w'] = merged.groupby(['Stkcd', 'month'])['p_delta'].transform('sum')
    merged['w'] = merged['delta_k'] / merged['w']

    # 5. 计算ST值
    merged['pi_w_Ri'] = merged['pi'] * merged['w'] * merged['Ri']
    merged['pi_Ri'] = merged['pi'] * merged['Ri']

    estv = merged.groupby(['Stkcd', 'month'])['pi_w_Ri'].sum().reset_index(name='Estv')
    r_avg = merged.groupby(['Stkcd', 'month'])['pi_Ri'].sum().reset_index(name='r_avg')

    st_df = pd.merge(estv, r_avg, on=['Stkcd', 'month'])
    st_df['ST'] = st_df['Estv'] - st_df['r_avg']

    return st_df[['Stkcd', 'month', 'ST']]


def calculate_st_from_files(
    stock_ret_path: str,
    market_ret_path: str,
    theta: float = 0.1,
    delta: float = 0.7,
) -> pd.DataFrame:
    """
    从CSV文件直接读取数据并计算ST因子。

    Parameters
    ----------
    stock_ret_path : str
        个股日收益率CSV文件路径
    market_ret_path : str
        市场日收益率CSV文件路径
    theta, delta : float
        同 calculate_st_simple

    Returns
    -------
    pd.DataFrame
        [Stkcd, month, ST]
    """
    stock_ret_df = pd.read_csv(
        stock_ret_path, dtype={'Stkcd': str}, parse_dates=['date']
    )
    market_ret_df = pd.read_csv(
        market_ret_path, parse_dates=['date']
    )
    return calculate_st_simple(stock_ret_df, market_ret_df, theta=theta, delta=delta)
