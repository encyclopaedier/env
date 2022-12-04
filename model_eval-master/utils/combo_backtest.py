from PqiDataSdk import *
from functools import partial
from itertools import product
from matplotlib import pyplot as plt
# from loguru import logger
from tabulate import tabulate
from matplotlib.gridspec import GridSpec
from numba import jit, njit, objmode
import seaborn as sns
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import numpy as np
import os
import pickle
import getpass
import warnings


warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
user = getpass.getuser()


class ComboBT():

    def __init__(self, cfg):
        self.ds = PqiDataSdk(user=user, size=1, pool_type="mt", log=False, offline=True)
        self.start_date = cfg.total_start_date
        self.end_date = cfg.total_end_date

    def data_prepare(self):
        # 获取票池
        stock_pool = self.ds.get_ticker_list(date='all')
        for ticker in ['000043', '000022', '601313']:
            stock_pool.remove(ticker)

        eod_data_dict = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.end_date)
        twap_data_dict = self.ds.get_eod_history(tickers=stock_pool, start_date=self.start_date, end_date=self.end_date,
                                                 source="ext_stock", fields=['TwapBegin30', 'TwapBegin60', 'TwapBegin120', 'Twap'])

        eod_data_dict['TwapOpen30'] = twap_data_dict['TwapBegin30'].copy()
        eod_data_dict['TwapOpen60'] = twap_data_dict['TwapBegin60'].copy()
        eod_data_dict['TwapOpen120'] = twap_data_dict['TwapBegin120'].copy()
        eod_data_dict['TwapOpen240'] = twap_data_dict['Twap'].copy()

        self.universe = self.ds.get_file(name='universe',
                                         tickers=stock_pool,
                                         start_date=self.start_date,
                                         end_date=self.end_date,
                                         format='ticker_date_real')

        self.up_feasible_stock = self.ds.get_file(name='up_feasible',
                                                  tickers=stock_pool,
                                                  start_date=self.start_date,
                                                  end_date=self.end_date,
                                                  format='ticker_date_real')

        self.down_feasible_stock = self.ds.get_file(name='down_feasible',
                                                    tickers=stock_pool,
                                                    start_date=self.start_date,
                                                    end_date=self.end_date,
                                                    format='ticker_date_real')

        self.price_dict = {}
        self.price_dict['Open'] = eod_data_dict['OpenPrice'] * eod_data_dict['AdjFactor']
        self.price_dict['Close'] = eod_data_dict['ClosePrice'] * eod_data_dict['AdjFactor']
        self.price_dict['TwapOpen30'] = eod_data_dict['TwapOpen30'] * eod_data_dict['AdjFactor']
        self.price_dict['TwapOpen60'] = eod_data_dict['TwapOpen60'] * eod_data_dict['AdjFactor']
        self.price_dict['TwapOpen120'] = eod_data_dict['TwapOpen120'] * eod_data_dict['AdjFactor']
        self.price_dict['TwapOpen240'] = eod_data_dict['TwapOpen240'] * eod_data_dict['AdjFactor']
        self.price_dict['Vwap'] = eod_data_dict['VWAP'] * eod_data_dict['AdjFactor']

        self.index_data = self.ds.get_eod_history(tickers=['000905', '000016', '000300', '000852', '000985'],
                                                  start_date=self.start_date, end_date=self.end_date, source='index')
        index_twap_dict = self.ds.get_eod_history(tickers=['000905', '000016', '000300', '000852', '000985'],
                                                  start_date=self.start_date, end_date=self.end_date, source='ext_index',
                                                  fields=['TwapBegin30', 'TwapBegin60', 'TwapBegin120', 'Twap'])

        self.index_data['TwapOpen30'] = index_twap_dict['TwapBegin30'].copy()
        self.index_data['TwapOpen60'] = index_twap_dict['TwapBegin60'].copy()
        self.index_data['TwapOpen120'] = index_twap_dict['TwapBegin120'].copy()
        self.index_data['TwapOpen240'] = index_twap_dict['Twap'].copy()

    @staticmethod
    def cal_maxdd_arr(array):
        drawdowns = []
        max_so_far = array[0]
        for i in range(len(array)):
            if array[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = array[i]
            else:
                drawdown = max_so_far - array[i]
                drawdowns.append(drawdown)
        return np.array(drawdowns)

    @staticmethod
    def cal_maxdd(array):
        drawdowns = []
        max_so_far = array[0]
        for i in range(len(array)):
            if array[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = array[i]
            else:
                drawdown = max_so_far - array[i]
                drawdowns.append(drawdown)
        return max(drawdowns)

    def annual_stat(self, test_ret, start_year, end_year):
        nc = test_ret[0]
        ac = test_ret[1]

        year_list = [str(i) for i in range(start_year, end_year + 1)]
        df = pd.DataFrame(index=year_list,
                          columns=['AlphaRet', 'AlphaRetNC', 'AlphaSharpe', 'AlphaSharpeNC', 'AlphaDD', 'TurnOver'],
                          dtype='float')

        annual_ret_nc_list = []
        annual_ret_ac_list = []
        for year in year_list:
            no_cost_ret = nc.loc[f'{year}0101':f'{year}1231']
            after_cost_ret = ac.loc[f'{year}0101':f'{year}1231']
            no_cost_sharpe = no_cost_ret.mean() / no_cost_ret.std() * np.sqrt(252)
            after_cost_sharpe = after_cost_ret.mean() / after_cost_ret.std() * np.sqrt(252)
            after_cost_dd = self.cal_maxdd(after_cost_ret.cumsum().dropna().values)
            tov = (no_cost_ret.sum() - after_cost_ret.sum()) / (0.0015 * len(no_cost_ret))
            df.loc[year] = [after_cost_ret.sum(), no_cost_ret.sum(), after_cost_sharpe, no_cost_sharpe,after_cost_dd, tov]
            annual_ret_nc_list.append(no_cost_ret.mean() * 245)
            annual_ret_ac_list.append(after_cost_ret.mean() * 245)

        df.loc['sum'] = df.mean()
        df.loc['sum', 'AlphaRet'] = np.nanmean(annual_ret_ac_list)
        df.loc['sum', 'AlphaRetNC'] = np.nanmean(annual_ret_nc_list)
        df.loc['sum', 'AlphaDD'] = self.cal_maxdd(test_ret[1].cumsum().values)
        return df

    def signal_backtest(self, signal_df, name='test', cost=0.0015, benchmark='index', index='000905',
                        return_type='Open_to_Open', start_date='20160101', end_date='20201231', plot=True):
        # 生成数据
        date_list_in_use = self.ds.get_trade_dates(start_date=start_date, end_date=end_date)

        # 生成因子矩阵
        # signal_df = signal_df[date_list_in_use] * local_up_feasible_stock * local_universe
        signal_df = (signal_df * self.universe)[date_list_in_use]
        long_signal_df = signal_df / signal_df.sum()
        long_signal_df = long_signal_df.fillna(0)

        # 生成回测收益率矩阵
        buy_type = return_type.split('_')[0]
        sell_type = return_type.split('_')[2]
        if 'Open' in buy_type:
            buy_df = self.price_dict[buy_type].shift(-1, axis=1) * self.up_feasible_stock
        else:
            buy_df = self.price_dict[buy_type] * self.up_feasible_stock.shift(1, axis=1)

        limit_price_df = self.price_dict[sell_type] * self.down_feasible_stock
        sell_price_df = (self.price_dict[sell_type] - self.price_dict[sell_type]) + limit_price_df.bfill(axis=1)
        if 'Open' in sell_type:
            sell_feasible_df = sell_price_df.shift(-2, axis=1)
            sell_df = self.price_dict[sell_type].shift(-2, axis=1)
        else:
            sell_feasible_df = sell_price_df.shift(-1, axis=1)
            sell_df = self.price_dict[sell_type].shift(-1, axis=1)

        bfill_ret_df = sell_feasible_df / buy_df - 1
        bfill_ret_df = bfill_ret_df[date_list_in_use]
        ret_df = sell_df / buy_df - 1

        # 生成多空头pnl
        long_ret_no_cost = (long_signal_df * bfill_ret_df).sum(axis=0) / long_signal_df.sum(axis=0)
        long_cost_df = np.abs(cost * (long_signal_df.shift(1, axis=1) - long_signal_df)) / 2
        long_ret_after_cost = long_ret_no_cost - long_cost_df.sum(axis=0) / long_signal_df.sum(axis=0)
        long_ret_no_cost = long_ret_no_cost.fillna(0)
        long_ret_after_cost = long_ret_after_cost.fillna(0)

        # 生成换手序列
        weight_df = (long_signal_df / long_signal_df.sum(axis=0)).fillna(0).replace(np.inf, 0)
        turnover = np.abs(weight_df - weight_df.shift(1, axis=1)).sum(axis=0)
        turnover_series = turnover.fillna(0).replace(np.infty, 0)

        # 生成指数
        if 'mean' in benchmark:
            index_ret = (ret_df * self.universe)[date_list_in_use].mean(axis=0)
        elif 'index' in benchmark:
            if 'Close' in buy_type:
                index_ret = self.index_data[buy_type + 'Price'].shift(-1, axis=1) / self.index_data[buy_type + 'Price'] - 1
                index_ret = index_ret.loc[index, date_list_in_use]
            else:
                index_ret = self.index_data[buy_type].shift(-2, axis=1) / self.index_data[buy_type].shift(-1, axis=1) - 1
                index_ret = index_ret.loc[index, date_list_in_use]
        else:
            index_ret = (bfill_ret_df * self.universe[bfill_ret_df.columns]).mean(axis=0)
            index_ret = index_ret * 0
        pool_ret = (ret_df * self.universe)[date_list_in_use].mean(axis=0)

        # ============================ 画图区域 ============================ #
        if plot:
            fig = plt.figure(figsize=(20, 20), dpi=500)
            gs = GridSpec(30, 2)

            ax1 = fig.add_subplot(gs[:9, 0])
            ax1.plot(list(long_ret_no_cost.index), list(long_ret_no_cost.cumsum().values), color='darkorange',
                     linewidth=0.8)
            ax1.plot(list(index_ret.index), list(index_ret.cumsum().values), color='indianred', linewidth=0.8)
            ax1.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax1.legend(['long', 'short', 'index'])
            ax1.grid(axis='y')
            ax1.set_title('Long Short Absolute No Cost Return')

            ax2 = fig.add_subplot(gs[:6, 1])
            ax2.plot(list(long_ret_no_cost.index), list((long_ret_no_cost - index_ret).cumsum().values), linewidth=0.8)
            ax2.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax2.legend(['Alpha'])
            ax2.grid(axis='y')
            ax2.set_title('Long Alpha No Cost Return')

            ax7 = fig.add_subplot(gs[7:9, 1])
            ax7.bar(list(long_ret_no_cost.index), list(self.cal_maxdd_arr((long_ret_no_cost - index_ret).cumsum().values)),
                    color='green')
            ax7.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax7.grid(axis='y')
            ax7.set_title('Long Alpha No Cost Max DrawDown')

            ax3 = fig.add_subplot(gs[10:19, 0])
            ax3.plot(list(long_ret_after_cost.index), list(long_ret_after_cost.cumsum().values), color='darkorange',
                     linewidth=0.8)
            ax3.plot(list(index_ret.index), list(index_ret.cumsum().values), color='indianred', linewidth=0.8)
            ax3.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax3.legend(['long', 'short', 'index'])
            ax3.grid(axis='y')
            ax3.set_title('Long Short Absolute After Cost Return')

            ax4 = fig.add_subplot(gs[10:16, 1])
            ax4.plot(list(long_ret_after_cost.index), list((long_ret_after_cost - index_ret).cumsum().values),
                     linewidth=0.8)
            ax4.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax4.legend(['Alpha'])
            ax4.grid(axis='y')
            ax4.set_title('Long Short Excess After Cost Return')

            ax8 = fig.add_subplot(gs[17:19, 1])
            ax8.bar(list(long_ret_after_cost.index),
                    list(self.cal_maxdd_arr((long_ret_after_cost - index_ret).cumsum().values)), color='green')
            ax8.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax8.grid(axis='y')
            ax8.set_title('Long Alpha After Cost Max DrawDown')

            ax5 = fig.add_subplot(gs[20:25, :])
            ax5.plot(list(turnover_series.index)[:-1], list((turnover_series).values / 2)[:-1], linewidth=0.8)
            ax5.plot(list(turnover_series.index)[:-1], [turnover_series.mean() / 2] * (len(turnover_series) - 1),
                     linewidth=0.7, linestyle='--', color='grey')
            ax5.set_xticks(list(turnover_series.index)[::int(len(list(turnover_series.values)) / 6)])
            ax5.legend(labels=[f'turnover: {round(turnover_series.mean() / 2, 3)}'])
            ax5.grid(b=True, axis='y')
            ax5.set_title('Turnover ts change')

            ax6 = fig.add_subplot(gs[26:30, :])
            ax6.plot(list(long_signal_df.columns)[:-1], list(long_signal_df.count().values)[:-1], linewidth=0.8)
            ax6.plot(list(long_signal_df.columns)[:-1], [long_signal_df.count().mean()] * (long_signal_df.shape[1] - 1),
                     linewidth=0.7, linestyle='--', color='grey')
            ax6.set_xticks(list(long_signal_df.columns)[::int(len(list(long_signal_df.columns)) / 6)])
            ax6.legend(labels=[f'Holding Count: {round(long_signal_df.count().mean(), 2)}'])
            ax6.grid(b=True, axis='y')
            ax6.set_title('Stock Holding Count')

            fig.savefig('./backtest_res/signal/{}.png'.format(name), dpi=200)

        # 收益率数据统计
        annual_coef = 252 / len(long_ret_after_cost)

        data_dict = {}
        data_dict['TurnOver'] = turnover_series.mean()
        data_dict['AlphaRet'] = (long_ret_after_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaRetNC'] = (long_ret_no_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaSharpe'] = (long_ret_after_cost - index_ret).mean() / (
                    long_ret_after_cost - index_ret).std() * np.sqrt(252)
        data_dict['AlphaSharpeNC'] = (long_ret_no_cost - index_ret).mean() / (
                    long_ret_no_cost - index_ret).std() * np.sqrt(252)
        data_dict['AlphaDrawdown'] = self.cal_maxdd((long_ret_after_cost - index_ret).cumsum().dropna().values)
        data_dict['AlphaDrawdownNC'] = self.cal_maxdd((long_ret_no_cost - index_ret).cumsum().dropna().values)
        data_dict['DrawdownRatio'] = data_dict['AlphaDrawdownNC'] / data_dict['AlphaRetNC']

        drawdown_pool_df = pd.DataFrame(index=long_ret_after_cost.index, columns=['drawdown_pool'], dtype='float')
        drawdown_index_df = pd.DataFrame(index=long_ret_after_cost.index, columns=['drawdown_index'], dtype='float')
        drawdown_pool_df['drawdown_pool'] = self.cal_maxdd_arr((long_ret_after_cost - pool_ret).cumsum().values)
        drawdown_index_df['drawdown_index'] = self.cal_maxdd_arr((long_ret_after_cost - index_ret).cumsum().values)

        return data_dict, (long_ret_no_cost, long_ret_after_cost, index_ret, pool_ret), long_signal_df, (drawdown_index_df, drawdown_pool_df)

    def backtest(self, factor_df, name='test', head=400, method='factor', cost=0.0015, group_num=10,
                 benchmark='index', index='000905', return_type='Open_to_Open', start_date='20160101', end_date='20201231', plot=True):
        '''
        Input:
        factor_df, dataframe, 被测试的因子
        head, integer, 测试头组数目
        method, str, 'factor / equal', factor指因子值加权, equal指头组指定数目等权
        cost, float, 手续费
        group_num, integer, 分组个数
        index, str, 对标指数benchmark,
        return_type, str, 'Open_to_Open/TwapOpen_to_TwapOpen/TwapClose_to_TwapClose/Vwap_to_Vwap/Close_to_Close', 回测收益率
        start_date/end_date, str, 起始日期
        plot, bool, 是否画回测收益图
        risk_plot, bool, 是否画风格分析图

        Output:
        data_dict, pd.DataFrame, 回测统计指标
        factor_ret_no_cost, pd.Series, 多空收益pnl
        '''
        # 生成数据
        date_list_in_use = self.ds.get_trade_dates(start_date=start_date, end_date=end_date)

        # 生成因子矩阵
        # backtest_df = factor_df * local_universe * local_up_feasible_stock
        backtest_df = factor_df * self.universe
        backtest_df = backtest_df[date_list_in_use]
        demean_backtest_df = backtest_df - backtest_df.mean()
        std_backtest_df = demean_backtest_df / (demean_backtest_df.abs().sum() / 2)

        # 生成回测收益率矩阵
        buy_type = return_type.split('_')[0]
        sell_type = return_type.split('_')[2]
        if 'Open' in buy_type:
            buy_df = self.price_dict[buy_type].shift(-1, axis=1) * self.up_feasible_stock
        else:
            buy_df = self.price_dict[buy_type] * self.up_feasible_stock.shift(1, axis=1)

        limit_price_df = self.price_dict[sell_type] * self.down_feasible_stock
        sell_price_df = (self.price_dict[sell_type] - self.price_dict[sell_type]) + limit_price_df.bfill(axis=1)
        if 'Open' in sell_type:
            sell_feasible_df = sell_price_df.shift(-2, axis=1)
            sell_df = self.price_dict[sell_type].shift(-2, axis=1)
        else:
            sell_feasible_df = sell_price_df.shift(-1, axis=1)
            sell_df = self.price_dict[sell_type].shift(-1, axis=1)

        bfill_ret_df = sell_feasible_df / buy_df - 1
        bfill_ret_df = bfill_ret_df[date_list_in_use]
        ret_df = sell_df / buy_df - 1

        # 算ic和rankic
        ic_list = factor_df[date_list_in_use].corrwith(ret_df[date_list_in_use])
        ic = ic_list.mean()
        rankic_list = factor_df[date_list_in_use].corrwith(ret_df[date_list_in_use], method='spearman')
        rankic = rankic_list.mean()

        # 算ic decay
        ic_decay_dict = {}
        rankic_decay_dict = {}
        decay_series = [0, 1, 3, 5, 10, 20, 30, 60, 120]
        for i in decay_series:
            ic_decay_dict[i] = (
                factor_df[date_list_in_use].corrwith(ret_df.shift(-i, axis=1)[date_list_in_use]).mean())
            rankic_decay_dict[i] = (factor_df[date_list_in_use].corrwith(ret_df.shift(-i, axis=1)[date_list_in_use],
                                                                         method='spearman').mean())

        # 生成排序
        uprank_df = backtest_df.rank(axis=0, ascending=False)
        downrank_df = backtest_df.rank(axis=0, ascending=True)

        # 生成多头信号
        if 'factor' in method:
            long_signal_df = backtest_df.copy()
            long_signal_df.iloc[:, :] = np.where(std_backtest_df >= 0, std_backtest_df, 0)
            long_cost_df = np.abs(cost * (long_signal_df.shift(1, axis=1) - long_signal_df)) / 2
            short_signal_df = backtest_df.copy()
            short_signal_df.iloc[:, :] = np.where(std_backtest_df <= 0, -1 * std_backtest_df, 0)
            short_cost_df = np.abs(cost * (short_signal_df.shift(1, axis=1) - short_signal_df)) / 2
        else:
            long_signal_df = backtest_df.copy()
            long_signal_df.iloc[:, :] = np.where(uprank_df <= head, 1, 0)
            long_cost_df = np.abs(cost * (long_signal_df.shift(1, axis=1) - long_signal_df)) / 2
            short_signal_df = backtest_df.copy()
            short_signal_df.iloc[:, :] = np.where(downrank_df <= head, 1, 0)
            short_cost_df = np.abs(cost * (short_signal_df.shift(1, axis=1) - short_signal_df)) / 2

        # 生成分组信号
        group_signal_df_list = list(np.arange(group_num))
        stock_num_base = backtest_df.rank(axis=0).max() / group_num
        for i in range(group_num):
            group_signal_df_list[i] = (uprank_df <= (i + 1) * stock_num_base) * (uprank_df > i * stock_num_base)

        # 生成多空头pnl
        long_ret_no_cost = (long_signal_df * bfill_ret_df).sum(axis=0) / long_signal_df.sum(axis=0)
        short_ret_no_cost = (short_signal_df * bfill_ret_df).sum(axis=0) / short_signal_df.sum(axis=0)
        long_ret_after_cost = long_ret_no_cost - long_cost_df.sum(axis=0) / long_signal_df.sum(axis=0)
        short_ret_after_cost = short_ret_no_cost - short_cost_df.sum(axis=0) / short_signal_df.sum(axis=0)
        long_ret_no_cost = long_ret_no_cost.fillna(0)
        long_ret_after_cost = long_ret_after_cost.fillna(0)
        short_ret_no_cost = short_ret_no_cost.fillna(0)
        short_ret_after_cost = short_ret_after_cost.fillna(0)

        # 判断Long/Short的方向
        if long_ret_no_cost.sum() < short_ret_no_cost.sum():
            long_ret_no_cost, short_ret_no_cost = short_ret_no_cost, long_ret_no_cost
            long_ret_after_cost, short_ret_after_cost = short_ret_after_cost, long_ret_after_cost
            long_signal_df, short_signal_df = short_signal_df, long_signal_df

        # 生成换手序列
        weight_df = (long_signal_df / long_signal_df.sum(axis=0)).fillna(0).replace(np.inf, 0)
        turnover = np.abs(weight_df - weight_df.shift(1, axis=1)).sum(axis=0)
        turnover_series = turnover.fillna(0).replace(np.infty, 0)

        # 生成指数
        if 'mean' in benchmark:
            index_ret = (ret_df * self.universe)[date_list_in_use].mean(axis=0)
        elif 'index' in benchmark:
            if 'Close' in buy_type:
                index_ret = self.index_data[buy_type + 'Price'].shift(-1, axis=1) / self.index_data[buy_type + 'Price'] - 1
                index_ret = index_ret.loc[index, date_list_in_use]
            else:
                index_ret = self.index_data[buy_type].shift(-2, axis=1) / self.index_data[buy_type].shift(-1, axis=1) - 1
                index_ret = index_ret.loc[index, date_list_in_use]
        else:
            print('Wrong Type for benchmark. Use MEAN instead.')
            index_ret = (ret_df * self.universe[ret_df.columns]).mean(axis=0)
        pool_ret = (ret_df * self.universe)[date_list_in_use].mean(axis=0)

        # 生成分组pnl
        group_ret_series_list_no_cost = list(np.arange(group_num))
        group_ret_series_list_after_cost = list(np.arange(group_num))
        group_ret_list_no_cost = list(np.arange(group_num))
        group_ret_list_after_cost = list(np.arange(group_num))
        group_cost_df_list = list(np.arange(group_num))
        group_tov_list = list(np.arange(group_num))
        for i in range(group_num):
            group_ret_series_list_no_cost[i] = (group_signal_df_list[i] * bfill_ret_df).sum(axis=0) / group_signal_df_list[i].sum(axis=0)
            group_ret_series_list_no_cost[i] = group_ret_series_list_no_cost[i].fillna(0)
            group_cost_df_list[i] = np.abs(cost * (group_signal_df_list[i].shift(1, axis=1) - group_signal_df_list[i])) / 2
            group_ret_series_list_after_cost[i] = group_ret_series_list_no_cost[i] - group_cost_df_list[i].sum(axis=0) / group_signal_df_list[i].sum(axis=0)
            group_ret_series_list_after_cost[i] = group_ret_series_list_after_cost[i].fillna(0)

            group_ret_list_no_cost[i] = group_ret_series_list_no_cost[i].cumsum().values[-1]
            group_ret_list_after_cost[i] = group_ret_series_list_after_cost[i].cumsum().values[-1]

            weight_df = group_signal_df_list[i] / (group_signal_df_list[i].sum(axis=0)).fillna(0).replace(np.inf, 0)
            turnover = np.abs(weight_df - weight_df.shift(1, axis=1)).sum(axis=0)
            group_tov_list[i] = turnover.fillna(0).replace(np.infty, 0)

            # ============================ 画图区域 ============================ #
        if plot:
            fig = plt.figure(figsize=(20, 20), dpi=500)
            gs = GridSpec(32, 2)

            ax1 = fig.add_subplot(gs[:9, 0])
            ax1.plot(list(long_ret_no_cost.index), list(long_ret_no_cost.cumsum().values), color='darkorange')
            ax1.plot(list(short_ret_no_cost.index), list(short_ret_no_cost.cumsum().values), color='limegreen')
            ax1.plot(list(index_ret.index), list(index_ret.cumsum().values), color='indianred')
            ax1.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax1.legend(['long', 'short', 'index'])
            ax1.grid(axis='y')
            ax1.set_title('Long Short Absolute No Cost Return')

            ax2 = fig.add_subplot(gs[:6, 1])
            ax2.plot(list(long_ret_no_cost.index), list((long_ret_no_cost - index_ret).cumsum().values))
            ax2.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax2.legend(['Alpha'])
            ax2.grid(axis='y')
            ax2.set_title('Long Alpha No Cost Return')

            ax7 = fig.add_subplot(gs[7:9, 1])
            ax7.bar(list(long_ret_no_cost.index),
                    list(self.cal_maxdd_arr((long_ret_no_cost - index_ret).cumsum().values)), color='green')
            ax7.set_xticks(list(long_ret_no_cost.index)[::int(len(list(long_ret_no_cost.index)) / 6)])
            ax7.grid(axis='y')
            ax7.set_title('Long Alpha No Cost Max DrawDown')

            ax3 = fig.add_subplot(gs[10:19, 0])
            ax3.plot(list(long_ret_after_cost.index), list(long_ret_after_cost.cumsum().values), color='darkorange')
            ax3.plot(list(short_ret_after_cost.index), list(short_ret_after_cost.cumsum().values),
                     color='limegreen')
            ax3.plot(list(index_ret.index), list(index_ret.cumsum().values), color='indianred')
            ax3.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax3.legend(['long', 'short', 'index'])
            ax3.grid(axis='y')
            ax3.set_title('Long Short Absolute After Cost Return')

            ax4 = fig.add_subplot(gs[10:16, 1])
            ax4.plot(list(long_ret_after_cost.index), list((long_ret_after_cost - index_ret).cumsum().values))
            ax4.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax4.legend(['Alpha'])
            ax4.grid(axis='y')
            ax4.set_title('Long Short Excess After Cost Return')

            ax8 = fig.add_subplot(gs[17:19, 1])
            ax8.bar(list(long_ret_after_cost.index),
                    list(self.cal_maxdd_arr((long_ret_after_cost - index_ret).cumsum().values)), color='green')
            ax8.set_xticks(list(long_ret_after_cost.index)[::int(len(list(long_ret_after_cost.index)) / 6)])
            ax8.grid(axis='y')
            ax8.set_title('Long Alpha After Cost Max DrawDown')

            ax5 = fig.add_subplot(gs[20:25, 0])
            ax5.plot(list(turnover_series.index)[:-1], list((turnover_series).values / 2)[:-1], linewidth=0.7)
            ax5.plot(list(turnover_series.index)[:-1], [turnover_series.mean() / 2] * (len(turnover_series) - 1),
                     linewidth=0.7, linestyle='--', color='grey')
            ax5.set_xticks(list(turnover_series.index)[::int(len(list(turnover_series.values)) / 6)])
            ax5.legend(labels=[f'turnover: {round(turnover_series.mean() / 2, 3)}'])
            ax5.grid(b=True, axis='y')
            ax5.set_title('Turnover ts change')

            ax6 = fig.add_subplot(gs[20:25, 1])
            total_ret = [np.nansum(ret) for ret in group_ret_series_list_no_cost]
            ax6.bar(range(len(total_ret)), total_ret)
            ax6.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret) - 1, color='r')
            ax6.set_xticks(range(group_num))
            ax6.grid(b=True, axis='y')
            ax6.set_title('Group Return No Cost Bar')

            ax9 = fig.add_subplot(gs[26:, 0])
            width = 0.4
            ax9.bar(np.arange(len(ic_decay_dict)) - width / 2, list(ic_decay_dict.values()), width)
            ax9.bar(np.arange(len(rankic_decay_dict)) + width / 2, list(rankic_decay_dict.values()), width)
            ax9.set_xticks(np.arange(len(ic_decay_dict)))
            ax9.set_xticklabels([str(i) for i in decay_series])
            ax9.legend(labels=['IC', 'rankIC'])
            ax9.grid(b=True, axis='y')
            ax9.set_title('IC and rankIC decay')

            ax10 = fig.add_subplot(gs[26:, 1])
            ax10.plot(list(ic_list.index), list(ic_list.cumsum().values))
            ax10.set_xticks(list(ic_list.index)[::int(len(list(ic_list.index)) / 6)])
            ax10.grid(b=True, axis='y')
            ax10.set_title('Cumulated IC_IR: {}'.format(round(ic_list.mean() / ic_list.std(), 3)))

            fig.savefig('./backtest_res/combo/{}.png'.format(name), dpi=200)

        # 收益率数据统计
        annual_coef = 252 / len(long_ret_after_cost)

        data_dict = {}
        data_dict['IC'] = ic
        data_dict['rankIC'] = rankic
        data_dict['GroupIC'] = np.corrcoef(group_num - np.arange(group_num), np.array(group_ret_list_after_cost))[0, 1]
        data_dict['GroupICNC'] = np.corrcoef(group_num - np.arange(group_num), np.array(group_ret_list_no_cost))[0, 1]
        data_dict['IR'] = ic_list.mean() / ic_list.std()
        data_dict['TurnOver'] = turnover_series.mean()
        data_dict['AlphaRet'] = (long_ret_after_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaRetNC'] = (long_ret_no_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaSharpe'] = (long_ret_after_cost - index_ret).mean() / (long_ret_after_cost - index_ret).std() * np.sqrt(252)
        data_dict['AlphaSharpeNC'] = (long_ret_no_cost - index_ret).mean() / (long_ret_no_cost - index_ret).std() * np.sqrt(252)
        data_dict['AlphaDrawdown'] = self.cal_maxdd((long_ret_after_cost - index_ret).cumsum().dropna().values)
        data_dict['AlphaDrawdownNC'] = self.cal_maxdd((long_ret_no_cost - index_ret).cumsum().dropna().values)
        data_dict['DrawdownRatio'] = data_dict['AlphaDrawdownNC'] / data_dict['AlphaRetNC']
        data_dict['Score'] = data_dict['AlphaRetNC'] ** 2 * data_dict['AlphaSharpeNC'] / (data_dict['AlphaDrawdownNC'] * data_dict['TurnOver'])

        drawdown_pool_df = pd.DataFrame(index=long_ret_after_cost.index, columns=['drawdown_pool'], dtype='float')
        drawdown_index_df = pd.DataFrame(index=long_ret_after_cost.index, columns=['drawdown_index'], dtype='float')
        drawdown_pool_df['drawdown_pool'] = self.cal_maxdd_arr((long_ret_after_cost - pool_ret).cumsum().values)
        drawdown_index_df['drawdown_index'] = self.cal_maxdd_arr((long_ret_after_cost - index_ret).cumsum().values)

        return data_dict, (long_ret_no_cost, long_ret_after_cost, index_ret, pool_ret), long_signal_df, (drawdown_index_df, drawdown_pool_df)


def get_comboBT_result(cfg):
    # 读取因子
    ds = PqiDataSdk(user=user, size=1, pool_type="mt", log=False, offline=True)
    date_list = ds.get_trade_dates(start_date=cfg.start_date, end_date=cfg.end_date)
    tickers = ds.get_ticker_list(date='all')
    for ticker in ['000043', '000022', '601313']:
        tickers.remove(ticker)

    ComboBT = ComboBT()
    ComboBT.data_prepare()

    print('开始回测组合.')
    combo_df = ds.get_eod_feature(fields=[cfg.combo_name],
                                    where=cfg.combo_path,
                                    tickers=tickers,
                                    dates=ds.get_trade_dates(start_date=cfg.total_start_date, end_date=cfg.total_end_date))[cfg.combo_name].to_dataframe()
    # combo_df = pd.read_pickle('~/lowfre_project_yzhou0308/results/result0315_1.pkl')
    data_dict, combo_long_result, combo_long_signal_df, drawdown_tuple = ComboBT.backtest(factor_df = combo_df,
                                                                                            name = cfg.combo_name,
                                                                                            head = cfg.head,
                                                                                            method = cfg.method,
                                                                                            cost = cfg.cost,
                                                                                            group_num = cfg.group_num,
                                                                                            benchmark = cfg.benchmark,
                                                                                            index = cfg.index,
                                                                                            return_type = cfg.return_type,
                                                                                            start_date = cfg.start_date,
                                                                                            end_date = cfg.end_date,
                                                                                            plot = True)

    excess_500_tuple = (combo_long_result[0] - combo_long_result[2], combo_long_result[1] - combo_long_result[2])
    annual_result_df = ComboBT.annual_stat(excess_500_tuple, int(cfg.start_date[:4]), int(cfg.end_date[:4]))    
    return annual_result_df

class ComboConfig():

    '''
    输入 & 回测config
    '''
    # 测试类型
    # 包括两类，'signal'为权重信号；而'combo'为组合因子类型
    test_type = 'combo'

    # --combo名称和路径（有需要填写，可DIY）
    # combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_v2'
    # combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_v3'
    # combo_path = '/data/shared/low_fre_alpha/paper_trading_combo'
    combo_path = '/home/yzhou/eod_xgb/results/20220722/os'
    # combo_name = 'eod_xgb_20211228100736_reg'
    # combo_name = 'eod_xgb_20211227133912_reg'
    # combo_name = 'eod_test_fac'
    # combo_name = 'eod_xgb_202202111723'
    # combo_name = 'eod_xgb_202202141428'
    # combo_name = 'eod_factor_all_equal_zz1800'
    # combo_name = 'eod_factor_best_equal_zz1800'
    # combo_name = 'eod_factor_all_wgt_zz1800'
    # combo_name = 'eod_factor_best_wgt_zz1800'
    # combo_name = 'eod_xgb_20220221195353_cls_eod_zz1800_TwapOpen_rtn1_200_5_linear'
    # combo_name = 'eod_xgb_lq8060_reg_1351020_all'
    # combo_name = 'eod_xgb_20220221172840_reg'
    combo_name = 'eod_zz1800_r10rank_os_1744'
    # combo_name = 'eod_yhzhou_alpha029'

    # --signal名称和路径（有需要填写，可DIY）
    # signal_name = 'eod_xjb_test_0_signal'
    signal_name = 'eod_opt_y1_r0.1_l0.01'
    # signal_name = 'eod_opt_all_roll_30_0.3_0.3'
    # signal_path = '/home/yhzhou/03_data/data/optimized_result'
    # signal_path = '/home/yhzhou/03_data/platform_test'
    signal_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_v2'

    # 总起讫日期，为重要数据的读取时间，包括信号/因子，以及ret_df等，以防有缺漏
    total_start_date = '20150101'
    total_end_date = '20210331'
    # 回测起讫日期
    # start_date = '20170101'
    # end_date = '20201231'
    start_date = '20180101'
    end_date = '20210331'


    # 回测因子及信号的参数，return_type有6中可选，TwapOpen30/60/120/240及Open和Close
    # 部分参数在回测信号时不需要被使用
    # method可选equal/factor,
    ## -- equal表示头组等权，需要设定下述head；
    ## -- factor表示因子值加权，不需要设定head；
    method = 'equal'
    head = 400
    cost = 0.0013
    group_num = 20
    benchmark = 'index'
    index = '000905'
    return_type = 'TwapOpen60_to_TwapOpen60'
    # return_type = 'TwapOpen240_to_TwapOpen240'


    '''
    风格分析config
    '''
    # benchmark指数
    bm_index = index
    # 排序方式
    rank_method = 'exposure' # 可以按照暴露和归因排序，exposure / attribute
    # 信号类型
    signal_type = 'val' # 有两个选项，传入的是手数选择 vol，传入的是市值选择val
    # 信号测试类型，低频长周期回测应该只使用‘multi'，此不需要修改
    signal_test_type = 'multi' # 有两个选项，单日交易测试选择single，长时段回测测试选择multi
    # 多日日期
    multi_start_date = start_date
    multi_end_date = end_date # 只在signal_test_type = 'multi'的情形下使用
    # 单日日期
    single_date = '20211130' # 只在signal_test_type = 'single'的情形下使用


    '''
    画图config
    '''
    # 结果储存路径
    save_fig_path = './res/'






















