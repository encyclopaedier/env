import pandas as pd
import numpy as np
import random
import pickle
import pyarrow.parquet as pq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.tools import *


def get_eval_result(param_dict, res_df_all, **kwargs):
    data_path = param_dict['data_path']
    y_name = param_dict['y_name']
    start_date = param_dict['start_date']
    end_date = param_dict['end_date']
    crss_ratio_thr = param_dict.get('crss_ratio_thr', 0.05)
    res_df_all = res_df_all.copy()
    res_df_all = res_df_all[(res_df_all.date>=start_date)&(res_df_all.date<=end_date)]
    if kwargs.get('trade_type', 'origin') == 'origin':
        res_df_all['crss_rank'] = res_df_all.groupby(['date', "TimeStamp"])['model_pred'].rank(axis=0, pct=True)
    elif kwargs.get('trade_type', 'origin') == 'index':
        res_df_all['_group'] = res_df_all['index_group']
        res_df_all['crss_rank'] = res_df_all.groupby(['date', "TimeStamp", 'group_'])['model_pred'].rank(axis=0, pct=True)
    elif kwargs.get('trade_type', 'origin') == 'group':
        res_df_all['_group'] = res_df_all['m1_ts_eod_rank_group']
        res_df_all['crss_rank'] = res_df_all.groupby(['date', "TimeStamp", 'group_'])['model_pred'].rank(axis=0, pct=True)

    crss_res = res_df_all[res_df_all['crss_rank']>=1-crss_ratio_thr].copy()
    result_dict = {}
    ret_list = crss_res.groupby('date')[y_name].mean()
    result_dict['rtn'] = ret_list.sum()
    result_dict['sharpe'] = ret_list.mean()*np.sqrt(252)/(ret_list.std()+0.00001)

    max_trade_time = param_dict.get('max_trade_time', 1)
    crss_res["stock_trade_times"] = crss_res.groupby(['date', "ticker"])["model_pred"].cumcount() + 1
    crss_res_mtt = crss_res[crss_res["stock_trade_times"] <= max_trade_time]
    ret_list_mtt = crss_res_mtt.groupby('date')[y_name].mean()
    result_dict['rtn_mtt'] = ret_list_mtt.sum()
    result_dict['sharpe_mtt'] = ret_list_mtt.mean()*np.sqrt(252)/(ret_list_mtt.std()+0.00001)

    holds_info = crss_res.groupby(['date','ticker'])['model_pred'].count()
    holds_info = holds_info.reset_index().pivot_table(values='model_pred', index='ticker', columns='date')
    holds_info[holds_info>0] = 1
    holds_info = holds_info.fillna(0)
    holds_info_diff = holds_info.diff(axis=1)
    holds_info_diff[holds_info_diff<0] = 0
    tr_ratio = (holds_info_diff.sum()/holds_info.sum()).mean()
    result_dict['turnover'] = tr_ratio
    result_dict['rtn_fee'] = result_dict['rtn_mtt'] - tr_ratio*0.0015*ret_list_mtt.shape[0]


    index_code = param_dict['index_code']
    data_fe = pd.read_pickle(f'{data_path}/support_data/factor_exposure.pkl')
    data_iw = pd.read_pickle(f'{data_path}/support_data/index_weight_{index_code}.pkl').loc[:, start_date:end_date]

    data_fe_size = data_fe['size'].loc[:, start_date:end_date]
    exp_size = ((holds_info/holds_info.sum()*data_fe_size).sum() - (data_fe_size*data_iw/data_iw.sum()).sum())
    result_dict['exp_size_mean'] = exp_size.mean()
    result_dict['exp_size_min'] = exp_size.min()
    result_dict['exp_size_max'] = exp_size.max()

    data_fe_bp = data_fe['book_to_price'].loc[:, start_date:end_date]
    exp_bp = ((holds_info/holds_info.sum()*data_fe_bp).sum() - (data_fe_bp*data_iw/data_iw.sum()).sum())
    result_dict['exp_bp_mean'] = exp_bp.mean()
    result_dict['exp_bp_min'] = exp_bp.min()
    result_dict['exp_bp_max'] = exp_bp.max()

    data_fe_rv = data_fe['residual_volatility'].loc[:, start_date:end_date]
    exp_rv = ((holds_info/holds_info.sum()*data_fe_rv).sum() - (data_fe_rv*data_iw/data_iw.sum()).sum())
    result_dict['exp_rv_mean'] = exp_rv.mean()
    result_dict['exp_rv_min'] = exp_rv.min()
    result_dict['exp_rv_max'] = exp_rv.max()

    holds_kcb = crss_res[crss_res.ticker>='688000'].groupby(['date','ticker'])['model_pred'].count()
    holds_kcb = holds_kcb.reset_index().pivot_table(values='model_pred', index='ticker', columns='date')
    holds_kcb[holds_kcb>0] = 1
    holds_kcb = holds_kcb.fillna(0)
    kcb_ratio = (holds_kcb.sum()/holds_info.sum())
    result_dict['kcb_ratio_mean'] = kcb_ratio.mean()
    result_dict['kcb_ratio_max'] = kcb_ratio.max()

    holds_cyb = crss_res[(crss_res.ticker>='300000') & (crss_res.ticker<='310000')].groupby(['date','ticker'])['model_pred'].count()
    holds_cyb = holds_cyb.reset_index().pivot_table(values='model_pred', index='ticker', columns='date')
    holds_cyb[holds_cyb>0] = 1
    holds_cyb = holds_cyb.fillna(0)
    cyb_ratio = (holds_cyb.sum()/holds_info.sum())
    result_dict['cyb_ratio_mean'] = cyb_ratio.mean()
    result_dict['cyb_ratio_max'] = cyb_ratio.max()

    return result_dict


def plot_sizepnl_groupby(res_df_all, group_dict={}):
    group_num = group_dict.get('group_num', 4)
    label_3d = group_dict.get('ytrue_3d', 'm1_ts_y_60twap_2n3open_Wgted_fullmarket_ex')
    label = group_dict['y_true']
    sep_label = 'm1_ts_y_mktval'

    res_df_all = res_df_all.copy()
    res_df_all['model_pred'] = res_df_all['model_pred'] + np.random.rand(len(res_df_all)) * 1e-8
    
    plot_index = False

    _data_test = res_df_all[['date', 'TimeStamp', sep_label]]
    if 'group_col' in group_dict:
        res_df_all['group_'] = group_dict['group_col'].values
    else:
        if 'group_index' in group_dict:
            group_info_test = get_index_group(res_df_all[['ticker', 'date', 'TimeStamp']], group_dict['group_index'])
            res_df_all['group_'] = group_info_test.values
        else:
            group_info_test = _data_test.groupby(['date', 'TimeStamp']).transform(lambda x: pd.qcut(x, group_num, labels=range(group_num)))
            res_df_all['group_'] = group_info_test[sep_label].values

    start_date = group_dict.get('start_date', res_df_all['date'].min())
    end_date = group_dict.get('end_date', res_df_all['date'].max())

    res_df_all = res_df_all[(res_df_all.date>=start_date)&(res_df_all.date<=end_date)&res_df_all.model_pred.notna()]

    chance_ma = group_dict.get('chance_ma', 20)
    crss_ratio_thr = group_dict.get('crss_ratio_thr', 0.02)

    res_df_all['crss_rank_ongroup'] = res_df_all.groupby(['date', "TimeStamp", 'group_'])['model_pred'].rank(axis=0, pct=True)

    if 'group_index' in group_dict:
        group_index = group_dict.get('group_index')
        pre_name_list = [i for i in group_index]
    else:
        pre_name_list = [f'SIZE_{i}' for i in range(group_num)]

    label = group_dict.get('y_true')
    dates = res_df_all.date.unique().tolist()
    return_dict = {}
    fig = plt.figure(figsize=(16, 16), dpi=100, facecolor="white")
    matplotlib.rcParams['axes.linewidth'] = 2
    cur_gs = gridspec.GridSpec(3, 2,hspace=0.1, top=0.9, bottom=0.1, left=0.1, right=0.9)
    

    crss_pnl_plt = fig.add_subplot(cur_gs[0, 0])
    crss_pnl_plt.set_title(f'原始和重新分配权重 截面收益 {crss_ratio_thr*100:.1f}%')
    crss_pnl_plt.grid(linestyle="--", linewidth=1.5)
    crss_pnl_plt.set_xticks(range(0, len(dates), 20))
    crss_pnl_plt.set_xticklabels([dates[idx] for idx in crss_pnl_plt.get_xticks()], rotation=20)

    crss_chance = fig.add_subplot(cur_gs[0, 1])
    crss_chance.set_title(f'原始截面机会占比 {chance_ma}天MA')
    crss_chance.grid(linestyle="--", linewidth=1.5)
    crss_chance.set_xticks(range(0, len(dates), 20))
    crss_chance.set_xticklabels([dates[idx] for idx in crss_chance.get_xticks()], rotation=20)

    crss_pnl_plt_sg = fig.add_subplot(cur_gs[1, 0])
    crss_pnl_plt_sg.set_title(f'各市值分组下取头部信号截面收益 {crss_ratio_thr*100:.1f}%')
    crss_pnl_plt_sg.grid(linestyle="--", linewidth=1.5)
    crss_pnl_plt_sg.set_xticks(range(0, len(dates), 20))
    crss_pnl_plt_sg.set_xticklabels([dates[idx] for idx in crss_pnl_plt_sg.get_xticks()], rotation=20)

    crss_pnl_plt_sgtp = fig.add_subplot(cur_gs[1, 1])
    crss_pnl_plt_sgtp.set_title(f'各市值分组下取头部信号截面收益 去风格 {crss_ratio_thr*100:.1f}%')
    crss_pnl_plt_sgtp.grid(linestyle="--", linewidth=1.5)
    crss_pnl_plt_sgtp.set_xticks(range(0, len(dates), 20))
    crss_pnl_plt_sgtp.set_xticklabels([dates[idx] for idx in crss_pnl_plt_sgtp.get_xticks()], rotation=20)

    if label_3d in res_df_all.columns:
        crss_pnl_plt_sgtp_3d = fig.add_subplot(cur_gs[2, 1])
        crss_pnl_plt_sgtp_3d.set_title(f'各市值分组下取头部信号截面收益 去风格 3天{crss_ratio_thr*100:.1f}%')
        crss_pnl_plt_sgtp_3d.grid(linestyle="--", linewidth=1.5)
        crss_pnl_plt_sgtp_3d.set_xticks(range(0, len(dates), 20))
        crss_pnl_plt_sgtp_3d.set_xticklabels([dates[idx] for idx in crss_pnl_plt_sgtp.get_xticks()], rotation=20)
        

    res_df_all['crss_rank'] = res_df_all.groupby(['date', "TimeStamp"])['model_pred'].rank(axis=0, pct=True)
    crss_res = res_df_all[res_df_all['crss_rank']>=1-crss_ratio_thr]
    
    crss_temp_pnl_data = crss_res.groupby(['date'])[label].mean().cumsum()
    _result = crss_temp_pnl_data.iloc[-1]
    crss_pnl_plt.plot(crss_temp_pnl_data, label=f'ORIGIN: {_result:.4f}')
    return_dict['rtn_origin'] = _result
    return_dict['origin_sharpe'] = _result/crss_temp_pnl_data.shape[0]*np.sqrt(252)/(crss_temp_pnl_data.diff().std()+0.00001)

    crss_res_group = res_df_all[res_df_all['crss_rank_ongroup']>=1-crss_ratio_thr]
    if 'max_trade_times' in group_dict:
        max_trade_times = group_dict.get('max_trade_times')
        crss_res_group["stock_trade_times"] = crss_res_group.groupby(['date', "ticker"])["model_pred"].cumcount() + 1
        if type(max_trade_times) == int:
            crss_res_group = crss_res_group[crss_res_group["stock_trade_times"] <= max_trade_times]
            max_trade_times = [max_trade_times for i in pre_name_list]
        elif type(max_trade_times) == list:
            _df_list = []
            for ii in range(len(pre_name_list)):
                _crss_res_group = crss_res_group[(crss_res_group["group_"] == ii) & (crss_res_group["stock_trade_times"] <= max_trade_times[ii])]
                _df_list.append(_crss_res_group)
            crss_res_group = pd.concat(_df_list)
    else:
        max_trade_times = [10 for i in pre_name_list]

    crss_res_group = check_eod_filter(crss_res_group)

    chance_count = crss_res.groupby(['date'])[label].count()
    for ii in range(len(pre_name_list)):
        prename = pre_name_list[ii]
        chance_count_num = (crss_res[crss_res.group_==ii].groupby(['date'])[label].count()/chance_count).rolling(10, min_periods=1).mean()
        chance_count_num = chance_count_num.reindex(dates)
        _result = chance_count_num.mean()
        crss_chance.plot(chance_count_num, label=f'{prename}: {_result:.4f}')
    crss_chance.legend(loc = 'upper left')

    size_ratio_dict = group_dict.get('size_ratio', {})
    size_ratio_rtn = {}
    for srg in size_ratio_dict:
        if len(size_ratio_dict[srg]) != group_num:
            continue
        size_ratio_rtn[f'size_ratio_{srg}'] = pd.DataFrame()

    for ii in range(len(pre_name_list)):
        prename = pre_name_list[ii]
        # _idx = group_info_test==ii
        res_df = res_df_all[res_df_all.group_==ii]
        size_pool_pnl = res_df[['date', label]].groupby(['date']).mean().cumsum()

        _crss_res = crss_res_group[crss_res_group.group_==ii]

        crss_temp_pnl_data = _crss_res.groupby(['date'])[label].mean().cumsum()
        _result = crss_temp_pnl_data.diff().mean()*100
        crss_pnl_plt_sg.plot(crss_temp_pnl_data, label=f'{prename}: {_result:.4f}')

        crss_temp_pnl_data_size = (_crss_res.groupby(['date'])[label].mean()).cumsum() - size_pool_pnl.iloc[:, 0]
        crss_temp_pnl_data_size = crss_temp_pnl_data_size.reindex(dates)
        _result = crss_temp_pnl_data_size.diff().mean()*100
        crss_pnl_plt_sgtp.plot(crss_temp_pnl_data_size, label=f'{prename}: {_result:.4f}')
        return_dict[f'rtn_{prename}_1d'] = _result

        size_pool_pnl_3d = res_df[['date', label_3d]].groupby(['date']).mean().cumsum()/3
        crss_temp_pnl_data_size_3d = (_crss_res.groupby(['date'])[label_3d].mean()).cumsum()/3 - size_pool_pnl_3d.iloc[:, 0]
        crss_temp_pnl_data_size_3d = crss_temp_pnl_data_size_3d.reindex(dates)
        _result = crss_temp_pnl_data_size_3d.diff().mean()*100
        crss_pnl_plt_sgtp_3d.plot(crss_temp_pnl_data_size_3d, label=f'{prename}: {_result:.4f}')
        return_dict[f'rtn_{prename}_3d'] = _result

        for srg in size_ratio_dict:
            if len(size_ratio_dict[srg]) != group_num:
                continue
            ratio_i = size_ratio_dict[srg][ii]
            
            if ratio_i == 0:
                continue
            crss_res_new = res_df[res_df['crss_rank_ongroup']>=1-crss_ratio_thr*ratio_i*5]
            size_ratio_rtn[f'size_ratio_{srg}'] = size_ratio_rtn[f'size_ratio_{srg}'].append(crss_res_new)                


    crss_temp_pnl_data = crss_res_group.groupby(['date'])[label].mean().cumsum()
    _result = crss_temp_pnl_data.iloc[-1]
    return_dict['rtn_equalweight'] = _result
    crss_pnl_plt.plot(crss_temp_pnl_data, label=f'EQUAL WEIGHT: {_result:.4f}')

    

    for srgn in size_ratio_rtn:
        crss_res_rg = size_ratio_rtn[srgn]
        crss_temp_pnl_data = crss_res_rg.groupby(['date'])[label].mean().cumsum()
        crss_pnl_plt.plot(crss_temp_pnl_data, label=srgn)            

    crss_pnl_plt_sg.legend(loc = 'upper left')
    crss_pnl_plt.legend(loc = 'upper left')
    crss_pnl_plt_sgtp.legend(loc = 'upper left')
    crss_pnl_plt_sgtp_3d.legend(loc = 'upper left')

    def get_num_dict(x):
        _l = list(x.ticker)
        _s = set(_l)
        _d = {i:_l.count(i) for i in _s}
        # print(len(_l), len(_s), len(_d))
        return _d
    def get_ticker_group(x):
        x = x.drop_duplicates()
        _d = {getattr(r, 'ticker'):getattr(r, 'group_') for r in  x.itertuples()}
        return _d

    ticker_date = crss_res_group.groupby(['date'])[['ticker']].apply(get_num_dict)
    ticker_unique = crss_res_group.groupby(['date'])['ticker'].unique()
    ticker_group_info = crss_res_group.groupby(['date'])[['ticker','group_']].apply(get_ticker_group)
    ticker_mean = crss_res_group.groupby(['date'])['ticker'].unique().apply(len).mean()

    info_list = []
    for i_, i_date in enumerate(ticker_date.index):
        ticker_n = ticker_unique.iloc[i_]
        if i_ == 0:
            _info = [0 for i in pre_name_list]

        else:
            ticker_num_n = ticker_date.iloc[i_]
            ticker_num_y = ticker_date.iloc[i_-1]
            ticker_num_g = ticker_group_info.iloc[i_]
            _info = []
            for i__ in range(len(pre_name_list)):
                _tr = [max(ticker_num_n[i]-ticker_num_y.get(i,0), 0) for i in ticker_n if ticker_num_g[i] == i__]
                _tr = np.sum(_tr)
                _info.append(_tr)
        
        info_list.append(_info)

    dates_2 = ticker_date.index
    df_crsstr = pd.DataFrame(info_list, index=dates_2, columns=pre_name_list)
    crss_turnover_plt = fig.add_subplot(cur_gs[2, 0])
    crss_turnover_plt.grid(linestyle="--", linewidth=1.5)
    crss_turnover_plt.set_xticks(range(0, len(dates_2), 20), )
    crss_turnover_plt.set_xticklabels([dates_2[idx] for idx in crss_turnover_plt.get_xticks()], rotation=20)
    pos_limit = group_dict.get('pos_limit', [100 for i in pre_name_list])
    tr_all = 0
    tr_ratio = group_dict.get('to_ratio', [i/sum(max_trade_times) for i in max_trade_times])
    for i_ in range(len(pre_name_list)):
        _ser = df_crsstr[pre_name_list[i_]]/pos_limit[i_]/max_trade_times[i_]
        crss_turnover_plt.plot(dates_2, _ser.rolling(5, min_periods=1).mean(), label=f'{pre_name_list[i_]} : {_ser.mean():.2}')
        tr_all += _ser.mean()*tr_ratio[i_]
    tr_all_n = tr_all*sum(pos_limit)

    return_dict['turnover_ratio'] = tr_all
    return_dict['turnover_chgnum'] = tr_all_n
    return_dict['turnover_tradenum'] = ticker_mean

    crss_turnover_plt.set_title(f'各分组换手率 {tr_all:.2f} 换 {tr_all_n:3.0f}总{ticker_mean:3.0f}: ma5')
    crss_turnover_plt.legend(loc='upper left')

    fig.suptitle(f'{label}', fontsize=20)
    cur_gs.tight_layout(fig)
    if 'save_fig_path' in group_dict:
        plt.savefig(group_dict['save_fig_path'])
    else:
        plt.show()    

    return return_dict





