import pandas as pd
import numpy as np
import random
import pickle
import pyarrow.parquet as pq
from PqiDataSdk import *
from numba import njit, prange


def get_index_group(new_test, index_group=['000300','000905','000852', 'others'], keep_index=None):
    ds = PqiDataSdk(user="yzhou", size=1, pool_type="mp", str_map=False)
    date_list = new_test.date.drop_duplicates()

    date_start, date_end = date_list.iloc[0], date_list.iloc[-1]
    cur_hs300_info = ds.get_index_weight(ticker="000300", start_date=date_start, end_date=date_end, format='eod').T
    cur_hs300_info[cur_hs300_info>0] = 0

    cur_zz500_info = ds.get_index_weight(ticker="000905", start_date=date_start, end_date=date_end, format='eod').T
    cur_zz500_info[cur_zz500_info>0] = 1

    cur_zz1000_info = ds.get_index_weight(ticker="000852", start_date=date_start, end_date=date_end, format='eod').T
    cur_zz1000_info[cur_zz1000_info>0] = 2

    cur_index_info = pd.concat([cur_hs300_info, cur_zz500_info, cur_zz1000_info])
    cur_index_info = cur_index_info.stack()
    cur_index_info = cur_index_info.reset_index() 
    cur_index_info.columns = ['date', 'ticker', 'index_group']

    new_test['TimeStamp'] = new_test['TimeStamp'].apply(lambda x:str(int(x)))
    new_test = pd.merge(new_test, cur_index_info, how='left', )
    new_test['index_group'].fillna(3, inplace=True)
    if '399006' in index_group or 'cyb' in index_group:
        is_cyb = new_test.ticker.str[:2] == '30'
        new_test.loc[is_cyb, 'index_group'] = 4
    if keep_index is not None:
        if keep_index == 'all':
            print(f"keep_index: {keep_index}")
        elif keep_index == '000300': 
            new_test = new_test[new_test.index_group == 0].reset_index(drop=True)
        elif keep_index == '000905': 
            new_test = new_test[new_test.index_group == 1].reset_index(drop=True)
        elif keep_index == '000852': 
            new_test = new_test[new_test.index_group == 2].reset_index(drop=True)
        elif keep_index == 'others': 
            new_test = new_test[new_test.index_group == 3].reset_index(drop=True)
        else:
            raise ValueError("Illegal input for keep_index")
        return new_test
    else:
        return new_test['index_group']

def get_eval_result(param_dict, df_eval, df_pred):
    ds = PqiDataSdk(user="wpxu", size=1, pool_type="mp", str_map=False)
    y_name = param_dict['y_name']
    start_date = param_dict['start_date']
    end_date = param_dict['end_date']
    res_df_all = pd.merge(df_eval, df_pred, how='left')
    if 'index_group' in param_dict:
        index_info = get_index_group(res_df_all)
        # print(res_df_all)
        if param_dict['index_group'] == '000300': 
            print(len(res_df_all))
            res_df_all = res_df_all[index_info == 0].reset_index(drop=True)
            print(len(res_df_all))
        elif param_dict['index_group'] == '000905': 
            print(len(res_df_all))
            res_df_all = res_df_all[index_info == 1].reset_index(drop=True)
            print(len(res_df_all))
        elif param_dict['index_group'] == '000852': 
            print(len(res_df_all))
            res_df_all = res_df_all[index_info == 2].reset_index(drop=True)
            print(len(res_df_all))
        elif param_dict['index_group'] == 'others': 
            print(len(res_df_all))
            res_df_all = res_df_all[index_info == 3].reset_index(drop=True)
            print(len(res_df_all))
    
    res_df_all = res_df_all[(res_df_all.date>=start_date)&(res_df_all.date<=end_date)]
    crss_ratio_thr = param_dict.get('crss_ratio_thr', 0.05)
    res_df_all['crss_rank'] = res_df_all.groupby(['date', "TimeStamp"])['model_pred'].rank(axis=0, pct=True)
    crss_res = res_df_all[res_df_all['crss_rank']>=1-crss_ratio_thr]
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
    data_fe = ds.get_factor_exposure(start_date=start_date,end_date=end_date,factors=['book_to_price','size','residual_volatility'])
    data_iw = ds.get_index_weight(ticker = index_code, start_date=start_date,end_date=end_date,format='eod')

    exp_size = ((holds_info/holds_info.sum()*data_fe['size']).sum() - (data_fe['size']*data_iw/data_iw.sum()).sum())
    result_dict['exp_size_mean'] = exp_size.mean()
    result_dict['exp_size_min'] = exp_size.min()
    result_dict['exp_size_max'] = exp_size.max()
    exp_bp = ((holds_info/holds_info.sum()*data_fe['book_to_price']).sum() - (data_fe['book_to_price']*data_iw/data_iw.sum()).sum())
    result_dict['exp_bp_mean'] = exp_bp.mean()
    result_dict['exp_bp_min'] = exp_bp.min()
    result_dict['exp_bp_max'] = exp_bp.max()
    exp_rv = ((holds_info/holds_info.sum()*data_fe['residual_volatility']).sum() - (data_fe['residual_volatility']*data_iw/data_iw.sum()).sum())
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

if __name__ == '__main__':
    
    df_pred = pd.read_pickle('/home/wpxu/nn_experiment/nn_models/mlp_wpxu_0702_yrank-group-123d_fea469_17012005_5min/model_pred.pkl')
    
    # 获取评价用收益
    # 起始日期
    date_s = '20200601'
    # 结束日期
    date_e = '20210331'
    y_name = 'm1_ts_y_60twap_2n1open_Wgted_fullmarket_ex' #指定1天后收益y值

    ds = PqiDataSdk(user="wpxu", size=1, pool_type="mp", str_map=False)
    dates = ds.get_trade_dates(start_date=date_s, end_date=date_e)
    data_path = '/data/local_data/shared/102/intern_data_yzhou/zy4_parquet'
    files = [f'{data_path}/{i}.parquet' for i in dates] #所有数据文件

    zt_filter = ('m1_ts_z_tag_up_limit','=',0.0)
    dt_filter = ('m1_ts_z_tag_down_limit','=',0.0)
    yna_filter = (y_name,'!=',np.nan)
    filters = [zt_filter,dt_filter,yna_filter]
    dataset = pq.ParquetDataset(files, use_legacy_dataset=False, filters=filters)
    cols = ['date', 'TimeStamp','ticker', y_name]
    df_eval = dataset.read(columns=cols).to_pandas()
    
    param_dict = {
            'crss_ratio_thr': 0.03,
            'max_trade_time': 1,
            'index_code':'000905',
            'start_date':'20200601',
            'end_date':'20210331',
            'neu_start_date':'20200601',
            'neu_end_date':'20210331',
            'y_name': y_name
        }
    
    print(get_neu_result(param_dict, df_eval, df_pred))
    
    
    
    
