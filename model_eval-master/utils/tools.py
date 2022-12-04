import multiprocessing
import os
import shutil
import pandas as pd
from PqiDataSdk import *
import numpy as np
import pyarrow.parquet as pq

def get_pred_from_tradefile(pred_path):
    _results = []
    file_list = sorted(os.listdir(f'{pred_path}/y_pred'))
    for i_f in file_list:
        _df = pd.read_pickle(f'{pred_path}/y_pred/{i_f}')
        _df = _df.stack().reset_index()
        _df.columns = ['ticker', 'TimeStamp', 'model_pred']
        _df['date'] = i_f[:-4]
        _results.append(_df)
    _results = pd.concat(_results)
    return _results

def get_pred_from_modelpred(model_name, model_path):
    _df_pred_temp = pd.read_pickle(f'{model_path}/{model_name}/model_pred.pkl')
    return _df_pred_temp


def output2trade(save_path, res_df_out=pd.DataFrame()):
    def check_dirs(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    model_result_name = save_path

    if not os.path.exists(model_result_name):
        os.mkdir(model_result_name)

    check_dirs(f'{model_result_name}/y_pred')
    check_dirs(f'{model_result_name}/ticker')

    res_df_out = res_df_out.rename(columns={'model_pred': 'y_pred'})

    date_list = res_df_out.date.drop_duplicates()
    print(res_df_out.shape)
    def gen_trade_file(res_df_out, date_list, result_path):
        for date in date_list:
            _df = res_df_out[res_df_out.date==date]
            tickers = _df.ticker.drop_duplicates().sort_values()
            timestamp = _df.TimeStamp.drop_duplicates().sort_values()

            _pred = _df.pivot_table(values='y_pred', index='ticker',columns='TimeStamp')
            _pred = _pred.reindex(index=tickers, columns=timestamp)
            _pred.to_pickle(f'{result_path}/y_pred/{date}.pkl')

            with open(f'{result_path}/ticker/{date}.txt', 'w') as f:
                f.write(','.join(tickers.to_list()))


    p_list = []
    for i in range(32):
        _date_list = date_list[i::32]
        p = multiprocessing.Process(target=gen_trade_file, args=(res_df_out, _date_list, model_result_name))
        p.start()

        p_list.append(p)
    
    for i in range(32):
        p = p_list[i]
        p.join()
    print(model_result_name)

def get_eval_df(start_date, end_date, **kwargs):
    # 获取评价用收益
    ds = PqiDataSdk(user="yzhou", size=1, pool_type="mp", str_map=False)
    all_dates = ds.get_trade_dates(start_date=start_date, end_date=end_date)
    data_path = kwargs.get('data_path', '/data/local_data/shared/102/intern_data_yzhou/zy4_parquet')
    files = [f'{data_path}/{i}.parquet' for i in all_dates] #所有数据文件

    y_1d = kwargs.get('y_1d', 'm1_ts_y_60twap_2n1open_Wgted_fullmarket_ex') #指定1天后收益y值
    y_2d = kwargs.get('y_2d', 'm1_ts_y_60twap_2n2open_Wgted_fullmarket_ex') #指定2天后收益y值
    y_3d = kwargs.get('y_3d', 'm1_ts_y_60twap_2n3open_Wgted_fullmarket_ex') #指定3天后收益y值
    zt_filter = ('m1_ts_z_tag_up_limit','=',0.0)
    dt_filter = ('m1_ts_z_tag_down_limit','=',0.0)
    yna_filter = (y_1d,'!=',np.nan)
    filters = [zt_filter,dt_filter,yna_filter]
    dataset = pq.ParquetDataset(files, use_legacy_dataset=False, filters=filters)
    cols = ['date', 'TimeStamp','ticker', 'm1_ts_y_mktval', y_1d, y_2d, y_3d]
    df_eval = dataset.read(columns=cols).to_pandas()
        
    return df_eval



def get_index_group(new_test, index_group=['000300','000905','000852', 'others']):
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
        
    return new_test['index_group']

def check_dirs(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_trade_filter(eod_filter_info={}):
    start_date = eod_filter_info.get('start_date', '20160101')
    end_date = eod_filter_info.get('end_date', '20211231')
    TradeValueMA10 = eod_filter_info.get("TradeValueMA10", 30000)
    STStatus = eod_filter_info.get("STStatus", 1)
    ds = PqiDataSdk(user="yzhou", size=1, pool_type="mp", str_map=False)
    cur_stock_eod_data = ds.get_eod_history(start_date=start_date,end_date=end_date,
                                            fields=["TradeStatus", "TradeValue","STStatus"])
    cur_stock_eod_data["TradeValueMA10"] = cur_stock_eod_data["TradeValue"].rolling(10, axis=1).mean().shift(1, axis=1)
    filter_all = cur_stock_eod_data["TradeValue"].copy()
    filter_all.loc[:, :] = 0
    f1 = cur_stock_eod_data["TradeValueMA10"]>TradeValueMA10
    f2 = cur_stock_eod_data["STStatus"]<STStatus
    filter_all[f1&f2] = 1
    check_dirs('./temp_data')
    filter_all.to_pickle('./temp_data/eod_filter_mp.pkl')
    # filter_all = filter_all.stack().reset_index()
    # filter_all.columns=['ticker', 'date', 'eod_filter']
    # filter_all.to_pickle('./temp_data/eod_filter.pkl')
    return filter_all

def check_eod_filter(test_data, eod_filter_info={}):
    if os.path.exists('./temp_data/eod_filter_mp.pkl'):
        filter_all = pd.read_pickle('./temp_data/eod_filter_mp.pkl')
    else:
        filter_all = get_trade_filter(eod_filter_info)
    filter_all = filter_all.stack().reset_index()
    filter_all.columns=['ticker', 'date', 'eod_filter']
    new_test = pd.merge(test_data, filter_all, how='left', )
    new_test = new_test[new_test.eod_filter==1]
    return new_test


def rolling_method(df_pred, rolling_ratio):
    df_pred_rolling = []
    TimeStamp_all = df_pred.TimeStamp.drop_duplicates().values
    for i_t in TimeStamp_all:
        df_pred_part = df_pred[df_pred.TimeStamp==i_t]
        pred_ori = df_pred_part.pivot_table(values='model_pred', index='ticker',columns='date')
        pred_ = pred_ori*rolling_ratio[0]
        for i_r in range(1, len(rolling_ratio)):
            pred_ += pred_ori.shift(i_r, axis=1).fillna(0)*rolling_ratio[i_r]
            
        pred_['ticker'] = pred_.index
        pred_ = pd.merge(df_pred_part[['date','ticker']], pd.melt(pred_, id_vars=['ticker']), how='left')['value'].values
        df_pred_part['model_pred'] = pred_
        df_pred_rolling.append(df_pred_part)


    df_pred_rolling = pd.concat(df_pred_rolling)
    df_pred_rolling = pd.merge(df_pred[['date','TimeStamp','ticker']], df_pred_rolling, how='left')
    return df_pred_rolling

def simple_stacking(stacking_list, stacking_type='rank'):

    for i, _df_pred_temp in enumerate(stacking_list):

        if stacking_type == 'rank':
            _df_pred_temp['model_pred'] = _df_pred_temp.groupby(['date','TimeStamp'])['model_pred'].rank(axis=0, pct=True)
        elif stacking_type == 'origin':
            pass
        if i == 0:
            df_pred = _df_pred_temp
        else:
            df_pred = pd.merge(df_pred, _df_pred_temp, how='inner', on=['date','ticker', 'TimeStamp'])
            df_pred['model_pred'] = df_pred['model_pred_x']+df_pred['model_pred_y']
            df_pred = df_pred[['date','TimeStamp', 'ticker', 'model_pred']]

    return df_pred

