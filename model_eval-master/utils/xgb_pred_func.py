import utils
import pandas as pd
import numpy as np
import random
import pickle
import xgboost as xgb
import pyarrow.parquet as pq

from PqiDataSdk import *
import itertools
import os

ds = PqiDataSdk(user='yzhou')
def get_pred_data(data_path, start_date, end_date, feas):
    all_dates = ds.get_trade_dates(start_date=start_date, end_date=end_date)
    files = [f'{data_path}/{i}.parquet' for i in all_dates]
    dataset = pq.ParquetDataset(files, use_legacy_dataset=False)
    df_factor = dataset.read(columns=feas).to_pandas()
    return df_factor

def get_xgb_pred(model_path, model_name, start_date, end_date, data_path, **kwags):

    model_subpath = kwags.get('model_subpath', model_name)
    if kwags.get('from_file', True) and os.path.exists(f'{model_path}/{model_subpath}/model_pred.pkl'):
        print('from file')
        _df_pred_temp = pd.read_pickle(f'{model_path}/{model_subpath}/model_pred.pkl')
        return _df_pred_temp

    _temp_model = xgb.Booster()
    model_name_final = f'{model_path}/{model_subpath}/{model_name}.model'
    fea_name_final = f'{model_path}/{model_subpath}/feas_{model_name}.pkl'
    _temp_model.load_model(model_name_final)
    _temp_feas = pd.read_pickle(fea_name_final)

    if kwags.get('transfea', None) == 'x2fea':
        x2fea_dict = pd.read_pickle('/data/local_data/mid_frequence/102/zy_results/intern_manage/zy4data_colname_dict.pkl')
        fea2x_dict = {x2fea_dict[i]:i for i in x2fea_dict}
        _temp_feas = [x2fea_dict[i] for i in _temp_feas]

        
    elif kwags.get('transfea', None) == 'fea2x':
        x2fea_dict = pd.read_pickle('/data/local_data/mid_frequence/102/zy_results/intern_manage/zy4data_colname_dict.pkl')
        fea2x_dict = {x2fea_dict[i]:i for i in x2fea_dict}
        _temp_feas = [fea2x_dict[i] for i in _temp_feas]


    if 'df_factor' in kwags:
        df_factor = kwags['df_factor']
        _df_pred_temp = df_factor[['date', 'TimeStamp', 'ticker']]
        test_data_part = df_factor.iloc[:, [list(df_factor.columns).index(i) for i in _temp_feas]].replace([np.inf, -np.inf], np.nan)        
    else:
        _df_pred_temp = get_pred_data(data_path, start_date, end_date, ['date', 'TimeStamp', 'ticker'])
        test_data_part = get_pred_data(data_path, start_date, end_date, _temp_feas)

    test_data_part = test_data_part.replace([np.inf, -np.inf], np.nan)

    pred_res_model_part = _temp_model.predict(xgb.DMatrix(test_data_part))
    _df_pred_temp['model_pred'] = pred_res_model_part

    if kwags.get('save', False):
        _df_pred_temp.to_pickle(f'{model_path}/{model_subpath}/model_pred.pkl')
        print('save success')
    return _df_pred_temp