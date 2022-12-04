import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from PqiDataSdk import *    
import onnx
import onnxruntime
import multiprocessing
import shutil
ds = PqiDataSdk(user='yzhou')


sessionOptions = onnxruntime.SessionOptions()
# sessionOptions.intra_op_num_threads = 1
# sessionOptions.inter_op_num_threads = 1

DATA_PATH = '/data/local_data/shared/102/intern_data_yzhou/zy4_parquet'
def get_pred_df_mp(data_path, model_path, feas, mean_std_df, date_start, stacking_type, transfea=None):
    nn_models = []
    for i in os.listdir(model_path):
        if 'onnx' not in i:
            continue
        # print(i)
        nn_models.append(onnxruntime.InferenceSession(os.path.join(model_path, i), providers=['CPUExecutionProvider'], sess_options=sessionOptions))

    files = f'{data_path}/{date_start}.parquet'
    
    means = mean_std_df.loc[feas, 'mean']
    stds = mean_std_df.loc[feas, 'std'] 

    if transfea == 'x2fea':
        x2fea_dict = pd.read_pickle('/data/local_data/mid_frequence/102/zy_results/intern_manage/zy4data_colname_dict.pkl')
        fea2x_dict = {x2fea_dict[i]:i for i in x2fea_dict}
        feas = [x2fea_dict[i] for i in feas]
        means.index = feas
        stds.index = feas
        
    elif transfea == 'fea2x':
        x2fea_dict = pd.read_pickle('/data/local_data/mid_frequence/102/zy_results/intern_manage/zy4data_colname_dict.pkl')
        fea2x_dict = {x2fea_dict[i]:i for i in x2fea_dict}
        feas = [fea2x_dict[i] for i in feas]
        means.index = feas
        stds.index = feas

        
    dataset = pq.ParquetDataset(files, use_legacy_dataset=False)
    df_pred = dataset.read(columns=['date', 'TimeStamp', 'ticker']).to_pandas()
    df_pred['model_pred'] = 0.0
    _df_pred = df_pred.copy()

    test_x = dataset.read(columns=feas).to_pandas()
    
    test_x.replace([-np.inf, np.inf], np.nan, inplace=True)
    # test_x.fillna(method='ffill', inplace=True)
    
    test_x = (test_x - means) / (stds + 1e-10)
    test_x.fillna(0, inplace=True)

    test_x = test_x.astype(np.float32)
    input = test_x.values
    for nn_model in nn_models:
        ort_output = nn_model.run(['output_0'], {'input_0': input})[0] 
        _df_pred['model_pred'] = ort_output.reshape(-1)
        if stacking_type == 'rank':
            _df_pred['model_pred'] = _df_pred.groupby(['TimeStamp'])['model_pred'].rank(axis=0, pct=True)
        df_pred['model_pred'] += _df_pred['model_pred']

    print(date_start, 'finished')
    return df_pred

def get_nn_pred(data_path, model_path, start_date, end_date, transfea, stacking_type, **kwargs):
    if kwargs.get('from_file', True) and os.path.exists(f'{model_path}/model_pred.pkl'):
        df_pred_all = pd.read_pickle(f'{model_path}/model_pred.pkl')
        return df_pred_all


    all_dates = ds.get_trade_dates(start_date=start_date, end_date=end_date)
    feas = pd.read_pickle(f'{model_path}/feas.pkl')
    mean_std_df = pd.read_pickle(f'{model_path}/mean_std_all.pkl')

    mul_dfs = []
    df_pred_all = []
    pool_size = kwargs.get('pool_size', 1)
    if pool_size > 1:
        pool = multiprocessing.Pool(32)
        for i, dt in enumerate(all_dates):
            mul_dfs.append(pool.apply_async(get_pred_df_mp, (data_path, model_path, feas, mean_std_df, dt, stacking_type)))

        for item in mul_dfs:
            df_pred_all.append(item.get())
        pool.close()
    elif pool_size == 1:
        for i, dt in enumerate(all_dates):
            df_pred_all.append(get_pred_df_mp(data_path, model_path, feas, mean_std_df, dt, stacking_type, transfea))


    df_pred_all = pd.concat(df_pred_all)
    df_pred_all = df_pred_all.sort_values(by=['date','TimeStamp','ticker'])
    if kwargs.get('save', False):
        df_pred_all.to_pickle(f'{model_path}/model_pred.pkl')
    return df_pred_all
