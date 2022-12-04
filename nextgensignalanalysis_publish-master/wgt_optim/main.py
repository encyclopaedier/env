# coding: utf-8
import os

import pandas as pd
import configure
import opt_solver
import pickle
from configure import risk_opt_param
from configure import config_param
from PqiDataSdk import *

default_number_of_thread = 10
os.environ["MKL_NUM_THREADS"] = str(default_number_of_thread)
os.environ["NUMEXPR_NUM_THREADS"] = str(default_number_of_thread)
os.environ["OMP_NUM_THREADS"] = str(default_number_of_thread)
os.environ["OPENBLAS_NUM_THREADS"] = str(default_number_of_thread)
os.environ["OPENBLAS_NUM_THREADS"] = str(default_number_of_thread)


if __name__ == '__main__':
    config_dict = config_param.copy()
    ds = PqiDataSdk(user='yhzhou', size=1)
    date_list = ds.get_trade_dates(start_date=config_dict['start_date'], end_date=config_dict['end_date'])

    if config_dict["input_fmt"] == "ds":
        signal = ds.get_eod_feature(fields=[config_dict['signal_name']],
                                    where=config_dict['signal_path'],
                                    dates=date_list)[config_dict['signal_name']].to_dataframe()
        signal = signal.dropna(how='all', axis=1)
    elif config_dict["input_fmt"] == "csv":
        signal = pd.read_csv(os.path.join(config_dict['signal_path'], "{}.csv".format(config_dict['signal_name'])),
                             index_col=[0])

        signal.index = [str(x).zfill(6) for x in signal.index]
        signal.columns = [str(x) for x in signal.columns]

        signal = signal.loc[:, date_list]
    # signal = signal.T.loc['20210130':].T

    # multiple dates run at a time, without turnover confinement
    # need to set turnover_limit in configure to None
    risk_opt_param["turnover_limit"] = None
    final_weight_matrix = opt_solver.run_opt(configure_dict=risk_opt_param, signal_data=signal,
                                             ticker_list=list(signal.index),
                                             date_list=list(signal.columns),
                                             process_number=risk_opt_param["number_of_processor"],
                                             confine_turnover=False)
    print("multiple dates, without turnover confinement")
    print(final_weight_matrix.T)

    final_weight_matrix.T.to_csv('./opt_result_magic_conch.csv')
    ds.save_eod_feature(data={'eod_opt_style{}_{}'.format(configure.style_limitation, config_dict['signal_name'][4:]): final_weight_matrix.T},
                        where=config_dict['save_path'],
                        feature_type='eod',
                        save_method='update',
                        encrypt=False)