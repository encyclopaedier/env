# coding: utf-8

config_param = {
    'signal_path': '/home/xqli/pycharm_project/nextgensignalanalysis/sample_data',
    'signal_name': 'final_score_5d_delay1_iter5_20220902_no_crtl',
    'start_date': '20210104',
    'end_date': '20220902',
    'save_path': '/home/xqli/pycharm_project/nextgensignalanalysis/sample_data',
    "input_fmt": "csv",
}

# for risk optimizer
style_limitation = 0.5

risk_opt_x = style_limitation
risk_opt_y = style_limitation
risk_opt_z = style_limitation
risk_opt_param = {
    # Experiment Config
    # "start_date": system_param["start_date"],
    # "end_date": system_param["end_date"],
    "ml_factor_path": './predictions.pkl',
    # "output_signal_path": system_param["output_path"],
    "number_of_thread": 10,
    "number_of_processor": 20,
    "opt_signal_lower_bound": None,
    "ml_factor_name": 'opt_weight',
    "class_name": ['residual_volatility', 'comovement', 'growth',
                   'momentum', 'liquidity', 'earnings_yield',
                   'size', 'leverage', 'book_to_price', 'beta',
                   'non_linear_size'],

    # Constraints and penalty
    "qp_method": 1,
    "benchmark_index": '399905',  # 比较基准
    "index_list": ['000985'],  # 可选票池

    "ratio": 0.8,  # 流动性票池参数
    "period": 60,  # 流动性票池参数

    "penalty_lambda":  1,
    "turnover_limit": 0.35,
    "obj_func": 'ret_var',

    "style_neutralize": True,  # 是否进行风格中性约束
    "set_individual_style_limit": True,   # 是否独立设置每个风格因子敞口上下限。若为True，则在下面的dict中手动设置每个敞口；若为False，则使用下面统一的敞口；
    "x": risk_opt_x,
    "y": risk_opt_y,
    "z": risk_opt_z,
    "style_low_limit_individual": {'liquidity': -risk_opt_x,
                                   'momentum': -risk_opt_x,
                                   'book_to_price': -risk_opt_x,
                                   'leverage': -risk_opt_x,
                                   'size': -risk_opt_y,
                                   'beta': -risk_opt_x,
                                   'residual_volatility': -risk_opt_x,
                                   'non_linear_size': -risk_opt_y,
                                   'comovement': -risk_opt_z,
                                   'earnings_yield': -risk_opt_x,
                                   'growth': -risk_opt_x},

    "style_high_limit_individual": {'liquidity': risk_opt_x,
                                    'momentum': risk_opt_x,
                                    'book_to_price': risk_opt_x,
                                    'leverage': risk_opt_x,
                                    'size': risk_opt_y,
                                    'beta': risk_opt_x,
                                    'residual_volatility': risk_opt_x,
                                    'non_linear_size': risk_opt_y,
                                    'comovement': risk_opt_z,
                                    'earnings_yield': risk_opt_x,
                                    'growth': risk_opt_x},
    "index_constraint": False,
    "index_low_limit": {'000300': 0.1,
                        '000905': 0.2,
                        '000852': 0.1,
                        'others': 0},

    "index_high_limit": {'000300': 0.4,
                         '000905': 0.7,
                         '000852': 0.7,
                         'others': 0.5},

    "ind_neutralize": False,  # 是否进行行业中性约束
    "ind_low_limit": -0.05,  # 行业中性因子统一敞口
    "ind_high_limit": 0.05,

    "weight_low": 0.0001,   # 权重下限
    "weight_high": 0.001,  # 权重上限, 当使用优化器时, 股票数量由weight_high参数决定

    "print_process": True
}
