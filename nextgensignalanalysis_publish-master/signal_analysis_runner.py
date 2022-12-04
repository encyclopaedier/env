import os

from signal_analysis_exec.SignalAnalysis import SignalAnalysis


RESULT_ANALYSIS_DATA_DIR = "/home/shared/Data/data/local_data/mid_frequence/102/mid_freq_bt_data_DONT_DELETE/" \
                           "result_analysis_data/"


if __name__ == "__main__":
    signal_dir = "sample_data"

    bm_index = "000905"
    bm_pool_path = os.path.join(RESULT_ANALYSIS_DATA_DIR, "TOP1800_ALL_EQ_WGT_POOL.pkl")

    market_regime_path = os.path.join(RESULT_ANALYSIS_DATA_DIR, "mvp1_classify_res_whole_market.pkl")
    market_regime_tag_path = os.path.join(RESULT_ANALYSIS_DATA_DIR, "mvp1_classify_res_whole_market_tag_dict.pkl")

    ticker_pool_path = os.path.join(RESULT_ANALYSIS_DATA_DIR, "index_stock_group_info.pkl")

    start_date = "20180102"
    end_date = None

    td_price_mode = "TwapBegin60"

    signal_name_list = [
        "eod_jxma_fund_combo_v1",
    ]

    signal_fmt = "pred_val"
    # signal_fmt = "weight"

    benchmark_fmt = "index"
    # benchmark_fmt = "pool"

    file_fmt = "ds"
    # file_fmt = "csv"

    signal_shift = 1

    for signal_name in signal_name_list:
        cur_saving_path = os.path.join("res", "{}_{}_{}".format(signal_name, start_date, end_date,))

        if not os.path.exists(cur_saving_path):
            os.makedirs(cur_saving_path)

        cur_run = SignalAnalysis(signal_dir, signal_name, signal_fmt, file_fmt, signal_shift, bm_index, bm_pool_path,
                                 start_date, end_date,
                                 td_price_mode, market_regime_path, market_regime_tag_path, ticker_pool_path,
                                 cur_saving_path, benchmark_fmt)
        cur_run.exec()
