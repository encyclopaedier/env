参数说明  
config.py中定义优化求解参数  
必须自定义的内容：  
地址：  
start_date/end_date: 优化起止时间  
data_path: 风格因子相关数据文件地址  
ml_factor_path: 预测收益率路径(eod_feature的上层路径，不需要进入eod_feature)  
out_put_signal_path: 优化结果输出地址  
求解目标：  
ml_factor_name: 预测收益率名称（不包含"eod_"前缀）例如，xgb_2021 -> eod_xgb_2021.h5  （signal文件名， 可以不读文件， 直接传入optimizer）
benchmark_index: 优化的benchmark指数   
penalty_lambda: 根据收益率预测值的方差进行调整，如果输入的收益率预测值与实际收益率波动近似，可以考虑在0.5-2的范围内调整。默认为1.  
turnover_limit: 换手率上限约束  
style_low_limit_individual/style_high_limit_individual: 可以对单个风格设置不同的约束  
weight_low/weight_high: 持仓权重上下限，对持仓股票数有较大影响  

我们的求解目标是最大化风险惩罚后的收益，并对风格/行业相对中证500的暴露约束在给定区间内，同时对换手率进行约束。最大持仓的限制会在优化中实现，但最小持仓限制是通过优化后剔除重新赋权  


目前稳定支持的优化方式为qp_method = 1


