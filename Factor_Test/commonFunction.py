import math
import statsmodels.api as sma
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import datetime
import time
from functools import wraps


# 计时器
def timer(function):
    @wraps(function)
    def function_timer(*args,**kwargs):
        start_time = datetime.datetime.now()
        result = function(*args, **kwargs)
        end_time = datetime.datetime.now()
        running_time = end_time-start_time
        print("Running time:{}".format(running_time))
        return result
    return function_timer


def rtn2nav(rtn):
    '''
    return变成net asset value
    '''
    nav = (rtn + 1).cumprod()
    return nav


def nav2rtn(nav):
    '''
    net asset value变成return
    '''
    return nav.pct_change()

    
def grouping(data, n):
    '''
    计算分层收益
    '''
    size = data.shape
    group = []
    for i in range(0, size[0]):
        df_x = data.iloc[i, :]
        df_x = df_x.dropna()
        quantiles = [x / n for x in range(n + 1)]
        group_temp = []
        for i in range(n):
            q0 = df_x.quantile(quantiles[i])
            q1 = df_x.quantile(quantiles[i + 1])
            x_i = df_x[(df_x >= q0) & (df_x < q1 + 1e-10)]
            group_temp.append(x_i.index.tolist())
        group.append(group_temp)

    return group


def calTurnover(group):
    '''
    计算每一组换手率
    '''
    size = np.shape(group)
    turnover = []
    for t in range(1, size[0]):
        turnover_temp = []
        for g in range(0, size[1]):
            num_change = len(set(group[t][g]) - set((group[t - 1][g])))
            num = len(group[t][g])
            if num > 0:
                turnover_temp.append(num_change / num)
            else:
                turnover_temp.append(0)
        turnover.append(turnover_temp)
    return turnover


def calc_layer_returns(df_factor, df_f_ret, ret_lag=1, n_quantile=10):
    '''
    计算分层收益
    '''

    def quantile_ret(df_x, df_ret, n=10):
        try:
            df_x = df_x.dropna()
            quantiles = [x / n for x in range(n + 1)]
            df_mu = []
            for i in range(n):
                q0 = df_x.quantile(quantiles[i])
                q1 = df_x.quantile(quantiles[i + 1])
                x_i = df_x[(df_x >= q0) & (df_x < q1 + 1e-10)]
                ret_i = df_ret.reindex(x_i.index).dropna()

                mu_i = ret_i.mean()
                df_mu.append({'group': i + 1, 'mu': mu_i})
            df_mu = pd.DataFrame(df_mu).set_index('group')['mu']
        except Exception as e:
            pass
        return df_mu

    df_ret_1 = df_f_ret.copy().reindex(df_factor.index)
    df_factor = df_factor.reindex(df_ret_1.index)

    df_group_ret = df_factor.apply(lambda x: quantile_ret(x, df_ret_1.loc[x.name], n=n_quantile), axis=1)
    df_group_ret = df_group_ret.shift(ret_lag)
    return df_group_ret


def getNavStat(rtn):
    '''
    计算 net asset value的performance ratios
    '''
    nav = rtn2nav(rtn)
    totalRtn = nav.iloc[-1, :] - 1
    size = nav.shape
    yearlyRtn = pow(totalRtn + 1, 240 / size[0]) - 1  # 几何平均

    dailyVol = rtn.std()  # pd中是无偏估计
    yearlyVol = dailyVol * math.sqrt(240)
    sr = yearlyRtn / yearlyVol

    winning = rtn > 0
    winningRatio = winning.sum() / rtn.shape[0]

    rtn_pos = rtn.copy()
    rtn_neg = rtn.copy()
    rtn_pos[rtn < 0] = 0
    rtn_neg[rtn > 0] = 0
    plRatio = -rtn_pos.mean() / rtn_neg.mean()

    navStat = list(zip(totalRtn, yearlyRtn, yearlyVol, sr, winningRatio, plRatio))
    navStat = pd.DataFrame(navStat).T
    navStat.index = ['Total Rtn', 'Annualized Rtn', 'Annualized Vol', 'Sharpe Ratio', 'Winning Ratio', 'PL Ratio']
    navStat.columns = rtn.columns

    dd, ddStat, _, _ = getMaxDD(nav)
    navStat = navStat.append(ddStat)

    return nav, dd, navStat


def getMaxDD(nav):    
    
    size = nav.shape
    dd = np.zeros(size)    
    idx1 = np.zeros(size)
    idx2 = np.zeros(size)
    for j in range(0,size[1]):
        
        max_temp = -np.inf
        idx1_temp = -1
        idx2_temp = -1
        for i in range(0,size[0]):            
            if max_temp < nav.values[i,j]:
                max_temp = nav.values[i,j]
                idx1_temp = i
                idx2_temp = i
            else:
                idx2_temp = i
                   
            dd[i,j] = nav.values[i,j]/max_temp-1   
            idx1[i,j] = idx1_temp
            idx2[i,j] = idx2_temp
    
    # 统计最大回撤
    maxdd = dd.min(0) # 按列求MIN
    idx = np.argmin(dd,0) # 求第一个最小值所在的位置
    
    startIdx = []
    endIdx = []
    for j in range(0,size[1]):
        startIdx.append(idx1[idx[j]][j])
        endIdx.append(idx2[idx[j]][j])    

    times = pd.Series(nav.index)
    startDate = times.iloc[startIdx]
    endDate = times.iloc[endIdx]
    
    # 转换格式
    dd = pd.DataFrame(dd)
    dd.index = nav.index
    dd.columns = nav.columns   
    
    ddStat = list(zip(maxdd,startDate,endDate)) # 合并series
    ddStat = pd.DataFrame(ddStat).T
    ddStat.index = ['maxDD','startDate','endDate']
    ddStat.columns = nav.columns    
    
    return dd,ddStat,idx1,idx2


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def getICStat(factor, rtn):
    num = rtn.shape[0]
    ic = np.zeros((num, 1))  # 入参是tuple型
    for k in range(0, num):
        try:
            ic[k] = factor.iloc[k, :].corr(rtn.iloc[k, :])
        except Exception as e:
            ic[k] = 0

    ic = pd.DataFrame(ic)
    ic.columns = ['IC']
    ic.index = rtn.index
    ic = ic.shift(1)

    icStat = getDataStat(ic)

    return ic, icStat


def getRankICStat(factor, rtn):
    factor_rank = factor.rank(axis=1)  # rank直接忽略nan
    rtn_rank = rtn.rank(axis=1)

    num = rtn.shape[0]
    rankIC = np.zeros((num, 1))  # 入参是tuple型
    for k in range(0, num):
        try:
            rankIC[k] = factor_rank.iloc[k, :].corr(rtn_rank.iloc[k, :])
        except Exception as e:
            rankIC[k] = 0

    rankIC = pd.DataFrame(rankIC)
    rankIC.columns = ['rankIC']
    rankIC.index = rtn.index
    rankIC = rankIC.shift(1)

    rankICStat = getDataStat(rankIC)

    return rankIC, rankICStat


def cal_corr(factor_name_A, factor_name_B, factor_path_A, factor_path_B):

    df_A = pd.read_csv(factor_path_A, index_col=0)
    df_B = pd.read_csv(factor_path_B, index_col=0)

    corr_mean = df_A.corrwith(df_B, method='spearman', axis=1).mean()    # 需要考虑加入方向吗？或者计算std或者mean/std

    return {'factor_name_A': factor_name_A, 'factor_name_B': factor_name_B, 'corr': corr_mean}


def getDataStat(data):
    
    data = data.dropna()
    t_value, pvalue = stats.ttest_1samp(data, 0)  # 计算ic的t
    dataStat = list(zip(data.mean(), data.std(), data.skew(), data.kurt(), t_value))
    dataStat = pd.DataFrame(dataStat).T
    dataStat.columns = data.columns
    dataStat.index = ['mean', 'std', 'skew', 'kurt', 't_value']

    return dataStat


def getCSRegStat(factor, rtn):
    # 横截面一元线性回归
    size = factor.shape
    beta = np.full((2, size[0]), np.nan)
    t = np.full((2, size[0]), np.nan)
    regStat = np.full((2, size[0]), np.nan)  # 入参是tuple型

    for k in range(0, size[0]):
        try:
            a = factor.iloc[k, :]
            b = rtn.iloc[k, :]

            valid = a.to_frame('factor').join(b.to_frame('rtn'), how='outer').dropna()
            x = valid['factor']
            y = valid['rtn']
            X = sma.add_constant(x)
            model = sma.OLS(y, X)
            results = model.fit()

            beta_temp = results.params.squeeze()
            beta[:, k] = np.array(beta_temp)

            t_temp = results.tvalues
            t[:, k] = np.array(t_temp)

            regStat[0, k] = results.fvalue
            regStat[1, k] = results.rsquared_adj
        except Exception as e:
            print(e)

    regStat = pd.DataFrame(regStat).T
    regStat.columns = ['F', 'adj_R2']
    regStat.index = factor.index

    beta = pd.DataFrame(beta).T
    beta.index = factor.index
    beta.columns = ['beta0', 'beta1']

    t = pd.DataFrame(t).T
    t.index = factor.index
    t.columns = ['t0', 't1']

    return beta, t, regStat


def getDataStat(data):
    
    dataStat = list(zip(data.mean(),data.std(),data.skew(),data.kurt()))
    dataStat = pd.DataFrame(dataStat).T
    dataStat.columns = data.columns
    dataStat.index = ['mean','std','skew','kurt']
   
    return dataStat