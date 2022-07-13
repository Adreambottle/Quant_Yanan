didi
import os
import pandas as pd
import numpy as np
import datetime
import commonFunction as func
import commonFigure as figFunc
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

# Create the class of parameter for backtest
class BacktestParm(object):
    """
    Parameter format for Back Test 
    """
    
    def __init__(self, groupNum, startDate, endDate, factorDirection):
        """
        groupNum: The number of groups for all the stocks
        factorDirection: Ascending or Descending
        """
        self.groupNum = groupNum
        self.startDate = startDate
        self.endDate = endDate
        self.factorDirection = factorDirection
        
class BacktestResults(object):
    def __init__(self, groupRtn, groupNav, groupDD, groupNavStat):
        self.groupRtn = groupRtn
        self.groupNav = groupNav
        self.groupDD = groupDD
        self.groupNavStat = groupNavStat

class FigFormat(object):
    def __init__(self, figSize,fontSize,color):
        self.figSize = figSize
        self.fontSize = fontSize
        self.color = color

# Daily BackTest
class Backtest(object):
    def __init__(self, dataPath, factorPath, factorFile, backtestParm, figFormat, cost, mask, shift_num):
        self.dataPath = dataPath
        self.factorPath = factorPath
        self.backtestParm = backtestParm
        self.figFormat = figFormat
        self.shift_num = shift_num          # The number of shift window
        self.cost = cost
        
        # Extract the daily close price for the market. cp: close price
        cp = pd.read_csv(os.path.join(dataPath, "close_price.csv"), index_col=0, parse_dates=True)
        cp = cp.loc[backtestParm.startDate: backtestParm.endDate]
        stock_return = cp.pct_change(shift_num).shift(-shift_num)

        # Extract the daily data for the factor
        factor = pd.read_csv(os.path.join(factorPath, factorFile), index_col=0, parse_dates=True)
        factor = factor.loc[backtestParm.startDate: backtestParm.endDate]
        
        # If the factor direction is descending, the factor should be taken negative
        if self.backtestParm.factorDirection == 'ascending':
            pass
        elif self.backtestParm.factorDirection == 'descending':
            factor = -factor
        else:
            raise Exception('factorDirection only could be ascending or descending')
        
        # Reindex the stock_return by the factor columns
        stock_return = stock_return.reindex(columns=factor.columns)
        mask = mask.reindex(index=factor.index, columns=factor.columns)
        stock_return[mask.isna()] = np.nan
        factor[mask.isna()] = np.nan

        # Attach attributes to the class
        self.factor = factor
        self.stock_return = stock_return 

    # Coverage
    def getCoverage(self):
        
        coverage = self.factor.count(axis=1) / self.stock_return.count(axis=1)
        figFunc.plotCoverage(self.figFormat, coverage, self.factorPath)
        
        return coverage

    #%% serial correlation
    def getSerialCorr(self):
        
        serial_corr = self.factor.corrwith(self.factor.shift(1),axis=1)
        figFunc.plotSerialCorr(self.figFormat, serial_corr, self.factorPath)
        
        return serial_corr
    
    #%% Rank IC
    def getRankICStat(self):    
        
        factor = self.factor       
        rtn = self.stock_return
        rankIC, rankICStat = func.getRankICStat(factor, rtn)

        figFunc.plotICTimeSeries(self.figFormat, rankIC, 'RankIC', self.factorPath)
        figFunc.plotICDist(self.figFormat, rankIC, 'RankIC', self.factorPath)
        
        return rankIC,rankICStat
    
    #%% Rank IC Decay
    def getRankICDecay(self):

        factor_rank = self.factor.rank(axis=1)
        rtn_rank = self.stock_return.rank(axis=1)
        
        delayNum = 12
        size = factor.shape
        rankIC = np.zeros((size[0], delayNum))
        for i in range(delayNum):            
            rankIC_temp = factor_rank.corrwith(rtn_rank.shift(-i), axis=1)
            rankIC[:, i] = rankIC_temp
        
        rankIC = pd.DataFrame(rankIC)
        rankIC.index = factor.index
        
        rankICStat = func.getDataStat(rankIC)
        avgRankICDecay = rankICStat.loc['mean', :]
        avgRankICDecay = pd.DataFrame(avgRankICDecay)
        avgRankICDecay.columns = ['rankICDecay']
        figFunc.plotICDecay(self.figFormat, avgRankICDecay, 'rankIC', self.factorPath)
        
        return rankIC, rankICStat
    
    # 截面线性回归检验
    def getCSRegStat(self):
        
        factor = self.factor           
        rtn = self.stock_return
        
        beta, t, regStat = func.getCSRegStat(factor, rtn)   
        tStat = dp.getDataStat(abs(t))
        
        figFunc.plotTTimeSeries(self.figFormat, abs(t['t1']), 'Abs(t)', self.factorPath)
        figFunc.plotAdjR2TimeSeries(self.figFormat, regStat['adj_R2'], 'adj_R2', self.factorPath)
        
        return beta, t , regStat, tStat    
    
    # 分组回测
    def groupBacktest(self):
        groupNum = self.backtestParm.groupNum
        
        factor_temp = self.factor # 不可交易股票不参与选股
        group = func.grouping(factor_temp, groupNum) 

        # 计算factor turnover
        turnover = func.calTurnover(group)
        turnover = pd.DataFrame(turnover, index=factor_temp.index[1:], columns=np.arange(groupNum) + 1)
        # turnover 画图  (一般factor turnover是指long short组的turnover相加
        figFunc.plotTurnoverTimeSeries(self.figFormat, turnover.iloc[:,0]+turnover.iloc[:,-1], 
                                       'Factor Turnover', self.factorPath)

        # 计算factor分层收益
        groupRtn = func.calc_layer_returns(self.factor, self.stock_return, self.shift_num, groupNum)
        groupRtn = groupRtn.fillna(0)
        
        # 增加交易手续费
        # (买千2的滑点，卖千2的滑点，印花税千1，手续费万2暂不考虑，所以总成本按照千5计算)
        groupRtn.loc[turnover.index, :] = groupRtn.loc[turnover.index, :] - turnover * self.cost

        groupNav, groupDD, groupNavStat = func.getNavStat(groupRtn)
        
        # 加入turnover
        turnover_mean = turnover.mean()
        turnover_mean.name = 'Turnover'
        groupNavStat = groupNavStat.append(turnover_mean)

        # 分组作图
        figFunc.plotDataFramePlot(self.figFormat, groupNav, 'NetAssetValue', self.factorPath)
        figFunc.plotDataFramePlot(self.figFormat, groupDD, 'DrawDown', self.factorPath)

        groupPerf = groupNavStat.copy()
        groupPerf = groupPerf.T
        titles = groupPerf.columns.tolist()

        for title in titles:
            if title in ['startDate', 'endDate']:
                continue
            else:
                temp = groupPerf[title]
                temp = pd.DataFrame(temp)
                temp.index = groupPerf.index
                temp.columns = [title]
                figFunc.plotDataFrameGroup(self.figFormat, temp, title, self.factorPath)

        # 合并输出
        results = BacktestResults(groupRtn,groupNav,groupDD,groupNavStat)
        
        return results
    
    #%% 超额收益
    def alpha(self, groupRtn, hedgeRtnFile='CSI500.csv'):
        '''
        这里的alpha是指long组减去benchmark（默认CSI500）的portfolio
        '''
        hedgeRtn = pd.read_csv(os.path.join(self.dataPath, hedgeRtnFile), index_col=0, parse_dates=True)

        groupNum = self.backtestParm.groupNum
        portRtn = groupRtn[groupNum] # 仅仅计算最后一组的超额收益

        df_merge = pd.merge(portRtn, hedgeRtn, how='left', left_index=True, right_index=True)
        df_merge['alpha'] = df_merge.iloc[:, 0] - df_merge.iloc[:, -1]
        df_merge.replace(np.nan, 0, inplace=True)

        excessRtn = pd.DataFrame(df_merge['alpha'].copy())
        excessNav, excessDD, excessNavStat = func.getNavStat(excessRtn)
        
        figFunc.plotDataFramePlot(self.figFormat, excessNav, 'HedgeBenchmark nav', self.factorPath)
        figFunc.plotDataFramePlot(self.figFormat, excessDD, 'HedgeBenchmark Drawdown', self.factorPath)
        
        # 合并输出
        results = BacktestResults(excessRtn,excessNav,excessDD,excessNavStat)      
        
        return results
    
    #%% 多空收益
    def longShort(self,groupRtn):
              
        groupNum = self.backtestParm.groupNum
        longShortRtn = groupRtn[groupNum] - groupRtn[1]
        longShortRtn = pd.DataFrame(longShortRtn)
        longShortRtn.index = groupRtn.index
        longShortRtn.columns = ['longShort']
        longShortNav, longShortDD, longShortNavStat = func.getNavStat(longShortRtn)
        
        figFunc.plotDataFramePlot(self.figFormat, longShortNav, 'LongShort nav', self.factorPath)
        figFunc.plotDataFramePlot(self.figFormat, longShortDD, 'LongShort Drawdown', self.factorPath)
        
        # 合并输出
        results = BacktestResults(longShortRtn,longShortNav,longShortDD,longShortNavStat) 
        
        return results


def main(factorFile, factorDirection, startDate, endData, groupNum, cost, mask, shift_num):
    
    # init file path
    DataPath = os.path.join(os.path.abspath('./'), 'market_data')
    factorName = factorFile[:-4]
    factorPath = os.path.join(os.path.abspath('./'), factorName)
    if not os.path.isdir(factorPath):
        os.makedirs(factorPath)

    stgy_parm = BacktestParm(groupNum, startDate, endDate, factorDirection)

    figSize = [12, 8]
    fontSize = {'title': 24, 'label': 16, 'ticks': 12, 'legend': 12}
    color = {'type': {'major': 'c', 'minor': 'b'}, 'alpha': {'major': 0.5, 'minor': 1}}
    fig_format = FigFormat(figSize, fontSize, color)

    stgy = Backtest(DataPath, factorPath, factorFile, stgy_parm, fig_format, cost, mask, shift_num)

    coverage = stgy.getCoverage()
    serialCorr = stgy.getSerialCorr()
    rankIC, rankICStat = stgy.getRankICStat()
    rankIC_D, rankICStat_D = stgy.getRankICDecay()

    # need to change factor direction if RankIC is negative
    if (np.sign(rankICStat.loc['mean'].values[0]) == -1) and (factorDirection == 'ascending'):
        return 'RankIC Negative'

    # linear Regression
    beta, t, regStat, tStat = stgy.getCSRegStat()

    # group backtest
    groupResults = stgy.groupBacktest()
    groupRtn = groupResults.groupRtn
    groupNav = groupResults.groupNav
    alphaResults = stgy.alpha(groupRtn)
    longShortResults = stgy.longShort(groupRtn)

    # Performance Output
    decimal = 4

    df_perf = pd.DataFrame({
        # info
        'updated_time': [datetime.datetime.now().strftime('%Y-%m-%d')],
        'direction': [factorDirection],
        'shift_num': [shift_num],
        'group_num': [groupNum],
        'start_date': [startDate],
        'end_date': [endDate],
        'coverage_mean': [round(coverage.mean(), decimal)],
        'serial_corr_mean': [round(serialCorr.mean(), decimal)],

        # IC
        'RankIC_mean': [round(rankICStat.loc['mean'].values[0], decimal)],
        'RankIC_std': [round(rankICStat.loc['std'].values[0], decimal)],
        'RankIC_mean/std': [round(rankICStat.loc['mean'].values[0] / rankICStat.loc['std'].values[0], decimal)],
        'RankIC_min': [round(rankIC.min().values[0], decimal)],
        'RankIC_decay0_mean': [round(rankICStat_D.loc['mean', 0], decimal)],
        'RankIC_decay1_mean': [round(rankICStat_D.loc['mean', 1], decimal)],
        'RankIC_decay2_mean': [round(rankICStat_D.loc['mean', 2], decimal)],
        'RankIC_decay3_mean': [round(rankICStat_D.loc['mean', 3], decimal)],

        # Reg
        'reg_abs(t)_mean': [round(tStat.loc['mean', 't1'], decimal)],
        'reg_R2_mean': [round(regStat['adj_R2'].mean(), decimal)],

        # Group
        'group_LongOnly_Annual_rtn': [round(groupResults.groupNavStat.loc['Annualized Rtn', groupNum], decimal)],
        'group_LongOnly_Annual_std': [round(groupResults.groupNavStat.loc['Annualized Vol', groupNum], decimal)],
        'group_LongOnly_SR': [round(groupResults.groupNavStat.loc['Sharpe Ratio', groupNum], decimal)],
        'group_LongOnly_MDD': [round(groupResults.groupNavStat.loc['maxDD', groupNum], decimal)],

        'group_LongHedge_Annual_rtn': [round(alphaResults.groupNavStat.loc['Annualized Rtn', 'alpha'], decimal)],
        'group_LongHedge_Annual_std': [round(alphaResults.groupNavStat.loc['Annualized Vol', 'alpha'], decimal)],
        'group_LongHedge_SR': [round(alphaResults.groupNavStat.loc['Sharpe Ratio', 'alpha'], decimal)],
        'group_LongHedge_MDD': [round(alphaResults.groupNavStat.loc['maxDD', 'alpha'], decimal)],

        'group_LongShort_Annual_rtn': [
            round(longShortResults.groupNavStat.loc['Annualized Rtn', 'longShort'], decimal)],
        'group_LongShort_Annual_std': [
            round(longShortResults.groupNavStat.loc['Annualized Vol', 'longShort'], decimal)],
        'group_LongShort_SR': [round(longShortResults.groupNavStat.loc['Sharpe Ratio', 'longShort'], decimal)],
        'group_LongShort_MDD': [round(longShortResults.groupNavStat.loc['maxDD', 'longShort'], decimal)],

        'turnover_long_mean': [round(groupResults.groupNavStat.loc['Turnover', groupNum], decimal)],
        'turnover_short_mean': [round(groupResults.groupNavStat.loc['Turnover', 1], decimal)],
        'turnover_others_mean': [round(groupResults.groupNavStat.loc['Turnover'].iloc[1:-1].mean(), decimal)],

    }, index=[factorName])

    df_perf.to_csv(os.path.join(factorPath, 'Performance.csv'))

    return True


def main_final_result_backtest(weight, rtn, HedgeBenchmark, cost, outputPath):

    figSize = [12, 8]
    fontSize = {'title': 24, 'label': 16, 'ticks': 12, 'legend': 12}
    color = {'type': {'major': 'c', 'minor': 'b'}, 'alpha': {'major': 0.5, 'minor': 1}}
    figFormat = FigFormat(figSize, fontSize, color)

    portfolio_rtn_df = weight * rtn
    turnover = abs(weight.diff())
    portfolio_rtn = portfolio_rtn_df.sum(axis=1) - (turnover * cost / 2).sum(axis=1)
    portfolio_rtn = pd.DataFrame(portfolio_rtn)
    portfolio_rtn.columns = ['portfolio']

    HedgeBenchmark = HedgeBenchmark.fillna(0)
    portfolio_rtn[HedgeBenchmark.columns.tolist()] = HedgeBenchmark[HedgeBenchmark.columns.tolist()]
    portfolio_rtn['alpha'] = portfolio_rtn['portfolio'] - portfolio_rtn[HedgeBenchmark.columns.tolist()[0]]

    portfolio, dd, NavStat = func.getNavStat(portfolio_rtn)
    figFunc.plotDataFramePlot(figFormat, portfolio, 'NetAssetValue', outputPath)
    figFunc.plotDataFramePlot(figFormat, dd, 'DrawDown', outputPath)

    # Performance Output
    decimal = 4
    df_perf = pd.DataFrame({
        'updated_time': [datetime.datetime.now().strftime('%Y-%m-%d')],
        'portfolio_rtn': [round(NavStat.loc['Annualized Rtn', 'portfolio'], decimal)],
        'portfolio_maxDD': [round(NavStat.loc['maxDD', 'portfolio'], decimal)],
        'alpha_rtn': [round(NavStat.loc['Annualized Rtn', 'alpha'], decimal)],
        'alpha_maxDD': [round(NavStat.loc['maxDD', 'alpha'], decimal)],
        'HedgeBenchmark_rtn': [round(NavStat.loc['Annualized Rtn', HedgeBenchmark.columns.tolist()[0]], decimal)],
        'HedgeBenchmark_maxDD': [round(NavStat.loc['maxDD', HedgeBenchmark.columns.tolist()[0]], decimal)],
    }, index=['portfolio'])

    df_perf.to_csv(os.path.join(outputPath, 'Performance.csv'))

    return True


#%% module测试
if __name__ == '__main__':

    factorFile = 'revs60.csv'
    mask = pd.read_csv('mask.csv', index_col=0, parse_dates=True)
    main(factorFile, factorDirection, '2016-01-04', '2019-12-31', 10, 'ascending', 0.0003, mask, 1)
