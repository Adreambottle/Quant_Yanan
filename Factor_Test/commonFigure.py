## 作图功能
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import seaborn as sns
from scipy import stats
import statsmodels.api as sma


def plotCoverage(figFormat, coverage, figurePath):

    # calculate moving average of coverage
    coverage_12w_ma = coverage.rolling(12).mean()

    # plot
    plt.figure(figsize=figFormat.figSize)
    plt.bar(coverage.index, coverage.values,
            width=5,
            color=figFormat.color['type']['major'],
            alpha=figFormat.color['alpha']['major'],
            label='Coverage')
    plt.plot(coverage_12w_ma,
             color=figFormat.color['type']['minor'],
             alpha=figFormat.color['alpha']['minor'],
             label='3 month moving average')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 设置时间显示格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))
    plt.title('Coverage of Factor', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.ylabel('# of stocks', size=figFormat.fontSize['label'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, 'Coverage'))
    # plt.show()
    plt.close()


def plotSerialCorr(figFormat, serial_corr, figurePath):

    # calculate moving average of series correlation
    serial_corr_12w_ma = serial_corr.rolling(12).mean()

    # plot
    plt.figure(figsize=figFormat.figSize)
    plt.bar(serial_corr.index, serial_corr.values,
            width=5,
            color=figFormat.color['type']['major'],
            alpha=figFormat.color['alpha']['major'],
            label='correlation')
    plt.plot(serial_corr_12w_ma,
             color=figFormat.color['type']['minor'],
             alpha=figFormat.color['alpha']['minor'],
             label='3 month moving average')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 设置时间显示格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))
    plt.title('Factor Score Serial Correlation', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.ylabel('correlation', size=figFormat.fontSize['label'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, 'Factor Serial Correlation'))
    # plt.show()
    plt.close()


def plotICTimeSeries(figFormat, IC, title, figurePath):

    # calculate moving averagae of IC
    IC_12w_ma = IC.rolling(12).mean()

    # plot
    plt.figure(figsize=figFormat.figSize)
    plt.bar(IC.index, IC.values.flatten(),                     #因为IC是df所以需要flatten
            width=8,
            color=figFormat.color['type']['major'],
            alpha=figFormat.color['alpha']['major'],
            label=title)
    plt.plot(IC_12w_ma,
             color=figFormat.color['type']['minor'],
             alpha=figFormat.color['alpha']['minor'],
             label='3 month moving average')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 设置时间显示格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))
    plt.title(title+' Time Series', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.ylabel(title, size=figFormat.fontSize['label'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, title+' TimeSeries'))
    # plt.show()
    plt.close()


def plotICDist(figFormat, IC, title, figurePath):

    num = IC.shape[0]
    bins_num = max(math.floor(num / 10), 10)
    bins_num = min(bins_num, 25)

    plt.figure(figsize=figFormat.figSize)
    sns.distplot(IC,color="c",bins=bins_num,kde=True)

    text1 = '$\mu$ = ' + str(round(IC.mean()[0], 3))
    text2 = '$\sigma$ = ' + str(round(IC.std()[0], 3))
    text3 = '$IR$ = ' + str(round(IC.mean()[0]/IC.std()[0]))
    text4 = '$pval$ = ' + str(round(stats.normaltest(IC)[1][0], 5))
    text5 = '$prob(IC>0)$ = ' + str(round(np.sum(IC>0)[0]/len(IC), 3))

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .1, text1)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .15, text2)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .20, text3)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .25, text4)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .30, text5)

    plt.title(title + ' Distribution', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.xlabel(title + ' score', size=figFormat.fontSize['label'])
    plt.ylabel('# of ' + title + ' Score per Interval', size=figFormat.fontSize['label'])
    plt.savefig(os.path.join(figurePath, title+' Dist'))
    # plt.show()
    plt.close()


def plotICDecay(figFormat, avgICDecay, title, figurePath):

    avgICDecay.plot.bar(figsize=figFormat.figSize, width=0.7, color=figFormat.color['type']['major'])
    plt.title(title + ' Decay', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'], rotation=0)
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.xlabel('N', size=figFormat.fontSize['label'])
    plt.ylabel(title + ' score', size=figFormat.fontSize['label'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, title+' Decay'))
    # plt.show()
    plt.close()


def plotTTimeSeries(figFormat, t, title, figurePath):

    # calculate moving averagae of t-value
    t_12w_mm = t.rolling(12).median()

    # plot
    plt.figure(figsize=figFormat.figSize)
    plt.bar(t.index, t.values.flatten(),                     #因为IC是df所以需要flatten
            width=8,
            color=figFormat.color['type']['major'],
            alpha=figFormat.color['alpha']['major'],
            label=title)
    plt.plot(t_12w_mm,
             color=figFormat.color['type']['minor'],
             alpha=figFormat.color['alpha']['minor'],
             label='3 month moving median')

    subline = t.copy()
    subline.iloc[:] = 2
    plt.plot(subline, c='y', linestyle='--', label='threshold: 2')        #添加辅助线

    text1 = '$\mu$ = ' + str(round(t.mean(), 2))
    text2 = '$\sigma$ = ' + str(round(t.std(), 2))
    text4 = '$prob(|t|>2)$ = ' + str(round(np.sum(t>2)/len(t), 3))

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .2, text1)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .26, text2)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .32, text4)

    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 设置时间显示格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))
    plt.title(title+' Time Series', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, title+' TimeSeries'))
    # plt.show()
    plt.close()


def plotAdjR2TimeSeries(figFormat, Adj_R2, title, figurePath):

    # calculate moving averagae of t-value
    Adj_R2_12w_ma = Adj_R2.rolling(12).mean()

    # plot
    plt.figure(figsize=figFormat.figSize)
    plt.bar(Adj_R2.index, Adj_R2.values.flatten(),  # 因为IC是df所以需要flatten
            width=8,
            color=figFormat.color['type']['major'],
            alpha=figFormat.color['alpha']['major'],
            label=title)
    plt.plot(Adj_R2_12w_ma,
             color=figFormat.color['type']['minor'],
             alpha=figFormat.color['alpha']['minor'],
             label='3 month moving median')

    subline = Adj_R2.copy()
    subline.iloc[:] = Adj_R2.mean()
    plt.plot(subline, c='green', linestyle = '--', label='Mean')  # 添加辅助线

    text1 = '$\mu$ = ' + str(round(Adj_R2.mean(), 3))
    text2 = '$\sigma$ = ' + str(round(Adj_R2.std(), 3))

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .2, text1)
    ax.text(xmin + (xmax - xmin) * .05, ymax - (ymax - ymin) * .26, text2)

    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 设置时间显示格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))
    plt.title(title + ' Time Series', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, title + ' TimeSeries'))
    # plt.show()
    plt.close()


def plotTurnoverTimeSeries(figFormat, turnover, title, figurePath):

    # calculate moving average of series correlation
    turnover_12w_ma = turnover.rolling(12).mean()

    # plot
    plt.figure(figsize=figFormat.figSize)
    plt.bar(turnover.index, turnover.values,
            width=5,
            color=figFormat.color['type']['major'],
            alpha=figFormat.color['alpha']['major'],
            label='correlation')
    plt.plot(turnover_12w_ma,
             color=figFormat.color['type']['minor'],
             alpha=figFormat.color['alpha']['minor'],
             label='3 month moving average')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 设置时间显示格式
    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))
    plt.title('Factor Turnover', size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, 'Factor Turnover'))
    # plt.show()
    plt.close()


def plotDataFramePlot(figFormat, df, title, figurePath):

    if df.shape[1] == 1:
        df.plot(figsize=figFormat.figSize, color=figFormat.color['type']['major'])
    else:
        df.plot(figsize=figFormat.figSize)
    plt.title(title, size=figFormat.fontSize['title'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'])
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, title))
    # plt.show()
    plt.close()


def plotDataFrameGroup(figFormat, df, title, figurePath):

    df.plot.bar(figsize=figFormat.figSize, color=figFormat.color['type']['major'])
    plt.title(title, size=figFormat.fontSize['title'])
    plt.xlabel('Group',size=figFormat.fontSize['label'])
    plt.xticks(fontsize=figFormat.fontSize['ticks'], rotation=0)
    plt.yticks(fontsize=figFormat.fontSize['ticks'])
    plt.legend(prop={'size': figFormat.fontSize['legend']})
    plt.savefig(os.path.join(figurePath, title))
    # plt.show()
    plt.close()

