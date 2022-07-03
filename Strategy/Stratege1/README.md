# Readme

### Summary
Support vector machine (SVM) is one of the most commonly used machine learning algorithms for quantitative analysis. Linear support vector machines can solve linear classification problems, while kernel support vector machines can effectively solve traditional regressions such as nonlinearity and overfitting. Difficulty of the model. Wavelet transform (WT) is currently an algorithm widely used in applied mathematics and engineering. It has achieved remarkable results in signal analysis and image recognition. In recent years, it has gradually begun to be applied in the filtering of stock price time series.

From an empirical point of view, this report will explore the impact of the six major categories of factors including macroeconomics, finance, profitability, market conditions, technology, and valuation on the Windtron A Index. The main part of the model construction includes the analysis of the correlation and cointegration between the factor and the return sequence in terms of factor screening; the optimization of the Wind A index through the noise reduction function of the wavelet transform, so that it can better reflect the main index of the index Trend; in terms of model training, a rolling support vector machine model that can capture the short-term rise and fall trends of the index sequence is used; in the evaluation of model indicators, the accuracy, AUC, and f1 are calculated for in-sample, out-of-sample, and cross-validation sets. Common indicators such as score.

Based on the predicted value of the Wonderfull A Index, construct the timing strategy of long, long and short, and grid trading strategies, and compare their similarities and differences through indicators such as annualized yield, Sharpe ratio, and maximum drawdown rate.

### Effect
The time period for the backtest is from January 1, 2010 to December 31, 2020 (250 trading days). The model's out-of-sample accuracy rate is 58.1%, and the annualized return rate of the long strategy based on the model's prediction value is 19.2% and the Sharpe ratio is 1.045, which is much higher than the benchmark yield curve of 6.2% and 0.256. After using the grid trading strategy to optimize the long strategy, we can reduce the maximum drawdown rate to 11.04% and further increase the Sharpe ratio. Observing the effect diagram of the grid trading strategy combined with the predicted value, the net value curve tends to be flat, and the predicted value can accurately grasp the rising and falling trend.

### Conclusion

In the factor screening process, after considering the lagging factors, most of the factors have cointegration and correlation with the Windtron A index return rate. The overall prediction accuracy of the model is relatively high, and the performance of the timing strategy based on the predicted value of the model in terms of return and risk control is far better than the index benchmark net value curve, indicating that the WT-SVM model has a good performance in the 2010-2020 backtest interval. Forecast effect. However, in evaluating the effect of wavelet noise reduction, whether to use wavelet analysis has little effect on the accuracy of the model, f1 score, AUC and other indicators, and only slightly improves the annualized rate of return and Sharpe ratio of the timing strategy, which shows that the wavelet transform Filtering has limited influence on the prediction effect of the model.