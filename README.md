# Cryptocurrency-Strategy-Backtesting-Frame
Cryptocurrency Strategy + Backtesting Frame targeting on Binance kline data

Investment Target we choose: BTC/USDT, ETH/USDT, 

Time Frame we use: 1 hour, 1 day
  reason: We tried aggregation data on 1 min and 5 min, but there's no enough trades in each time peroid. We also tried aggregation data on 1 week, but there's no enough data points since Binance data starts after 2017 or even later years.  

Trading Strategy we use: 
  Type 1: {EMA(x),EMA(y)}, where x<y. When EMA(x) goes across EMA(y) from bottom, the model buys in, and when EMA(y) goes up across EMA(x) from bottom, the model sells out. We can tune parameters of this model, and by far the best sharp ratio is 1.2 (babytester.py file)

  Type 2: {RSI(x),H,L,a,b} where H > L. When RSI(x)> H, the model buys a% of cash, and when RSI(x)< H, the model sells b% of equity. We can tune parameters of this model, and by far the best sharp ratio is 1.2 (ETH_1d.py file)

Backtester:
  We build a list align with dataframe, called "signal", where each entry is among {-1,0,1}. When model reaches trigger "buys in", signal records 1, when model reaches trigger "sells out", signal records -1, and otherwise signal records 0. We then excute trading strategy based on signal and record {trade, current cash, current equity} in a list called "equity curve".

Visulization:
  It's explained in each strategy.py

Next setp on Trading Strategy:
  1: build up model based on Gradient-Boosted Decision Trees (GBDT) and kline data.
    related reference: https://c3.ai/glossary/data-science/gradient-boosted-decision-trees-gbdt/#:~:text=Gradient%2Dboosted%20decision%20trees%20are%20a%20popular%20method%20for%20solving,to%20a%20sufficiently%20optimal%20solution.
    
  2: build up model based on transformer and kline data + volume data.
    related reference: Kronos: A Foundation Model for the Language of Financial Markets. https://arxiv.org/abs/2508.02739
    This paper use methods of LLM models to tokenize kline data and train model to predict upcoming kline performance. We want to implement this idea on our binance kline data + volume data. The advantage of this method is that we can have a pre-trained model at first step, which provides a solid prototype. Since our crypto targets have history less than 10 years, we can therefore use DPO method to fine-tune Kronos prototype, and make it a good predictor on next several days' kline. 
