from binance.client import Client
from config import api_key, api_secret
import pickle
import binance_constants
import matplotlib.pyplot as plt
import numpy as np
import talib
"""
[
  [
    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "17928899.62484339" // Ignore.
  ]
]
"""


def get_data():
    client = Client(api_key, api_secret)
    XRP = client.get_historical_klines(symbol='XRPBTC', interval=binance_constants.KLINE_INTERVAL_4HOUR, start_str="12 month ago UTC");
    # pickle.dump(XRP, open('data.pkl', 'wb'))
    XRP = pickle.load(open('data.pkl', 'rb'))
    print("data size", len(XRP))
    open_values = list(map(lambda x: float(x[1]), XRP))
    # high_values = list(map(lambda x: float(x[2]), XRP))
    # volume_values = list(map(lambda x: float(x[5]), XRP))
    import pdb
    # pdb.set_trace()
    prices = list(map(lambda x: ((float(x[2]) + float(x[3]) + float(x[4]))/3.0) * 100000, XRP))
    targets = prices[21:]
    targets = list(map(lambda x: x, targets))
    # prices = list(map(lambda x: (x - min(prices))/(max(prices) - min(prices)), prices))
    prices = prices[20:-1]
    rsi = talib.RSI(np.array(open_values), timeperiod=14)
    rsi = rsi[20:-1]
    high = np.array(list(map(lambda x: float(x[2]), XRP)))
    low = np.array(list(map(lambda x: float(x[3]), XRP)))
    close = np.array(list(map(lambda x: float(x[4]), XRP)))
    real = talib.ATR(high, low, close, timeperiod=14)
    real = real[20:-1]
    # rsi = list(map(lambda x: (x - min(rsi))/(max(rsi) - min(rsi)), rsi))
    #plt.plot(output)
    # plt.plot(open_values)
    plt.plot(prices)
    #plt.show()
    plt.plot(rsi)
    # plt.show()
    #print(len(XRP))
    # import pdb
    # pdb.set_trace()
    return [prices, rsi, real], targets


