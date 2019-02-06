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
    #XRP = client.get_historical_klines(symbol='XRPBTC', interval=binance_constants.KLINE_INTERVAL_4HOUR, start_str="12 month ago UTC");
    # pickle.dump(XRP, open('data.pkl', 'wb'))
    XRP = pickle.load(open('data.pkl', 'rb'))
    open_values = list(map(lambda x: float(x[1]), XRP))
    high_values = list(map(lambda x: float(x[2]), XRP))
    volume_values = list(map(lambda x: float(x[5]), XRP))
    prices = list(map(lambda x: (100000 * (float(x[2]) + float(x[3]) + float(x[4]))/3.0), XRP))

    # import pdb
    # pdb.set_trace()

    #output = talib.SMA(np.array(open_values), timeperiod=100)
    rsi = talib.RSI(np.array(open_values), timeperiod=14)

    #plt.plot(output)
    plt.plot(open_values)
    plt.plot(prices)
    #plt.show()
    plt.plot(rsi)
    #plt.show()
    #print(len(XRP))
    return prices[100:], rsi[100:]


