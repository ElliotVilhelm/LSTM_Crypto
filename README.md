# 🔥 LSTM 🔥

Predicting the price of crypto coins on Binance using an LSTM .. it's actually working

### Input
 -> TA (RSI, MACD, ETC .. )

 -> Current Coin Value
### Output
 -> Next Kline Coin Value

 Right now I am experimenting with sequence length 10-20 with Klines of 120 minutes.


uwu

## Thoughts Info Stuff
- Network is initialized using Xavier weight normalization
- Random sampling of data of window size sequence length (10-20)
- Input data must be normalized and in the case of low value currency such as XRP scaled.



#### Dependencies
```
brew install ta-lib
pip3 install TA-Lib
pip3 install pytorch
```
Probably more python stuff, .. yeah numpy ..


