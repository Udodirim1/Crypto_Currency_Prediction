import pandas as pd

tickers = ('BTC', 'ETH', 'USDT','USDC', 'BNB',
                  'BUSD', 'XRP', 'ADA', 'SOL', 'DOGE',
                  'DAI', 'DOT', 'WTRX', 'HEX', 'TRX',
                  'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC',
                  'MATIC', 'STETH', 'UNI1', 'LTC', 'FTT',
                  'LINK', 'CRO', 'XLM', 'NEAR', 'ATOM')

dropdown = st.multiselect('Pick your assets', tickers)

start = st.date_input('Start', value=pd.to_datetime('2018-01-01'))
end = st.date_input('End', value=pd.to_datetime('today'))

def relativereturns(df):
    rel = df.pct_change()
    cumulativeret = (1+rel).cumprod() - 1
    cumulativeret = cumulativeret.fillna(0)
    return cumulativeret

if len(dropdown) > 0:
     df = yf.download(dropdown,start,end)['Close']
     df = relativereturns(yf.download(dropdown,start,end)['Close'])
     st.header({'Close Predict': prediction}.format(dropdown))
     st.header('Returns of {}'.format(dropdown))
     st.line_chart(df)