import pandas as pd, numpy as np
from src.config import RAW
from src.data.collect_prices import add_targets

POS = ['to the moon','great earnings beat','bullish breakout','strong buy','love this','rocket']
NEG = ['this will crash','dumping shares','bearish setup','overvalued garbage','selling now','dead money']
NEU = ['holding steady','sideways today','no change','watching closely']

def gen(mode, seed=0, n_days=500, tickers=('AAPL','TSLA')):
    """mode='planted': today's net sentiment shifts tomorrow's return.
       mode='random' : sentiment and returns are independent."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    prows, mrows, mid = [], [], 0
    for t in tickers:
        tilt = rng.uniform(-1, 1, size=n_days)   # day i's sentiment tilt in [-1,1]
        # Build the close-price path FIRST so day i's tilt drives the return
        # realized between close[i] and close[i+1] (which is exactly next_ret[i]).
        closes = [100.0]
        for i in range(n_days):
            if mode == 'planted':
                ret = rng.normal(0.012 * tilt[i], 0.015)  # tilt[i] -> next_ret[i]
            else:
                ret = rng.normal(0, 0.02)                 # no signal
            closes.append(closes[-1] * (1 + ret))
        # closes has n_days+1 entries; use closes[i] as day i's close
        for i, d in enumerate(dates):
            n_msg = rng.integers(5, 15)
            for _ in range(n_msg):
                mid += 1
                if mode == 'planted':
                    p = (tilt[i] + 1) / 2  # higher tilt -> more positive msgs
                    pool = POS if rng.random() < p else NEG
                else:
                    pool = [POS, NEG, NEU][rng.choice(3, p=[0.4,0.4,0.2])]
                mrows.append({'id':mid,'ticker':t,
                              'created_at':pd.Timestamp(d, tz='UTC'),
                              'body':rng.choice(pool),'source':'synthetic'})
            price = closes[i]
            prows.append({'date':d,'ticker':t,'open':price,'high':price,
                          'low':price,'close':price,'volume':int(rng.integers(1e6,5e6))})
    RAW.mkdir(parents=True, exist_ok=True)
    add_targets(pd.DataFrame(prows)).to_parquet(RAW/'prices.parquet', index=False)
    pd.DataFrame(mrows).to_parquet(RAW/'messages.parquet', index=False)
    return len(prows), len(mrows)

if __name__ == '__main__':
    import sys
    p, m = gen(sys.argv[1], seed=42)
    print(f"{sys.argv[1]}: {p} price rows, {m} messages")
