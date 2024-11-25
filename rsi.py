import ccxt
import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Data Fetching Function
def fetch_ohlcv(exchange, symbol, timeframe, since=None, limit=1000):
    all_data = []
    since = exchange.parse8601(since) if since else None

    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not data:
            break
        all_data.extend(data)
        since = data[-1][0] + 1  # Fetch data after the last timestamp
        if len(data) < limit:
            break
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]  # Remove duplicate indices
    return df

# 2. RSI Calculation Function
def calculate_rsi(df, period=14):
    df = df.copy()
    df['rsi'] = talib.RSI(df['close'], timeperiod=period)
    return df

# 3. Divergence Detection Function (Enhanced)
def find_divergences(df, rsi_period=14, min_bars_lookback=5, max_bars_lookback=180,
                     bullish_rsi_threshold=30, bearish_rsi_threshold=70, price_prominence=1, rsi_prominence=1):
    df = df.copy()
    price = df['close'].values
    rsi = df['rsi'].values
    divergences = []

    # Find peaks and troughs in price and RSI
    price_peaks, _ = find_peaks(price, prominence=price_prominence)
    price_troughs, _ = find_peaks(-price, prominence=price_prominence)
    rsi_peaks, _ = find_peaks(rsi, prominence=rsi_prominence)
    rsi_troughs, _ = find_peaks(-rsi, prominence=rsi_prominence)

    # Convert indices to sets for faster lookup
    rsi_peaks_set = set(rsi_peaks)
    rsi_troughs_set = set(rsi_troughs)

    # Bullish Divergence Detection
    for idx2 in price_troughs:
        # Find previous troughs within the lookback range
        idx1_candidates = price_troughs[(price_troughs < idx2) & (idx2 - price_troughs >= min_bars_lookback) & (idx2 - price_troughs <= max_bars_lookback)]
        for idx1 in idx1_candidates:
            # Price makes lower lows
            if price[idx2] < price[idx1]:
                # Ensure RSI troughs correspond to price troughs
                if idx1 in rsi_troughs_set and idx2 in rsi_troughs_set:
                    rsi_idx1 = idx1
                    rsi_idx2 = idx2
                    # RSI makes higher lows
                    if rsi[rsi_idx2] > rsi[rsi_idx1]:
                        # Both RSI values strictly below threshold (30)
                        if rsi[rsi_idx1] <= bullish_rsi_threshold and rsi[rsi_idx2] <= bullish_rsi_threshold:
                            # Check that RSI values between idx1 and idx2 are not lower than a or b
                            rsi_between = rsi[rsi_idx1+1:rsi_idx2]
                            if np.all(rsi_between >= min(rsi[rsi_idx1], rsi[rsi_idx2])):
                                divergence_type = 'Bullish Divergence'
                                divergences.append({
                                    'start_datetime': df.index[idx1],
                                    'end_datetime': df.index[idx2],
                                    'divergence': divergence_type,
                                    'price_change': df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'rsi_change': df['rsi'].iloc[idx2 + min_bars_lookback] - df['rsi'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'future_return': (df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]) / df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan
                                })
                                break  # Stop after finding the first valid divergence

    # Bearish Divergence Detection
    for idx2 in price_peaks:
        # Find previous peaks within the lookback range
        idx1_candidates = price_peaks[(price_peaks < idx2) & (idx2 - price_peaks >= min_bars_lookback) & (idx2 - price_peaks <= max_bars_lookback)]
        for idx1 in idx1_candidates:
            # Price makes higher highs
            if price[idx2] > price[idx1]:
                # Ensure RSI peaks correspond to price peaks
                if idx1 in rsi_peaks_set and idx2 in rsi_peaks_set:
                    rsi_idx1 = idx1
                    rsi_idx2 = idx2
                    # RSI makes lower highs
                    if rsi[rsi_idx2] < rsi[rsi_idx1]:
                        # Both RSI values strictly above threshold (70)
                        if rsi[rsi_idx1] >= bearish_rsi_threshold and rsi[rsi_idx2] >= bearish_rsi_threshold:
                            # Check that RSI values between idx1 and idx2 are not higher than a or b
                            rsi_between = rsi[rsi_idx1+1:rsi_idx2]
                            if np.all(rsi_between <= max(rsi[rsi_idx1], rsi[rsi_idx2])):
                                divergence_type = 'Bearish Divergence'
                                divergences.append({
                                    'start_datetime': df.index[idx1],
                                    'end_datetime': df.index[idx2],
                                    'divergence': divergence_type,
                                    'price_change': df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'rsi_change': df['rsi'].iloc[idx2 + min_bars_lookback] - df['rsi'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'future_return': (df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]) / df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan
                                })
                                break  # Stop after finding the first valid divergence

    divergence_df = pd.DataFrame(divergences)
    if not divergence_df.empty:
        divergence_df.set_index('end_datetime', inplace=True)
    else:
        divergence_df = pd.DataFrame(columns=['divergence', 'price_change', 'rsi_change', 'future_return'])
    return divergence_df



# 4. Visualization Function (Improved)
def plot_divergence(df, divergence_df, title='', save_fig=False):
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df['close'], label='Price')
    ax1.set_title(f'{title} Price')
    ax1.legend()

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df.index, df['rsi'], label='RSI', color='orange')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_title(f'{title} RSI')
    ax2.legend()

    for idx, row in divergence_df.iterrows():
        start_datetime = row['start_datetime']
        end_datetime = idx
        divergence_type = row['divergence']

        # Get the RSI values
        rsi_start = df.loc[start_datetime, 'rsi']
        rsi_end = df.loc[end_datetime, 'rsi']

        # Plot lines on RSI chart
        if divergence_type == 'Bullish Divergence':
            color = 'green'
            marker = '^'
            ax1.plot([start_datetime, end_datetime], [df.loc[start_datetime, 'close'], df.loc[end_datetime, 'close']], color=color, linewidth=2)
            ax1.scatter([end_datetime], [df.loc[end_datetime, 'close']], marker=marker, color=color, s=100)
            ax2.plot([start_datetime, end_datetime], [rsi_start, rsi_end], color=color, linewidth=2)
            ax2.scatter([end_datetime], [rsi_end], marker=marker, color=color, s=100)
        elif divergence_type == 'Bearish Divergence':
            color = 'red'
            marker = 'v'
            ax1.plot([start_datetime, end_datetime], [df.loc[start_datetime, 'close'], df.loc[end_datetime, 'close']], color=color, linewidth=2)
            ax1.scatter([end_datetime], [df.loc[end_datetime, 'close']], marker=marker, color=color, s=100)
            ax2.plot([start_datetime, end_datetime], [rsi_start, rsi_end], color=color, linewidth=2)
            ax2.scatter([end_datetime], [rsi_end], marker=marker, color=color, s=100)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{title.replace('/', '_')}.png")
    plt.close()

# 5. Fibonacci Retracement Levels Calculation Function (Unchanged)
def calculate_fibonacci_levels(df, lookback=20):
    df = df.copy()
    df['max_price'] = df['high'].rolling(window=lookback).max()
    df['min_price'] = df['low'].rolling(window=lookback).min()
    df['fib_23.6'] = df['max_price'] - 0.236 * (df['max_price'] - df['min_price'])
    df['fib_38.2'] = df['max_price'] - 0.382 * (df['max_price'] - df['min_price'])
    df['fib_50.0'] = df['max_price'] - 0.500 * (df['max_price'] - df['min_price'])
    df['fib_61.8'] = df['max_price'] - 0.618 * (df['max_price'] - df['min_price'])
    df['fib_78.6'] = df['max_price'] - 0.786 * (df['max_price'] - df['min_price'])
    return df

# 6. Main Execution Function
def main():
    # Customizable parameters
    rsi_period = 14  # RSI length
    min_bars_lookback = 5  # Adjusted minimum bars to check
    max_bars_lookback = 180  # Adjusted maximum bars to check
    bullish_rsi_threshold = 30  # Slightly increased RSI threshold for bullish divergence
    bearish_rsi_threshold = 70  # Slightly decreased RSI threshold for bearish divergence
    price_prominence = 1  # Prominence for price peaks/troughs
    rsi_prominence = 1  # Prominence for RSI peaks/troughs

    divergence_data = pd.DataFrame()

    timeframe = '15m'
    df = pd.read_csv("15min.csv")
    df = calculate_rsi(df, period=rsi_period)
    df = calculate_fibonacci_levels(df)

    window = 5
    divergence_df = find_divergences(
        df,
        rsi_period=rsi_period,
        min_bars_lookback=min_bars_lookback,
        max_bars_lookback=max_bars_lookback,
        bullish_rsi_threshold=bullish_rsi_threshold,
        bearish_rsi_threshold=bearish_rsi_threshold,
        price_prominence=price_prominence,
        rsi_prominence=rsi_prominence
    )
    divergence_df['timeframe'] = timeframe
    divergence_data = pd.concat([divergence_data, divergence_df])

    # Visualization (optional)
    plot_divergence(df, divergence_df, title=f'BTC_USDT_{timeframe}', save_fig=True)

    # Output divergence data
    print(divergence_data.head())

    # Save to CSV
    divergence_data.to_csv('divergence_data.csv')

    # Save the entire dataset for model training

if __name__ == '__main__':
    main()
