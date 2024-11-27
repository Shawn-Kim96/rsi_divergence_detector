import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class DivergenceDetector:
    
    @staticmethod
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

        # Create arrays for fast indexing
        price_peaks_set = set(price_peaks)
        price_troughs_set = set(price_troughs)
        rsi_peaks_set = set(rsi_peaks)
        rsi_troughs_set = set(rsi_troughs)

        # Convert indices to arrays for vectorized operations
        price_troughs = np.array(price_troughs)
        price_peaks = np.array(price_peaks)
        rsi_troughs = np.array(rsi_troughs)
        rsi_peaks = np.array(rsi_peaks)


    # Function to find divergences
        def detect_divergence(idx2_list, price_points, rsi_points, divergence_type, rsi_threshold_check, price_condition, rsi_condition):
            for idx2 in idx2_list:
                # Define window boundaries
                window_start = max(0, idx2 - max_bars_lookback)
                window_end = idx2 - min_bars_lookback + 1

                if window_end <= window_start:
                    continue  # Skip if window is invalid

                # Get candidate indices within the window
                idx1_candidates = price_points[(price_points >= window_start) & (price_points < window_end)]

                # Skip if no candidates
                if len(idx1_candidates) == 0:
                    continue

                # Price and RSI at idx2
                price_idx2 = price[idx2]
                rsi_idx2 = rsi[idx2]

                for idx1 in idx1_candidates:
                    price_idx1 = price[idx1]
                    rsi_idx1 = rsi[idx1]

                    # Check price condition (higher high or lower low)
                    if price_condition(price_idx2, price_idx1):
                        # Check if RSI points correspond to price points
                        if idx1 in rsi_points and idx2 in rsi_points:
                            # Check RSI condition (lower high or higher low)
                            if rsi_condition(rsi_idx2, rsi_idx1):
                                # Check RSI thresholds
                                if rsi_threshold_check(rsi_idx1, rsi_idx2):
                                    # Check RSI between idx1 and idx2
                                    rsi_between = rsi[idx1+1:idx2]
                                    if len(rsi_between) == 0:
                                        continue
                                    if divergence_type == 'Bullish Divergence':
                                        if np.all(rsi_between >= min(rsi_idx1, rsi_idx2)):
                                            # Record divergence
                                            divergences.append({
                                                'start_datetime': df.index[idx1],
                                                'end_datetime': df.index[idx2],
                                                'divergence': divergence_type,
                                                'price_change': df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan,
                                                'rsi_change': df['rsi'].iloc[idx2 + min_bars_lookback] - rsi_idx2 if idx2 + min_bars_lookback < len(df) else np.nan,
                                                'future_return': (df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]) / df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan
                                            })
                                            break  # Break after finding the first valid divergence
                                    elif divergence_type == 'Bearish Divergence':
                                        if np.all(rsi_between <= max(rsi_idx1, rsi_idx2)):
                                            # Record divergence
                                            divergences.append({
                                                'start_datetime': df.index[idx1],
                                                'end_datetime': df.index[idx2],
                                                'divergence': divergence_type,
                                                'price_change': df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan,
                                                'rsi_change': df['rsi'].iloc[idx2 + min_bars_lookback] - rsi_idx2 if idx2 + min_bars_lookback < len(df) else np.nan,
                                                'future_return': (df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]) / df['close'].iloc[idx2] if idx2 + min_bars_lookback < len(df) else np.nan
                                            })
                                            break  # Break after finding the first valid divergence

        # Bullish Divergence Detection
        detect_divergence(
            idx2_list=price_troughs,
            price_points=price_troughs,
            rsi_points=rsi_troughs_set,
            divergence_type='Bullish Divergence',
            rsi_threshold_check=lambda a, b: (a <= bullish_rsi_threshold and b <= bullish_rsi_threshold),
            price_condition=lambda a, b: a < b,  # Price makes lower lows
            rsi_condition=lambda a, b: a > b     # RSI makes higher lows
        )

        # Bearish Divergence Detection
        detect_divergence(
            idx2_list=price_peaks,
            price_points=price_peaks,
            rsi_points=rsi_peaks_set,
            divergence_type='Bearish Divergence',
            rsi_threshold_check=lambda a, b: (a >= bearish_rsi_threshold and b >= bearish_rsi_threshold),
            price_condition=lambda a, b: a > b,  # Price makes higher highs
            rsi_condition=lambda a, b: a < b     # RSI makes lower highs
        )

        divergence_df = pd.DataFrame(divergences)
        if not divergence_df.empty:
            divergence_df.set_index('end_datetime', inplace=True)
        else:
            divergence_df = pd.DataFrame(columns=['divergence', 'price_change', 'rsi_change', 'future_return'])
        return divergence_df