import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class DivergenceDetector:
    @staticmethod
    def find_previous_peak(df, divergence_start_idx, is_bullish, bullish_peak_rsi_threshold=55, bearish_peak_rsi_threshold=45):
        # Consider only data before the divergence start index
        df_before_divergence = df.iloc[:divergence_start_idx]
        if is_bullish:
            # Bullish: Find previous peaks with RSI >= bullish_peak_rsi_threshold
            peaks_idx, _ = find_peaks(df_before_divergence['close'].values)
            if len(peaks_idx) == 0:
                return None
            valid_peaks = df_before_divergence.iloc[peaks_idx]
            valid_peaks = valid_peaks[valid_peaks['rsi'] >= bullish_peak_rsi_threshold]
            if valid_peaks.empty:
                return None
            return valid_peaks.index[-1]  # Return the index of the last valid peak
        else:
            # Bearish: Find previous valleys with RSI <= bearish_peak_rsi_threshold
            valleys_idx, _ = find_peaks(-df_before_divergence['close'].values)
            if len(valleys_idx) == 0:
                return None
            valid_valleys = df_before_divergence.iloc[valleys_idx]
            valid_valleys = valid_valleys[valid_valleys['rsi'] <= bearish_peak_rsi_threshold]
            if valid_valleys.empty:
                return None
            return valid_valleys.index[-1]  # Return the index of the last valid valley


    @staticmethod
    def calculate_tp_sl(previous_idx, divergence_idx, df, is_bullish):
        high = df['close'].iloc[previous_idx]
        low = df['close'].iloc[divergence_idx]
        if is_bullish:
            # For bullish, TP is at the 0.386 Fibonacci level
            tp = high - (high - low) * 0.386
            sl = low  # SL is the low at the divergence point
        else:
            # For bearish, TP is at the 0.618 Fibonacci level
            tp = low + (high - low) * 0.618
            sl = high  # SL is the high at the divergence point
        return tp, sl


    def find_divergences(self, df, rsi_period=14, min_bars_lookback=5, max_bars_lookback=180,
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

        # Create sets for fast lookup
        price_peaks_set = set(price_peaks)
        price_troughs_set = set(price_troughs)
        rsi_peaks_set = set(rsi_peaks)
        rsi_troughs_set = set(rsi_troughs)

        # Convert indices to arrays for vectorized operations
        price_troughs = np.array(price_troughs)
        price_peaks = np.array(price_peaks)
        rsi_troughs = np.array(rsi_troughs)
        rsi_peaks = np.array(rsi_peaks)

        # Function to detect divergences
        def detect_divergence(idx2_list, price_points, rsi_points, divergence_type, rsi_threshold_check,
                              price_condition, rsi_condition, rsi_threshold, is_bullish):
            for idx2 in idx2_list:
                # Set window boundaries
                window_start = max(0, idx2 - max_bars_lookback)
                window_end = idx2 - min_bars_lookback + 1

                if window_end <= window_start:
                    continue  # Skip if window is invalid

                # Candidate indices within the window
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
                    cond1 = price_condition(price_idx2, price_idx1)  # Check if RSI points correspond to price points
                    cond2 = (idx1 in rsi_points) and (idx2 in rsi_points)  # Check RSI condition (lower high or higher low)
                    cond3 = rsi_condition(rsi_idx2, rsi_idx1)  # Check RSI thresholds
                    cond4 = rsi_threshold_check(rsi_idx1, rsi_idx2)   # Check RSI between idx1 and idx2 (noise filtering)

                    if cond1 and cond2 and cond3 and cond4:
                        rsi_between = rsi[idx1+1:idx2]

                        if len(rsi_between) == 0:
                            continue

                        if divergence_type == 'Bullish Divergence':
                            if np.all(rsi_between >= np.minimum(rsi_idx1, rsi_idx2)):
                                # Calculate TP and SL
                                previous_peak_idx = self.find_previous_peak(df, idx1, is_bullish=True,
                                                                            bullish_rsi_threshold=55,
                                                                            bearish_rsi_threshold=45)
                                if previous_peak_idx is None:
                                    continue
                                tp, sl = self.calculate_tp_sl(previous_peak_idx, idx2, df, is_bullish=True)

                                entry_idx = idx2 + 2
                                if entry_idx >= len(df):
                                    continue
                                entry_datetime = df['datetime'].iloc[entry_idx]
                                
                                # Record divergence
                                divergences.append({
                                    'start_datetime': df['datetime'].iloc[idx1],
                                    'end_datetime': df['datetime'].iloc[idx2],
                                    'entry_datetime': entry_datetime,
                                    'divergence': divergence_type,
                                    'price_change': df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]
                                    if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'rsi_change': df['rsi'].iloc[idx2 + min_bars_lookback] - rsi_idx2
                                    if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'future_return': (df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]) /
                                                        df['close'].iloc[idx2]
                                    if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'TP': tp,
                                    'SL': sl
                                })
                                break  # Break after finding the first valid divergence
                        
                        elif divergence_type == 'Bearish Divergence':
                            if np.all(rsi_between <= np.maximum(rsi_idx1, rsi_idx2)):
                                # Calculate TP and SL
                                previous_valley_idx = self.find_previous_peak(df, idx1, is_bullish=False,
                                                                                bullish_rsi_threshold=55,
                                                                                bearish_rsi_threshold=45)
                                if previous_valley_idx is None:
                                    continue
                                tp, sl = self.calculate_tp_sl(previous_valley_idx, idx2, df, is_bullish=False)
                                # Record divergence
                                divergences.append({
                                    'start_datetime': df['datetime'].iloc[idx1],
                                    'end_datetime': df['datetime'].iloc[idx2],
                                    'entry_datetime': entry_datetime,
                                    'divergence': divergence_type,
                                    'price_change': df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]
                                    if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'rsi_change': df['rsi'].iloc[idx2 + min_bars_lookback] - rsi_idx2
                                    if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'future_return': (df['close'].iloc[idx2 + min_bars_lookback] - df['close'].iloc[idx2]) /
                                                        df['close'].iloc[idx2]
                                    if idx2 + min_bars_lookback < len(df) else np.nan,
                                    'TP': tp,
                                    'SL': sl
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
            rsi_condition=lambda a, b: a > b,    # RSI makes higher lows
            rsi_threshold=55,
            is_bullish=True
        )

        # Bearish Divergence Detection
        detect_divergence(
            idx2_list=price_peaks,
            price_points=price_peaks,
            rsi_points=rsi_peaks_set,
            divergence_type='Bearish Divergence',
            rsi_threshold_check=lambda a, b: (a >= bearish_rsi_threshold and b >= bearish_rsi_threshold),
            price_condition=lambda a, b: a > b,  # Price makes higher highs
            rsi_condition=lambda a, b: a < b,    # RSI makes lower highs
            rsi_threshold=45,
            is_bullish=False
        )

        divergence_df = pd.DataFrame(divergences)
        if not divergence_df.empty:
            divergence_df.set_index('end_datetime', inplace=True)
        else:
            divergence_df = pd.DataFrame(columns=['divergence', 'price_change', 'rsi_change', 'future_return', 'TP', 'SL'])
        return divergence_df
