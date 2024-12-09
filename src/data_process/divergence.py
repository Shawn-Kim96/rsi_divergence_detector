import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import logging
import time

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


class DivergenceDetector:
    @staticmethod
    def find_previous_peak(df, divergence_start_idx, is_bullish, bullish_peak_rsi_threshold=55, bearish_peak_rsi_threshold=45):
        # Consider only data before the divergence start index
        divergence_start_idx
        df_before_divergence = df.loc[:divergence_start_idx]
        if is_bullish:
            # Bullish: Find previous peaks with RSI >= bullish_peak_rsi_threshold
            peaks_idx, _ = find_peaks(df_before_divergence['high'].values)
            if len(peaks_idx) == 0:
                return None
            valid_peaks = df_before_divergence.iloc[peaks_idx]
            valid_peaks = valid_peaks[valid_peaks['rsi'] >= bullish_peak_rsi_threshold]
            if valid_peaks.empty:
                return None
            return valid_peaks.index[-1]  # Return the index of the last valid peak
        else:
            # Bearish: Find previous valleys with RSI <= bearish_peak_rsi_threshold
            valleys_idx, _ = find_peaks(-df_before_divergence['low'].values)
            if len(valleys_idx) == 0:
                return None
            valid_valleys = df_before_divergence.iloc[valleys_idx]
            valid_valleys = valid_valleys[valid_valleys['rsi'] <= bearish_peak_rsi_threshold]
            if valid_valleys.empty:
                return None
            return valid_valleys.index[-1]  # Return the index of the last valid valley


    @staticmethod
    def calculate_tp_sl(previous_idx, divergence_idx, df, is_bullish):
        previous_high = df.loc[previous_idx, 'high']
        divergence_low = df.loc[divergence_idx, 'low']
        previous_low = df.loc[previous_idx, 'low']
        divergence_high = df.loc[divergence_idx, 'high']

        if is_bullish:
            # For bullish: TP is at 0.382 Fibonacci level
            tp = divergence_low + (previous_high - divergence_low) * 0.382
            sl = divergence_low  # SL is the low at the divergence point
        else:
            # For bearish: TP is at 0.618 Fibonacci level
            tp = divergence_high - (divergence_high - previous_low) * (1 - 0.618)
            sl = divergence_high  # SL is the high at the divergence point
        return tp, sl


    def find_divergences(self, df, rsi_period=14, min_bars_lookback=5, max_bars_lookback=180,
                         bullish_rsi_threshold=30, bearish_rsi_threshold=70, price_prominence=1, rsi_prominence=1):
        
        logging.info(f"Start finding divergence {df.timeframe.values[0]}")
        start_time = time.time()

        df = df.copy()
        price_high = df['high']
        price_low = df['low']
        rsi = df['rsi']
        divergences = []

        # Find peaks and troughs in price and RSI
        price_high_peaks, _ = find_peaks(price_high, prominence=price_prominence)
        price_low_peaks, _ = find_peaks(price_low, prominence=price_prominence)

        rsi_peaks, _ = find_peaks(rsi.values, prominence=rsi_prominence)
        rsi_troughs, _ = find_peaks(-rsi.values, prominence=rsi_prominence)

        # Convert indices to arrays for vectorized operations
        price_high_peaks_df_idx = df.index[price_high_peaks]
        price_low_peaks_df_idx = df.index[price_low_peaks]
        rsi_troughs_df_idx = set(df.index[rsi_troughs])
        rsi_peaks_df_idx = set(df.index[rsi_peaks])

        # Function to detect divergences
        def detect_divergence(idx2_list, price_points, rsi_points, divergence_type, rsi_threshold_check,
                              price_condition, rsi_condition, rsi_threshold, is_bullish):
            
            price_points_positions = np.array([df.index.get_loc(idx) for idx in price_points])
            idx2_positions = np.array([df.index.get_loc(idx) for idx in idx2_list])
            
            for idx2_pos, idx2 in zip(idx2_positions, idx2_list):
                # Set window boundaries using positions
                window_start_pos = max(0, idx2_pos - max_bars_lookback)
                window_end_pos = idx2_pos - min_bars_lookback + 1

                if window_end_pos <= window_start_pos:
                    continue  # Skip if window is invalid

                # Candidate indices within the window
                idx1_candidates_positions = price_points_positions[
                    (price_points_positions >= window_start_pos) & (price_points_positions < window_end_pos)
                ]

                if len(idx1_candidates_positions) == 0:
                    continue

                idx1_candidates = df.index[idx1_candidates_positions]

                price_data = price_low if divergence_type == 'Bullish Divergence' else price_high

                # Price and RSI at idx2
                price_idx2 = price_data.loc[idx2]
                rsi_idx2 = rsi.loc[idx2]

                for idx1 in idx1_candidates:
                    idx1_pos = df.index.get_loc(idx1)
                    price_idx1 = price_data.loc[idx1]
                    rsi_idx1 = rsi.loc[idx1]

                    # Check price condition (higher high or lower low)
                    cond1 = price_condition(price_idx2, price_idx1)
                    cond2 = (idx1 in rsi_points) and (idx2 in rsi_points)
                    cond3 = rsi_condition(rsi_idx2, rsi_idx1)
                    cond4 = rsi_threshold_check(rsi_idx1, rsi_idx2)

                    if cond1 and cond2 and cond3 and cond4:
                        rsi_between = rsi.iloc[idx1_pos + 1: idx2_pos]

                        if len(rsi_between) == 0:
                            continue

                        if divergence_type == 'Bullish Divergence' and np.all(rsi_between >= np.minimum(rsi_idx1, rsi_idx2)):
                            # Calculate TP and SL
                            previous_peak_idx = self.find_previous_peak(df, idx1, is_bullish=True,
                                                                        bullish_peak_rsi_threshold=55,
                                                                        bearish_peak_rsi_threshold=45)
                            if previous_peak_idx is None:
                                continue
                            tp, sl = self.calculate_tp_sl(previous_peak_idx, idx2, df, is_bullish=True)

                            entry_pos = idx2_pos + 2
                            if entry_pos >= len(df):
                                continue
                            entry_idx = df.index[entry_pos]
                            entry_price = df.loc[entry_idx, 'open']

                            # Future return calculation
                            future_pos = idx2_pos + min_bars_lookback
                            if future_pos >= len(df):
                                future_return = np.nan
                                price_change = np.nan
                                rsi_change = np.nan
                            else:
                                future_price = price_data.iloc[future_pos]
                                price_change = future_price - price_data.loc[idx2]
                                rsi_change = df['rsi'].iloc[future_pos] - rsi_idx2
                                future_return = (future_price - price_data.loc[idx2]) / price_data.loc[idx2]

                            # Record divergence
                            divergences.append({
                                'start_datetime': idx1,
                                'end_datetime': idx2,
                                'entry_datetime': entry_idx,
                                'entry_price': entry_price,
                                'previous_peak_datetime': previous_peak_idx,
                                'divergence': divergence_type,
                                'price_change': price_change,
                                'rsi_change': rsi_change,
                                'future_return': future_return,
                                'TP': tp,
                                'SL': sl
                            })
                            break  # Break after finding the first valid divergence

                        elif divergence_type == 'Bearish Divergence' and np.all(rsi_between <= np.maximum(rsi_idx1, rsi_idx2)):
                            # Calculate TP and SL
                            previous_valley_idx = self.find_previous_peak(df, idx1, is_bullish=False,
                                                                            bullish_peak_rsi_threshold=55,
                                                                            bearish_peak_rsi_threshold=45)
                            if previous_valley_idx is None:
                                continue
                            tp, sl = self.calculate_tp_sl(previous_valley_idx, idx2, df, is_bullish=False)

                            entry_pos = idx2_pos + 2
                            if entry_pos >= len(df):
                                continue
                            entry_idx = df.index[entry_pos]
                            entry_price = df.loc[entry_idx, 'open']

                            # Future return calculation
                            future_position = df.index.get_loc(idx2) + min_bars_lookback
                            if future_position >= len(df):
                                future_return = np.nan
                                price_change = np.nan
                                rsi_change = np.nan
                            else:
                                future_price = price_data.iloc[future_position]
                                price_change = future_price - price_data.loc[idx2]
                                rsi_change = df['rsi'].iloc[future_position] - rsi_idx2
                                future_return = (future_price - price_data.loc[idx2]) / price_data.loc[idx2]

                            rsi_between = rsi.iloc[idx1_pos + 1: idx2_pos]
                            # Record divergence

                            divergences.append({
                                'start_datetime': idx1,
                                'end_datetime': idx2,
                                'entry_datetime': entry_idx,
                                'entry_price': entry_price,
                                'previous_peak_datetime': previous_valley_idx,
                                'divergence': divergence_type,
                                'price_change': price_change,
                                'rsi_change': rsi_change,
                                'future_return': future_return,
                                'TP': tp,
                                'SL': sl
                            })
                                
                            break  # Break after finding the first valid divergence

        # Bullish Divergence Detection
        detect_divergence(
            idx2_list=price_low_peaks_df_idx,
            price_points=price_low_peaks_df_idx,
            rsi_points=rsi_troughs_df_idx,
            divergence_type='Bullish Divergence',
            rsi_threshold_check=lambda a, b: (a <= bullish_rsi_threshold and b <= bullish_rsi_threshold),
            price_condition=lambda a, b: a < b,  # Price makes lower lows
            rsi_condition=lambda a, b: a > b,    # RSI makes higher lows
            rsi_threshold=55,
            is_bullish=True
        )

        # Bearish Divergence Detection
        detect_divergence(
            idx2_list=price_high_peaks_df_idx,
            price_points=price_high_peaks_df_idx,
            rsi_points=rsi_peaks_df_idx,
            divergence_type='Bearish Divergence',
            rsi_threshold_check=lambda a, b: (a >= bearish_rsi_threshold and b >= bearish_rsi_threshold),
            price_condition=lambda a, b: a > b,  # Price makes higher highs
            rsi_condition=lambda a, b: a < b,    # RSI makes lower highs
            rsi_threshold=45,
            is_bullish=False
        )
        
        logging.info(f"Finish generating divergenc :: {time.time() - start_time}[s]")
        divergence_df = pd.DataFrame(divergences)
        if not divergence_df.empty:
            divergence_df.set_index('end_datetime', inplace=True)
        else:
            divergence_df = pd.DataFrame(columns=['divergence', 'price_change', 'rsi_change', 'future_return', 'TP', 'SL'])

        return divergence_df
