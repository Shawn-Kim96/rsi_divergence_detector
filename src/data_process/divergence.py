import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import logging
import time

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

PROJECT_PATH = "/Users/shawn/Documents/personal/rsi_divergence_detector"

# TODO: make function for analyzing divergence data


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


    # def filter_longest_divergences(self, divergence_df):
    #     """
    #     Filters the divergences to keep only the longest one for overlapping divergences.
    #     """
    #     # Add a 'duration' column to calculate the length of each divergence
    #     divergence_df['duration'] = (divergence_df['end_datetime'] - divergence_df.index).total_seconds()

    #     # Group by the overlapping 'end_datetime' and 'divergence' type
    #     filtered_divergences = []
    #     grouped = divergence_df.groupby(['end_datetime', 'divergence'])

    #     for (end_datetime, divergence_type), group in grouped:
    #         # Sort by duration and keep the longest divergence
    #         longest_divergence = group.sort_values(by='duration', ascending=False).iloc[0]
    #         filtered_divergences.append(longest_divergence)

    #     # Rebuild the filtered DataFrame
    #     return pd.DataFrame(filtered_divergences)

    @staticmethod
    def clean_divergence_df(df_div):
        df_div['TP_percent'] = 100 * (df_div['TP'] - df_div['entry_price']) * np.where(df_div['divergence'] == 'Bullish Divergence', 1, -1) / df_div['entry_price'] 
        df_div['SL_percent'] = 100 * (df_div['entry_price'] - df_div['SL']) * np.where(df_div['divergence'] == 'Bullish Divergence', 1, -1) / df_div['entry_price']
        df_div['TP_/_SL'] = df_div['TP_percent'] / df_div['SL_percent']
        is_bullish = np.where(df_div['divergence'] == 'Bullish Divergence', 1, -1)
        df_div['profit'] = np.where(
            df_div['label'],
            is_bullish * (df_div['TP'] - df_div['entry_price']),
            -is_bullish * (df_div['entry_price'] - df_div['SL'])
        )
        
        df_div = df_div.sort_values(by='end_datetime').sort_index()
        df_div[['price_change', 'rsi_change', 'TP', 'SL', 'TP_percent', 'SL_percent', 'TP_/_SL', 'profit']] = df_div[['price_change', 'rsi_change', 'TP', 'SL', 'TP_percent', 'SL_percent', 'TP_/_SL', 'profit']].round(2)


    @staticmethod
    def upload_divergence_data_to_google_sheet(divergence_data, sheet_url = "https://docs.google.com/spreadsheets/d/1uJy2-CV63Pywc2GJJGRP6fSHXmPuTTybS8bMIWhH4Qc/edit?gid=0#gid=0"):
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(f'{PROJECT_PATH}/calcium-ember-444319-n7-3b60cf57e696.json', scope)
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_url(sheet_url)

        sheets = {}

        for key, value in divergence_data.items():
            ddf = value.copy()
            ddf['start_datetime'] = ddf.index
            ddf = ddf[['start_datetime'] + [col for col in ddf.columns if col != 'start_datetime']]
            ddf = ddf.astype(str)
            sheets[key] = ddf
        
        for sheet_name, df in sheets.items():
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="20")
            
            worksheet.clear()
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())


    def find_divergences(self, df, rsi_period=14, min_bars_lookback=5, max_bars_lookback=180,
                         bullish_rsi_threshold=30, bearish_rsi_threshold=70, price_prominence=1, rsi_prominence=1):
        
        logging.info(f"Start finding divergence {df.timeframe.values[0]}")
        start_time = time.time()

        df = df.copy()
        price_high = df['high']
        price_low = df['low']
        price_close = df['close']
        rsi = df['rsi']
        divergences = []

        # Find peaks and troughs in price and RSI
        # price_high_peaks, _ = find_peaks(price_high, prominence=price_prominence)
        # price_low_peaks, _ = find_peaks(-price_low, prominence=price_prominence)
        price_close_peaks, _ = find_peaks(price_close, prominence=price_prominence)
        price_close_peaks_reverse, _ = find_peaks(-price_close, prominence=price_prominence)

        rsi_peaks, _ = find_peaks(rsi.values, prominence=rsi_prominence)
        rsi_troughs, _ = find_peaks(-rsi.values, prominence=rsi_prominence)

        # Convert indices to arrays for vectorized operations
        # price_high_peaks_df_idx = df.index[price_high_peaks]
        # price_low_peaks_df_idx = df.index[price_low_peaks]
        price_high_peaks_df_idx = df.index[price_close_peaks]
        price_low_peaks_df_idx = df.index[price_close_peaks_reverse]
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

                # price_data = price_low if divergence_type == 'Bullish Divergence' else price_high
                price_data = price_close

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
                                # 'future_return': future_return,
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
                                # 'future_return': future_return,
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
        
        logging.info(f"Finish generating divergence :: {time.time() - start_time}[s]")
        divergence_df = pd.DataFrame(divergences)
        divergence_df = divergence_df.sort_index().sort_values(by='end_datetime')

        # Filtering to keep only the longest divergence if similar ones exist
        if not divergence_df.empty:
            divergence_df.set_index('start_datetime', inplace=True)
            # divergence_df = self.filter_longest_divergences(divergence_df)
        else:
            divergence_df = pd.DataFrame(columns=['end_datetime', 'entry_datetime', 'entry_price', 'previous_peak_datetime', 'divergence', 'price_change', 'rsi_change', 'future_return', 'TP', 'SL'])

        return divergence_df

    
    @staticmethod
    def compare_with_different_timeframes(divergence_data):
        # Assuming dd_filter is a dictionary where keys are timeframes ('5m', '15m', etc.)
        # and values are the respective DataFrames.
        timeframe_to_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440  # 하루는 1440분
        }
        # Define the timeframes for which to check divergence
        for timeframe_key in divergence_data.keys():
            for compare_key in divergence_data.keys():
                if timeframe_key != compare_key:
                    divergence_data[timeframe_key][f"div_{compare_key}"] = False

        # Iterate over all timeframes to compare divergences
        for base_timeframe, base_df in divergence_data.items():
            for compare_timeframe, compare_df in divergence_data.items():
                if base_timeframe == compare_timeframe:
                    continue  # 동일한 시간봉은 비교하지 않음

                # 비교 시간봉의 간격(분)을 가져옴
                compare_interval = pd.to_timedelta(timeframe_to_minutes[compare_timeframe], unit='m')

                # 각 base_df의 row에 대해 비교
                for base_index, _ in base_df.iterrows():
                    base_start = base_index

                    # 비교 시간봉의 모든 행에 대해 시작 시간 범위 체크
                    for compare_index, _ in compare_df.iterrows():
                        compare_start = compare_index

                        # 기준 시간(base_start)이 비교 시간 범위(compare_start ~ compare_start + compare_interval)에 있는지 확인
                        if compare_start <= base_start < (compare_start + compare_interval):
                            # 조건을 만족하면 서로의 다이버전스 컬럼을 True로 설정
                            base_df.at[base_index, f"div_{compare_timeframe}"] = True
                            compare_df.at[compare_index, f"div_{base_timeframe}"] = True
        
        return divergence_data