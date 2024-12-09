import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


class DataLabeler:
    def __init__(self, price_data):
        self.price_df = price_data

    def label_divergence_data(self, divergence_df):
        t = time.time()
        logging.info(f"Start labeling data")
        labels = []
        for _, row in divergence_df.iterrows():
            entry_datetime = row['entry_datetime']
            tp, sl = row['TP'], row['SL']

            # find entry_datetime
            if entry_datetime not in self.price_df.index:
                labels.append(np.nan)
                continue
                
            triggered_label = False
            hit_tp = False
            hit_sl = False

            # if candle's low <= SL -> SL hit, high >= TP -> TP hit
            for _, row2 in self.price_df[self.price_df.index >= entry_datetime]:
                c_high, c_low = row2['high'], row2['low']

                # SL 터치 여부 우선 확인
                if c_low <= sl:
                    hit_sl = True
                    break

                # TP 터치 여부 확인
                if c_high >= tp:
                    hit_tp = True
                    break

            if hit_tp and not hit_sl:
                triggered_label = True
            else:
                triggered_label = False

            labels.append(triggered_label)
        
        divergence_df['label'] = labels
        logging.info(f"Finished labeling data, total_time :: {time.time() - t}")
        
        return divergence_df