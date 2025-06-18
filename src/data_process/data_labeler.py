import numpy as np
import logging
import time
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


class DataLabeler:
    def __init__(self, price_data):
        self.price_df = price_data
        # 인덱스가 datetime이 아니면 변환
        if not isinstance(self.price_df.index, pd.DatetimeIndex):
            if 'datetime' in self.price_df.columns:
                self.price_df['datetime'] = pd.to_datetime(self.price_df['datetime'])
                self.price_df.set_index('datetime', inplace=True)

    def label_divergence_data(self, divergence_df):
        """최적화된 버전의 label_divergence_data 메서드"""
        t = time.time()
        logging.info(f"Start labeling data with optimized method")
        
        if divergence_df.empty:
            logging.info("Empty divergence dataframe, returning as is")
            return divergence_df
        
        # 결과를 저장할 리스트
        labels = []
        
        # 진행 상황을 표시하기 위한 tqdm 사용
        for idx, row in tqdm(divergence_df.iterrows(), total=len(divergence_df), desc="Labeling divergences"):
            entry_datetime = row['entry_datetime']
            tp, sl = row['TP'], row['SL']
            divergence_type = row['divergence']
            
            # entry_datetime이 price_df에 없으면 NaN 추가
            if entry_datetime not in self.price_df.index:
                labels.append(np.nan)
                continue
            
            # entry_datetime 이후의 데이터만 필터링 (한 번만 수행)
            future_prices = self.price_df[self.price_df.index >= entry_datetime]
            
            # 조건에 따라 TP/SL 히트 여부 확인 (벡터화된 연산)
            if divergence_type == "Bearish Divergence":
                # SL 히트 여부 확인
                sl_hit_idx = future_prices[future_prices['high'] >= sl].index.min()
                # TP 히트 여부 확인
                tp_hit_idx = future_prices[future_prices['low'] <= tp].index.min()
            else:  # Bullish Divergence
                # SL 히트 여부 확인
                sl_hit_idx = future_prices[future_prices['low'] <= sl].index.min()
                # TP 히트 여부 확인
                tp_hit_idx = future_prices[future_prices['high'] >= tp].index.min()
            
            # NaT는 히트하지 않은 경우
            hit_tp = not pd.isna(tp_hit_idx)
            hit_sl = not pd.isna(sl_hit_idx)
            
            # TP가 SL보다 먼저 히트했는지 확인
            if hit_tp and hit_sl:
                triggered_label = tp_hit_idx < sl_hit_idx
            else:
                triggered_label = hit_tp and not hit_sl
            
            labels.append(triggered_label)
        
        divergence_df['label'] = labels
        logging.info(f"Finished labeling data, total_time :: {time.time() - t}")
        
        return divergence_df