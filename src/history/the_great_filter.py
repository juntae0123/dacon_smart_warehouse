# ==============================================================================
# [Module] 1st Place Strategy: The Great Filter (Selective Prediction)
# File: src/the_great_filter.py
# Description: 혼잡도 임계치를 기준으로 지연 발생 구간을 엄격히 분리하여 예측
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def run_filter_strategy(train_df, test_df):
    print("--- [Step 1] Initializing The Great Filter ---")
    
    target_col = 'avg_delay_minutes_next_30m'
    
    # 1. [Physical Tipping Point] 혼잡도와 로봇 가동률 기반의 스트레스 지수 생성
    for df in [train_df, test_df]:
        # 스트레스 지수: 주문량과 혼잡도가 높고 유효 로봇이 적을수록 급증
        df['stress_idx'] = (df['order_inflow_15m'] * df['congestion_score']) / (df['robot_active'] + 1)
    
    # 2. [Data-Centric] 지연이 발생하는 '진짜' 구간만 학습
    # 스트레스 지수가 상위 25%인 데이터만 지연이 발생한다고 가정 (Train 기준)
    threshold_val = train_df['stress_idx'].quantile(0.75)
    
    # 지연 발생 가능성이 높은 'High-Stress' 데이터만 추출하여 회귀 학습
    train_high = train_df[train_df['stress_idx'] > threshold_val].copy()
    
    features = ['order_inflow_15m', 'congestion_score', 'robot_utilization', 
                'robot_active', 'stress_idx', 'battery_mean', 'shift_hour']
    
    print(f"Training on High-Stress clusters: {len(train_high)} samples")
    
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7, verbose=-1)
    model.fit(train_high[features], train_high[target_col])
    
    # 3. [Strict Zeroing] 테스트 데이터에 '그레이트 필터' 적용
    # 스트레스 지수가 하위 80%인 구간은 모델의 예측값을 무시하고 '무조건 0'으로 고정
    test_preds = np.zeros(len(test_df))
    
    # 상위 20%의 위험 구간에 대해서만 모델 예측 수행
    test_high_idx = test_df[test_df['stress_idx'] > test_df['stress_idx'].quantile(0.80)].index
    test_preds[test_high_idx] = model.predict(test_df.loc[test_high_idx, features])
    
    # 4. [Smoothing] 음수 및 미세값 제거
    test_preds = np.clip(test_preds, a_min=0, a_max=None)
    test_preds = np.where(test_preds < 2.0, 0, test_preds)
    
    return test_preds

if __name__ == "__main__":
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    
    final_preds = run_filter_strategy(train, test)
    
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = final_preds
    sub.to_csv('submissions/great_filter_submission.csv', index=False)
    
    non_zero_ratio = (final_preds > 0).mean()
    print(f"\n[Final Check] Non-zero Prediction Ratio: {non_zero_ratio:.2%}")
    print("[SUCCESS] 만약 이 비율이 20% 내외라면, 드디어 10점대 진입의 준비가 된 것입니다.")