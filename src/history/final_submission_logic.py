import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os

def build_final_engine():
    print("--- [Step 1] Data Loading & Physics Merge ---")
    # 경로 설정 (사용자 환경에 맞게 조정)
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    layout = pd.read_csv('data/raw/layout_info.csv')
    
    # 레이아웃 물리 정보 결합
    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')
    
    # 범주형 변환 (layout_type)
    train['layout_type'] = train['layout_type'].astype('category').cat.codes
    test['layout_type'] = test['layout_type'].astype('category').cat.codes

    print("--- [Step 2] Engineering The 'Golden' Features ---")
    def get_features(df):
        df = df.copy()
        df = df.sort_values(['scenario_id', 'shift_hour'])
        
        # 1. 시나리오별 누적 주문량 (시스템 부하 누적치)
        df['cum_order'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum()
        
        # 2. 로봇당 처리 부하 (물리적 한계 지표)
        df['workload_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1e-5)
        
        # 3. 혼잡도 임계점 돌파 지수 (비선형 폭발 트리거)
        df['stress_idx'] = (df['cum_order'] * df['congestion_score']) / (df['robot_active'] + 1)
        
        # 4. 시나리오 내 피크 위험도 (해당 시뮬레이션이 터질 놈인지 판별)
        df['sc_max_stress'] = df.groupby('scenario_id')['stress_idx'].transform('max')
        
        return df

    train = get_features(train)
    test = get_features(test)

    print("--- [Step 3] The 1st Place Strategy: Selective Learning ---")
    target_col = 'avg_delay_minutes_next_30m'
    
    # [핵심] 전체를 학습하지 않고, 지연 발생 가능성이 있는 상위 30% 위험 시나리오만 집중 학습
    # 나머지 70%는 모델이 '평균'을 학습하여 0을 오염시키는 것을 방지
    risk_threshold = train['sc_max_stress'].quantile(0.70)
    train_high = train[train['sc_max_stress'] > risk_threshold].copy()
    
    features = [
        'order_inflow_15m', 'cum_order', 'congestion_score', 'robot_active', 
        'robot_utilization', 'stress_idx', 'sc_max_stress', 'layout_type', 'battery_mean'
    ]
    
    # 1위권 파라미터: 깊고 넓은 트리로 비선형성 포착
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 255,
        'max_depth': 12,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'random_state': 42,
        'verbose': -1
    }

    train_ds = lgb.Dataset(train_high[features], label=train_high[target_col])
    model = lgb.train(params, train_ds, num_boost_round=5000)

    print("--- [Step 4] Strategic Inference & Post-Processing ---")
    # 테스트 셋 예측
    test_preds = np.zeros(len(test))
    
    # [필터] 테스트에서도 스트레스가 높은 상위 15% 시나리오만 지연이 있다고 판단
    # 1위의 8점대 점수는 '확실한 곳만 찌르기'에서 나옵니다.
    final_filter_thresh = test['sc_max_stress'].quantile(0.85)
    high_risk_idx = test[test['sc_max_stress'] > final_filter_thresh].index
    
    # 위험 구간에 대해서만 모델 예측값 할당
    raw_preds = model.predict(test.loc[high_risk_idx, features])
    
    # [비선형 증폭] 지연이 40분을 넘어가면 지연 속도가 빨라지는 물리 법칙 적용
    raw_preds = np.where(raw_preds > 40, raw_preds * 1.2, raw_preds)
    
    test_preds[high_risk_idx] = raw_preds
    
    # 최종 클리핑
    test_preds = np.clip(test_preds, 0, None)
    
    # 결과 저장
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = test_preds
    
    if not os.path.exists('submissions'): os.makedirs('submissions')
    sub.to_csv('submissions/final_ultimate_submission.csv', index=False)
    
    print("\n--- [Final Statistics] ---")
    print(f"Zero Ratio: {(test_preds == 0).sum() / len(test):.2%}")
    print(f"Max Delay: {test_preds.max():.2f} min")
    print(f"Mean Delay: {test_preds.mean():.2f} min")
    print("\n[SUCCESS] submissions/final_ultimate_submission.csv 생성 완료.")

if __name__ == "__main__":
    build_final_engine()