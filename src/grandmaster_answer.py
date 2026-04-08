import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. [Golden Feature #1] Scenario-Level Target Encoding
def apply_scenario_encoding(train_df, test_df, target_col):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['sc_target_enc'] = 0.0
    
    # Anti-Leakage를 위해 K-Fold 기반으로 학습 데이터 인코딩
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        sc_means = fold_train.groupby('scenario_id')[target_col].mean()
        train_df.loc[val_idx, 'sc_target_enc'] = train_df.iloc[val_idx]['scenario_id'].map(sc_means).fillna(train_df[target_col].mean())
    
    # Test는 전체 Train의 평균값 사용
    global_means = train_df.groupby('scenario_id')[target_col].mean()
    test_df['sc_target_enc'] = test_df['scenario_id'].map(global_means).fillna(train_df[target_col].mean())
    return train_df, test_df

# 2. [Golden Feature #2] Little's Law & Battery Crisis
def add_physics_features(df):
    df = df.copy()
    # 대기행렬이론: 트래픽 강도 (도착률 / 서비스율)
    df['arrival_rate'] = df['order_inflow_15m'] / 15.0
    df['service_rate'] = df['robot_active'] / (df['avg_trip_distance'] + 1e-5)
    df['traffic_intensity'] = df['arrival_rate'] / (df['service_rate'] + 1e-5)
    
    # 비선형 임계점: 트래픽 강도가 0.9를 넘으면 시스템 붕괴 시작
    df['queue_critical'] = (df['traffic_intensity'] > 0.9).astype(int)
    
    # 배터리-혼잡 시너지 (재앙 지수)
    df['battery_crisis'] = (100 - df['battery_mean']) * df['low_battery_ratio']
    df['disaster_idx'] = df['congestion_score'] * df['battery_crisis']
    
    # 시계열 누적 피로도
    df = df.sort_values(['scenario_id', 'shift_hour'])
    df['cum_congestion'] = df.groupby('scenario_id')['congestion_score'].cumsum()
    return df

# 3. [Golden Strategy] Simplified Hurdle Model (Classification + Regression)
def train_and_predict(train, test, features, target):
    # Stage 1: 분류 (지연 발생 여부)
    train['is_delayed'] = (train[target] > 0).astype(int)
    
    print("[Stage 1] Classifier Training...")
    clf = lgb.LGBMClassifier(objective='binary', metric='auc', learning_rate=0.03, num_leaves=127, verbose=-1)
    clf.fit(train[features], train['is_delayed'])
    prob_delay = clf.predict_proba(test[features])[:, 1]
    
    # Stage 2: 회귀 (지연 발생한 데이터만 학습)
    print("[Stage 2] Regressor Training (Positive Only)...")
    train_pos = train[train['is_delayed'] == 1].copy()
    reg = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.02, num_leaves=255, verbose=-1)
    reg.fit(train_pos[features], train_pos[target])
    pred_amount = reg.predict(test[features])
    
    # 최종 조합: 클로드의 'Soft Thresholding' 전략 적용
    # 지연 확률이 0.4(클로드 제안) 이상인 곳에만 회귀값의 120%를 할당 (과소예측 방지)
    final_preds = np.where(prob_delay > 0.4, pred_amount * 1.2, 0)
    return np.clip(final_preds, 0, None)

if __name__ == "__main__":
    # 데이터 로드
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    layout = pd.read_csv('data/raw/layout_info.csv')
    
    # Merge & Features
    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')
    
    target_col = 'avg_delay_minutes_next_30m'
    
    train, test = apply_scenario_encoding(train, test, target_col)
    train = add_physics_features(train)
    test = add_physics_features(test)
    
    features = ['order_inflow_15m', 'robot_active', 'congestion_score', 'battery_mean', 
                'sc_target_enc', 'traffic_intensity', 'queue_critical', 'disaster_idx', 'cum_congestion']
    
    # 결측치 채우기
    train = train.fillna(0)
    test = test.fillna(0)
    
    # 예측
    predictions = train_and_predict(train, test, features, target_col)
    
    # 제출 파일
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = predictions
    sub.to_csv('submissions/grandmaster_answer_sub.csv', index=False)
    
    print(f"\n[Final Status] Zero Ratio: {(predictions == 0).mean():.2%}")
    print(f"Max Delay: {predictions.max():.2f}")
    print("[SUCCESS] 1위 탈환을 위한 모델 완성.")