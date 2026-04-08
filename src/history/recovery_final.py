# ==============================================================================
# [Emergency Recovery] 1st Place Zero-Inflation Defense Model
# File: src/RECOVERY_FINAL.py
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb

def recovery_mission():
    print("--- [EMERGENCY] Running Recovery Mission ---")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    layout = pd.read_csv('data/raw/layout_info.csv')
    
    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')
    
    # 1. 클로드의 핵심 피처: 트래픽 강도 (Little's Law)
    for df in [train, test]:
        df['traffic_intensity'] = (df['order_inflow_15m']/15) / (df['robot_active']/(df['avg_trip_distance']+1e-5) + 1e-5)
        df['battery_disaster'] = (100 - df['battery_mean']) * df['low_battery_ratio']
        df['stress_idx'] = df['traffic_intensity'] * df['congestion_score']

    # 2. 타겟 인코딩 (전체 평균으로 단순화하여 과적합 방지)
    sc_map = train.groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()
    train['sc_enc'] = train['scenario_id'].map(sc_map)
    test['sc_enc'] = test['scenario_id'].map(sc_map).fillna(train['avg_delay_minutes_next_30m'].mean())

    features = ['order_inflow_15m', 'traffic_intensity', 'stress_idx', 'battery_disaster', 'sc_enc', 'congestion_score']
    target = 'avg_delay_minutes_next_30m'

    # Stage 1: 분류기 (지연 발생 확률)
    train['is_delayed'] = (train[target] > 0).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
    clf.fit(train[features], train['is_delayed'])
    prob_delay = clf.predict_proba(test[features])[:, 1]

    # Stage 2: 회귀기 (지연 발생 데이터만)
    train_pos = train[train[target] > 0].copy()
    reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
    reg.fit(train_pos[features], train_pos[target])
    pred_amount = reg.predict(test[features])

    # [CRITICAL FIX] 확률 상위 15%만 지연으로 인정 (Zero Ratio 85% 강제 확보)
    final_preds = np.zeros(len(test))
    # 지연 확률이 상위 15%인 인덱스 추출
    threshold = np.percentile(prob_delay, 85) 
    high_prob_idx = np.where(prob_delay >= threshold)[0]
    
    # 해당 구간에만 회귀값 주입 (비선형 가중치 1.2배 포함)
    final_preds[high_prob_idx] = pred_amount[high_prob_idx] * 1.2
    
    # 결과 저장
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target] = np.clip(final_preds, 0, None)
    sub.to_csv('submissions/RECOVERY_SUB.csv', index=False)
    
    print(f"\n[Fixed Status] Zero Ratio: {(final_preds == 0).mean():.2%}")
    print(f"Max Delay: {final_preds.max():.2f}")
    print("이제 Zero Ratio가 85%일 것입니다. 이 파일을 제출하십시오.")

if __name__ == "__main__":
    recovery_mission()