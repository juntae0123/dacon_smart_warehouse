# ==============================================================================
# [FINAL ATTACK] 8-Point Breakthrough Strategy
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def final_attack():
    print("=== [FINAL ATTACK] Loading Data ===")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    
    target = 'avg_delay_minutes_next_30m'
    
    # ==========================================================================
    # STEP 1: 훈련 데이터 분포 확인 (Critical!)
    # ==========================================================================
    print("\n[STEP 1] Target Distribution Analysis")
    print(f"Train Zero Ratio: {(train[target] == 0).mean():.2%}")
    print(f"Train Max Delay: {train[target].max():.2f}")
    print(f"Train 99th percentile: {train[target].quantile(0.99):.2f}")
    print(f"Train 99.9th percentile: {train[target].quantile(0.999):.2f}")
    
    actual_zero_ratio = (train[target] == 0).mean()
    
    # ==========================================================================
    # STEP 2: 핵심 피처 엔지니어링
    # ==========================================================================
    print("\n[STEP 2] Feature Engineering")
    
    for df in [train, test]:
        # Little's Law: 트래픽 강도 (도착률 / 서비스율)
        arrival_rate = df['order_inflow_15m'] / 15  # 분당 주문
        service_rate = df['robot_active'] / (df['avg_trip_distance'] + 1e-5)
        df['traffic_intensity'] = arrival_rate / (service_rate + 1e-5)
        
        # 대기열 폭발 임계점 (ρ > 1이면 큐가 무한대로)
        df['queue_critical'] = (df['traffic_intensity'] > 1).astype(int)
        
        # 배터리 재앙 지수
        df['battery_crisis'] = (100 - df['battery_mean']) * df['low_battery_ratio']
        
        # 복합 스트레스 지수
        df['stress_compound'] = df['traffic_intensity'] * df['congestion_score'] * (1 + df['battery_crisis']/100)
        
        # 로봇 가용률
        df['robot_availability'] = df['robot_active'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    
    # ==========================================================================
    # STEP 3: 시나리오 타겟 인코딩 (CV Fold로 누수 방지)
    # ==========================================================================
    print("\n[STEP 3] Scenario Target Encoding (Anti-Leak)")
    
    train['sc_target_enc'] = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train['is_delayed'] = (train[target] > 0).astype(int)
    
    for train_idx, val_idx in skf.split(train, train['is_delayed']):
        sc_mean = train.iloc[train_idx].groupby('scenario_id')[target].mean()
        train.loc[val_idx, 'sc_target_enc'] = train.loc[val_idx, 'scenario_id'].map(sc_mean)
    
    train['sc_target_enc'] = train['sc_target_enc'].fillna(train[target].mean())
    
    # Test에는 전체 평균 사용
    sc_global = train.groupby('scenario_id')[target].mean()
    test['sc_target_enc'] = test['scenario_id'].map(sc_global).fillna(train[target].mean())
    
    # ==========================================================================
    # STEP 4: 피처 리스트
    # ==========================================================================
    features = [
        'order_inflow_15m', 'robot_active', 'congestion_score',
        'traffic_intensity', 'queue_critical', 'battery_crisis',
        'stress_compound', 'robot_availability', 'sc_target_enc',
        'avg_trip_distance', 'battery_mean', 'low_battery_ratio'
    ]
    
    # 존재하는 피처만 사용
    features = [f for f in features if f in train.columns and f in test.columns]
    print(f"Using features: {features}")
    
    # ==========================================================================
    # STEP 5: Two-Stage Hurdle Model
    # ==========================================================================
    print("\n[STEP 5] Training Hurdle Model")
    
    # Stage 1: 분류기 (지연 여부)
    clf = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        verbose=-1,
        random_state=42
    )
    clf.fit(train[features], train['is_delayed'])
    prob_delay = clf.predict_proba(test[features])[:, 1]
    
    # Stage 2: 회귀기 (지연 발생 시 양만 예측) - Log Transform 적용!
    train_pos = train[train[target] > 0].copy()
    train_pos['log_target'] = np.log1p(train_pos[target])  # Log Transform
    
    reg = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        verbose=-1,
        random_state=42
    )
    reg.fit(train_pos[features], train_pos['log_target'])
    pred_log = reg.predict(test[features])
    pred_amount = np.expm1(pred_log)  # 역변환
    pred_amount = np.clip(pred_amount, 0, None)
    
    # ==========================================================================
    # STEP 6: 확률 기반 결합 + 극단값 보정
    # ==========================================================================
    print("\n[STEP 6] Combining Predictions")
    
    # 핵심: 훈련 데이터의 실제 Zero Ratio에 맞추기
    threshold = np.percentile(prob_delay, actual_zero_ratio * 100)
    
    final_preds = np.zeros(len(test))
    high_prob_mask = prob_delay >= threshold
    
    # 회귀 예측값 주입
    final_preds[high_prob_mask] = pred_amount[high_prob_mask]
    
    # ==========================================================================
    # STEP 7: 극단값 보정 (Upper Tail Calibration)
    # ==========================================================================
    print("\n[STEP 7] Extreme Value Calibration")
    
    # 훈련 데이터의 상위 분포 참고
    train_q99 = train[target].quantile(0.99)
    train_max = train[target].max()
    pred_q99 = np.percentile(final_preds[final_preds > 0], 99) if (final_preds > 0).sum() > 0 else 1
    
    print(f"Train 99th: {train_q99:.2f}, Train Max: {train_max:.2f}")
    print(f"Pred 99th: {pred_q99:.2f}, Pred Max: {final_preds.max():.2f}")
    
    # 예측이 실제보다 작으면 스케일 업
    if pred_q99 < train_q99 and pred_q99 > 0:
        scale_factor = train_q99 / pred_q99
        print(f"Applying scale factor: {scale_factor:.2f}")
        
        # 상위 10%만 스케일 업 (전체를 키우면 안 됨)
        upper_mask = final_preds > np.percentile(final_preds[final_preds > 0], 90) if (final_preds > 0).sum() > 0 else np.zeros(len(final_preds), dtype=bool)
        final_preds[upper_mask] = final_preds[upper_mask] * min(scale_factor, 3.0)  # 최대 3배로 제한
    
    # ==========================================================================
    # STEP 8: 최종 저장
    # ==========================================================================
    print("\n[STEP 8] Saving Results")
    
    final_preds = np.clip(final_preds, 0, train_max * 1.5)  # 합리적 상한선
    
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target] = final_preds
    sub.to_csv('submissions/FINAL_ATTACK_SUB.csv', index=False)
    
    # 진단
    print("\n" + "="*60)
    print("[FINAL DIAGNOSTICS]")
    print(f"Zero Ratio: {(final_preds == 0).mean():.2%} (Target: {actual_zero_ratio:.2%})")
    print(f"Mean Prediction: {final_preds.mean():.2f}")
    print(f"Max Prediction: {final_preds.max():.2f}")
    print(f"99th Percentile: {np.percentile(final_preds, 99):.2f}")
    print("="*60)
    
    return final_preds

if __name__ == "__main__":
    final_attack()
