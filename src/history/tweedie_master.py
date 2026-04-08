# ==============================================================================
# [Module] SOTA Tweedie Regression & Queueing Theory Magic
# File: src/tweedie_master.py
# Description: 1위 달성을 위한 Tweedie Loss 도입 및 대기행렬역학(Little's Law) 기반 피처링
# ==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from src.features.build_features import generate_sota_features

def apply_queueing_physics(df):
    """
    [Data-Centric] 대기행렬이론(Queueing Theory)에 기반한 물리적 지연시간 역산 피처
    시뮬레이션 엔진이 지연시간을 계산할 때 사용하는 공식을 근사(Approximation)함
    """
    df = df.copy()
    
    # 1. 시스템 내 처리해야 할 총 작업량 (대기열 길이)
    # 신규 유입 주문 + (혼잡도로 인한 기존 대기 작업 추정치)
    df['total_queue_workload'] = (df['order_inflow_15m'] * df['avg_items_per_order']) * (1 + df['congestion_score'] / 100)
    
    # 2. 단위 시간당 실제 처리 능력 (Service Rate)
    # 활성 로봇 수 * 팩 스테이션 수 * 로봇 가동률 (0 방지를 위해 epsilon 1e-5 추가)
    df['actual_service_rate'] = (df['robot_active'] * df['pack_station_count'] * df['robot_utilization']) + 1e-5
    
    # 3. [1st Place Magic] Little's Law 기반 예상 대기 시간 (대기열 / 처리 능력)
    df['littles_law_estimated_delay'] = df['total_queue_workload'] / df['actual_service_rate']
    
    # 4. 물리적 임계점 (Critical Point) 지시자
    # 처리 능력을 초과하는 주문이 들어온 '병목 확정' 상태 (1 or 0)
    df['is_bottleneck_breached'] = (df['total_queue_workload'] > df['actual_service_rate']).astype(int)
    
    return df

def run_tweedie_model(train_df, test_df, features, target_col='avg_delay_minutes_next_30m'):
    print("\n--- [SOTA Strategy] Tweedie Regression Model ---")
    
    gkf = GroupKFold(n_splits=5)
    groups = train_df['scenario_id']
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, train_df[target_col], groups)):
        X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx][target_col]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # [SOTA for 1st Place] Tweedie Objective 설정
        # tweedie_variance_power (p): 1.0(Poisson) ~ 2.0(Gamma) 사이의 값. 
        # 0이 많고 꼬리가 길면 1.1 ~ 1.5 사이에서 최적의 성능을 냅니다.
        params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.2, # 1등을 위한 핵심 튜닝 포인트
            'metric': 'rmse',
            'learning_rate': 0.02,
            'max_depth': 7,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42+fold,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(test_df[features]) / gkf.n_splits
        
    cv_score = np.sqrt(mean_squared_error(train_df[target_col], oof_preds))
    print(f"\n=> [Tweedie OOF RMSE]: {cv_score:.4f}")
    
    # 약한 예측값 노이즈 제거 (Zero-defense)
    test_preds = np.where(test_preds < 1.0, 0, test_preds)
    
    return test_preds

if __name__ == "__main__":
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    layout_info = pd.read_csv('data/raw/layout_info.csv')
    
    train_df = train_df.merge(layout_info, on='layout_id', how='left')
    test_df = test_df.merge(layout_info, on='layout_id', how='left')
    
    cat_cols = ['layout_type']
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype('category')
        train_df[col] = combined.iloc[:len(train_df)].cat.codes
        test_df[col] = combined.iloc[len(train_df):].cat.codes
        
    train_df_fe = generate_sota_features(train_df)
    test_df_fe = generate_sota_features(test_df)
    
    # 1위 쟁탈을 위한 핵심 물리 공식 주입
    train_df_fe = apply_queueing_physics(train_df_fe)
    test_df_fe = apply_queueing_physics(test_df_fe)
    
    target_col = 'avg_delay_minutes_next_30m'
    noise_to_drop = [
        'intersection_wait_time_avg', 'storage_density_pct', 'racking_height_avg_m', 
        'quality_check_rate', 'kpi_otd_pct', 'backorder_ratio', 'sort_accuracy_pct', 
        'task_reassign_15m'
    ]
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col] + noise_to_drop
    
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    final_predictions = run_tweedie_model(train_df_fe, test_df_fe, features)
    
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = final_predictions
    sub.to_csv('submissions/tweedie_submission.csv', index=False)
    print("\n[SUCCESS] Final Submission Created: submissions/tweedie_submission.csv")