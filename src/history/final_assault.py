# ==============================================================================
# [Module] 1st Place Strategic Assault: Scenario-Wise Differential Analysis
# File: src/final_assault.py
# Description: 시나리오 내 시계열 변화율(Delta) 및 누적 적체량(Backlog) 극대화
# ==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from src.features.build_features import generate_sota_features

def apply_golden_features(df):
    """
    [SOTA for 1st Place] 8점대 진입을 위한 시나리오별 누적 및 차분 피처링
    트리 모델이 '변화의 임계점'을 포착하도록 강제함
    """
    df = df.copy()
    df = df.sort_values(['scenario_id', 'shift_hour'])
    
    # 1. 누적 작업 유입량 (시나리오 시작부터 현재까지 얼마나 쌓였는가)
    df['cum_order_volume'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum()
    
    # 2. 작업 처리 가속도 (직전 타임슬롯 대비 가동률 변화)
    df['util_delta'] = df.groupby('scenario_id')['robot_utilization'].diff().fillna(0)
    
    # 3. 배터리 고갈 속도 (시스템 붕괴의 전조)
    df['battery_drop_rate'] = df.groupby('scenario_id')['battery_mean'].diff().fillna(0)
    
    # 4. [Golden Feature] 혼잡도 가중 적체량
    # 단순히 주문이 많은 것보다, 혼잡도가 높은 상태에서 주문이 유입될 때 지연이 기하급수적으로 폭증
    df['congested_backlog_risk'] = df['cum_order_volume'] * (df['congestion_score'] / 100)
    
    return df

def run_final_model(train_df, test_df, features, target_col='avg_delay_minutes_next_30m'):
    print("\n--- [Final Strategy] Physics-Informed Gradient Boosting ---")
    
    # 타겟 값이 0인 데이터가 너무 많으므로, 학습 시 0에 대한 가중치를 살짝 조절하거나
    # Tweedie 파라미터를 극단적으로 설정 (8점대에 진입하기 위한 도박적 설정)
    
    gkf = GroupKFold(n_splits=5)
    groups = train_df['scenario_id']
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, train_df[target_col], groups)):
        X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx][target_col]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression', # Tweedie에서 다시 Regression으로 회귀하되, L1 Loss(MAE) 고려
            'metric': 'rmse',
            'learning_rate': 0.015,     # 더 세밀한 학습
            'max_depth': -1,            # 트리 깊이 제한 해제 (복잡한 패턴 학습)
            'num_leaves': 127,          # 리프 노드 대폭 확장
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'random_state': 42+fold,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=5000,       # 충분한 학습 횟수
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(test_df[features]) / gkf.n_splits
        
    cv_score = np.sqrt(mean_squared_error(train_df[target_col], oof_preds))
    print(f"\n=> [Final Assault OOF RMSE]: {cv_score:.4f}")
    
    # [1st Place Magic Post-Process] 
    # 0에 수렴하는 값들을 과감하게 쳐내지 않으면 8점대에 절대 못 감
    # 1등은 아마도 매우 높은 확률로 지연이 없을 곳을 찾아내어 0으로 밀었을 것
    threshold = 1.5 # 이전보다 더 공격적인 클리핑
    test_preds = np.where(test_preds < threshold, 0, test_preds)
    
    return test_preds

if __name__ == "__main__":
    # 데이터 로드 및 전처리 (기존 로직 동일)
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    layout_info = pd.read_csv('data/raw/layout_info.csv')
    
    train_df = train_df.merge(layout_info, on='layout_id', how='left')
    test_df = test_df.merge(layout_info, on='layout_id', how='left')
    
    # 범주형 변환
    cat_cols = ['layout_type']
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype('category')
        train_df[col] = combined.iloc[:len(train_df)].cat.codes
        test_df[col] = combined.iloc[len(train_df):].cat.codes
        
    # 피처 엔지니어링 1단계 + 2단계(Golden)
    train_df_fe = generate_sota_features(train_df)
    test_df_fe = generate_sota_features(test_df)
    train_df_fe = apply_golden_features(train_df_fe)
    test_df_fe = apply_golden_features(test_df_fe)
    
    target_col = 'avg_delay_minutes_next_30m'
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col, 
                 'intersection_wait_time_avg', 'storage_density_pct', 'racking_height_avg_m', 
                 'quality_check_rate', 'kpi_otd_pct', 'backorder_ratio', 'sort_accuracy_pct', 
                 'task_reassign_15m']
    
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    final_predictions = run_final_model(train_df_fe, test_df_fe, features)
    
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = final_predictions
    sub.to_csv('submissions/final_assault_submission.csv', index=False)
    print("\n[SUCCESS] 1st Place Assault Submission Created.")