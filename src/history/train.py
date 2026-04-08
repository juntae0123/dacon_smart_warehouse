# ==============================================================================
# [Module] SOTA Training Pipeline with Asymmetric Custom Loss
# File: src/train.py
# Description: 과소 예측에 강한 페널티를 주는 커스텀 손실 함수 적용 및 노이즈 피처 하드 드롭
# ==============================================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from src.features.build_features import generate_sota_features

# 1. [SOTA for 1st Place] Asymmetric Custom Loss (비대칭 손실 함수)
# 실제 지연시간이 예측치보다 큰 경우(Under-prediction) 더 큰 페널티를 부여하여 병목을 민감하게 포착
def asymmetric_mse(y_pred, dataset):
    y_true = dataset.get_label()
    residual = y_true - y_pred
    
    # 페널티 가중치 (알파값이 클수록 과소 예측에 강한 페널티)
    alpha = 2.0 
    
    # residual > 0 이면 실제 지연이 더 길다는 뜻 (치명적 상황)
    grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
    hess = np.where(residual > 0, 2 * alpha, 2.0)
    
    return grad, hess

# 커스텀 평가 지표
def rmse_eval(y_pred, dataset):
    y_true = dataset.get_label()
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False

def run_custom_loss_cv(train_df, features, target_col='avg_delay_minutes_next_30m'):
    print("\n--- Running GroupKFold with SOTA Asymmetric Loss ---")
    gkf = GroupKFold(n_splits=5)
    groups = train_df['scenario_id']
    
    oof_preds = np.zeros(len(train_df))
    feature_importance_df = pd.DataFrame()
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, train_df[target_col], groups)):
        X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx][target_col]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': asymmetric_mse,  # [수정됨] fobj 대신 params 내부로 직접 주입
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42+fold,
            'verbose': -1
        }
        
        # 커스텀 손실 함수 강제 주입
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1500,
            valid_sets=[train_data, val_data],
            feval=rmse_eval,              # feval 지표 검증은 그대로 유지
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        
        fold_imp = pd.DataFrame()
        fold_imp['feature'] = features
        fold_imp['importance'] = model.feature_importance(importance_type='split')
        feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)
        
        print(f"Fold {fold+1} Custom RMSE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx])):.4f}")
        
    cv_score = np.sqrt(mean_squared_error(train_df[target_col], oof_preds))
    print(f"\n=> Overall SOTA OOF RMSE: {cv_score:.4f}")
    
    # 중요도 재확인
    agg_imp = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print("\n[병목 피처의 중요도 부활 여부 확인]")
    print(agg_imp[agg_imp.index.str.contains('bottleneck|battery|drain', na=False)])
    
    return oof_preds

if __name__ == "__main__":
    train_df = pd.read_csv('data/raw/train.csv')
    train_df_fe = generate_sota_features(train_df)
    
    target_col = 'avg_delay_minutes_next_30m'
    
    # 1. Null Importance 결과에 기반한 쓰레기 노이즈 변수 하드 드롭 (Efficiency)
    noise_to_drop = [
        'intersection_wait_time_avg', 'storage_density_pct', 'racking_height_avg_m', 
        'quality_check_rate', 'kpi_otd_pct', 'backorder_ratio', 'sort_accuracy_pct', 
        'task_reassign_15m'
    ]
    
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col] + noise_to_drop
    # 주의: 병목 피처(bottleneck_pressure_idx 등)는 살려두고 손실 함수 변경의 효과를 봅니다.
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    run_custom_loss_cv(train_df_fe, features)