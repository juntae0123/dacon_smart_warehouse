# ==============================================================================
# [Module] SOTA Stacking Ensemble & Layout Physics Integration
# File: src/ensemble.py
# Description: layout_info.csv 결합 및 다중 앙상블(LGBM, XGB, CatBoost) 스태킹
# ==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from src.features.build_features import generate_sota_features

# (이전 파일에서 성공한) Asymmetric Custom Loss
def asymmetric_mse(y_pred, dataset):
    y_true = dataset.get_label()
    residual = y_true - y_pred
    alpha = 2.0 
    grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
    hess = np.where(residual > 0, 2 * alpha, 2.0)
    return grad, hess

def rmse_eval(y_pred, dataset):
    y_true = dataset.get_label()
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False

def run_stacking_ensemble(train_df, test_df, features, target_col='avg_delay_minutes_next_30m'):
    print("\n--- [Step 1] Running Level 1 Models (OOF Predictions) ---")
    gkf = GroupKFold(n_splits=5)
    groups = train_df['scenario_id']
    
    # 1층 모델들의 예측값을 담을 배열 (메타 모델의 학습 데이터가 됨)
    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, train_df[target_col], groups)):
        X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx][target_col]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
        X_test = test_df[features]
        
        # 1-1. LightGBM (Custom Loss 특화)
        lgb_data = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_data)
        lgb_params = {'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8, 'random_state': 42+fold, 'verbose': -1}
        
        model_lgb = lgb.train(lgb_params, lgb_data, num_boost_round=1500, valid_sets=[lgb_data, lgb_val], 
                              feval=rmse_eval, callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict(X_val)
        test_preds_lgb += model_lgb.predict(X_test) / gkf.n_splits
        
        # 1-2. XGBoost (트리 구조의 안정성 보완)
        model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, random_state=42+fold)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_val)
        test_preds_xgb += model_xgb.predict(X_test) / gkf.n_splits
        
        # 1-3. CatBoost (Categorical 범주형 데이터 패턴 추출 특화)
        cat_features = ['layout_type'] if 'layout_type' in features else []
        model_cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_state=42+fold, verbose=False)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, early_stopping_rounds=100)
        oof_cat[val_idx] = model_cat.predict(X_val)
        test_preds_cat += model_cat.predict(X_test) / gkf.n_splits
        
        print(f"Fold {fold+1} Completed.")
        
    print(f"\n[Level 1 OOF RMSE] LGBM: {np.sqrt(mean_squared_error(train_df[target_col], oof_lgb)):.4f} | XGB: {np.sqrt(mean_squared_error(train_df[target_col], oof_xgb)):.4f} | CAT: {np.sqrt(mean_squared_error(train_df[target_col], oof_cat)):.4f}")

    print("\n--- [Step 2] Running Meta Model (Ridge Regression) ---")
    # 2층 메타 모델 데이터셋 구성 (1층 모델들의 예측값을 피처로 사용)
    stack_train = np.column_stack((oof_lgb, oof_xgb, oof_cat))
    stack_test = np.column_stack((test_preds_lgb, test_preds_xgb, test_preds_cat))
    
    # 노이즈를 제어하기 위해 선형 모델(Ridge)로 앙상블 조합의 최적 가중치 도출
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_train, train_df[target_col])
    
    final_train_preds = meta_model.predict(stack_train)
    final_test_preds = meta_model.predict(stack_test)
    
    final_rmse = np.sqrt(mean_squared_error(train_df[target_col], final_train_preds))
    print(f"\n=> [Final Stacking RMSE] {final_rmse:.4f}")
    
    # 메타 모델이 각 알고리즘에 부여한 가중치(Weights) 확인
    print(f"Model Weights [LGBM, XGB, CAT]: {meta_model.coef_}")
    
    return final_test_preds

if __name__ == "__main__":
    # 데이터 로드
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    layout_info = pd.read_csv('data/raw/layout_info.csv')
    
    # 1. 원본 데이터와 레이아웃 정보 Merge
    train_df = train_df.merge(layout_info, on='layout_id', how='left')
    test_df = test_df.merge(layout_info, on='layout_id', how='left')
    
    # [FIX: 추가된 핵심 로직] 범주형 문자열 데이터(layout_type)를 정수형(int)으로 라벨 인코딩
    # Train과 Test의 범주가 일치하도록 concat 후 변환
    cat_cols = ['layout_type']
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype('category')
        train_df[col] = combined.iloc[:len(train_df)].cat.codes
        test_df[col] = combined.iloc[len(train_df):].cat.codes
        
    # 2. 피처 엔지니어링 (이전 단계의 SOTA 함수 재사용)
    train_df_fe = generate_sota_features(train_df)
    test_df_fe = generate_sota_features(test_df)
    
    target_col = 'avg_delay_minutes_next_30m'
    
    # 3. 확정된 쓰레기 노이즈 변수 하드 드롭
    noise_to_drop = [
        'intersection_wait_time_avg', 'storage_density_pct', 'racking_height_avg_m', 
        'quality_check_rate', 'kpi_otd_pct', 'backorder_ratio', 'sort_accuracy_pct', 
        'task_reassign_15m'
    ]
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col] + noise_to_drop
    
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    # 4. 스태킹 앙상블 실행 및 최종 예측
    final_predictions = run_stacking_ensemble(train_df_fe, test_df_fe, features)
    
    # 최종 제출물 생성 (sample_submission.csv 형식 유지)
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = final_predictions
    sub.to_csv('submissions/final_stacking_submission.csv', index=False)
    print("\n[SUCCESS] Final Submission File Created: submissions/final_stacking_submission.csv")