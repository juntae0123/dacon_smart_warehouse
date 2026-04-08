# ==============================================================================
# [Module] SOTA Target Transformation & Zero-Defense Pipeline
# File: src/log_transform.py
# Description: Log1p 타겟 변환으로 Long-tail 억제 및 저신뢰도 예측값 Zero-Clipping
# ==============================================================================
'''폐기
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from src.features.build_features import generate_sota_features

def run_log_transform_model(train_df, test_df, features, target_col='avg_delay_minutes_next_30m'):
    print("\n--- [Strategy] Log1p Target Transformation Model ---")
    
    # 1. 타겟 변수에 log(1+x) 변환 적용 (극단적 이상치의 영향을 축소)
    train_df['log_target'] = np.log1p(train_df[target_col])
    
    gkf = GroupKFold(n_splits=5)
    groups = train_df['scenario_id']
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, train_df['log_target'], groups)):
        X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx]['log_target']
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['log_target']
        
        # Log 공간에서는 일반적인 RMSE(L2 Loss)가 가장 안정적임
        model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42+fold,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # 예측 후 exp(x)-1 로 원래 스케일 복원
        oof_preds[val_idx] = np.expm1(model.predict(X_val))
        test_preds += np.expm1(model.predict(test_df[features])) / gkf.n_splits
        
    cv_score = np.sqrt(mean_squared_error(train_df[target_col], oof_preds))
    print(f"\n=> [Log-Transformed OOF RMSE]: {cv_score:.4f}")
    
    print("\n--- [Post-Processing] Aggressive Zero-Defense ---")
    # 모델이 2.0분 이하로 애매하게 예측한 값들은 사실 '0'일 확률이 99%임.
    # RMSE를 깎아먹는 주범인 자잘한 예측값들을 가차 없이 0으로 짓누름.
    threshold = 2.0
    zero_replaced_count = (test_preds < threshold).sum()
    
    final_test_preds = np.where(test_preds < threshold, 0, test_preds)
    
    print(f"-> 예측값이 {threshold}분 미만인 {zero_replaced_count}개의 데이터를 0으로 강제 보정했습니다.")
    
    return final_test_preds

if __name__ == "__main__":
    # 데이터 로드
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
    
    noise_to_drop = [
        'intersection_wait_time_avg', 'storage_density_pct', 'racking_height_avg_m', 
        'quality_check_rate', 'kpi_otd_pct', 'backorder_ratio', 'sort_accuracy_pct', 
        'task_reassign_15m'
    ]
    target_col = 'avg_delay_minutes_next_30m'
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col] + noise_to_drop
    
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    final_predictions = run_log_transform_model(train_df_fe, test_df_fe, features)
    
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = final_predictions
    sub.to_csv('submissions/log_transform_submission.csv', index=False)
    print("\n[SUCCESS] Final Submission Created: submissions/log_transform_submission.csv")
    '''