# ==============================================================================
# [Module] SOTA Two-Stage Hurdle Model for Zero-Inflated Targets
# File: src/two_stage.py
# Description: 지연 발생 여부(Classification) -> 지연 시간 예측(Regression) 2단계 구조
# ==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.features.build_features import generate_sota_features

def run_two_stage_model(train_df, test_df, features, target_col='avg_delay_minutes_next_30m'):
    print("\n--- [Stage 1] Classification: Will it be delayed? (0 or 1) ---")
    
    # 지연이 0보다 크면 1, 아니면 0인 이진 타겟 생성
    train_df['is_delayed'] = (train_df[target_col] > 0).astype(int)
    print(f"Delay Ratio in Train: {train_df['is_delayed'].mean():.2%}")
    
    gkf = GroupKFold(n_splits=5)
    groups = train_df['scenario_id']
    
    oof_class_preds = np.zeros(len(train_df))
    test_class_preds = np.zeros(len(test_df))
    
    # 1. Classification (분류) 학습
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, train_df['is_delayed'], groups)):
        X_tr, y_tr = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx]['is_delayed']
        X_va, y_va = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['is_delayed']
        
        clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, max_depth=6, 
                                 is_unbalance=True, random_state=42+fold, verbose=-1)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        oof_class_preds[val_idx] = clf.predict_proba(X_va)[:, 1]
        test_class_preds += clf.predict_proba(test_df[features])[:, 1] / gkf.n_splits
        
    print(f"Stage 1 OOF AUC Score: {roc_auc_score(train_df['is_delayed'], oof_class_preds):.4f}")
    
    print("\n--- [Stage 2] Regression: How long is the delay? (Only for delayed targets) ---")
    
    # 지연이 실제로 발생한 데이터만 필터링하여 회귀 모델 학습 (노이즈 원천 차단)
    delayed_train = train_df[train_df['is_delayed'] == 1].copy()
    delayed_groups = delayed_train['scenario_id']
    
    oof_reg_preds = np.zeros(len(train_df))
    test_reg_preds = np.zeros(len(test_df))
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(delayed_train, delayed_train[target_col], delayed_groups)):
        X_tr, y_tr = delayed_train.iloc[trn_idx][features], delayed_train.iloc[trn_idx][target_col]
        X_va, y_va = delayed_train.iloc[val_idx][features], delayed_train.iloc[val_idx][target_col]
        
        reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, 
                                random_state=42+fold, verbose=-1)
        reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # Validation 셋에 대한 예측 기록 (실제 인덱스에 맵핑)
        val_real_idx = delayed_train.index[val_idx]
        oof_reg_preds[val_real_idx] = reg.predict(X_va)
        
        # Test 셋 예측 (Stage 1 결과와 상관없이 일단 전부 회귀 예측)
        test_reg_preds += reg.predict(test_df[features]) / gkf.n_splits
        
    print(f"Stage 2 OOF RMSE (Only Delayed): {np.sqrt(mean_squared_error(delayed_train[target_col], oof_reg_preds[delayed_train.index])):.4f}")
    
    print("\n--- [Final Stage] Blending Classification & Regression ---")
    # 최적의 확률 임계값(Threshold) 찾기
    best_score = float('inf')
    best_thresh = 0.5
    
    # OOF 기반으로 Threshold 튜닝 (0.3 ~ 0.7)
    for thresh in np.arange(0.3, 0.7, 0.05):
        # 확률이 Threshold를 넘으면 회귀 예측값 사용, 아니면 강제로 0 처리
        final_oof = np.where(oof_class_preds > thresh, oof_reg_preds, 0)
        score = np.sqrt(mean_squared_error(train_df[target_col], final_oof))
        if score < best_score:
            best_score = score
            best_thresh = thresh
            
    print(f"Best Threshold for Zero-Inflation: {best_thresh:.2f}")
    print(f"=> Final Two-Stage OOF RMSE: {best_score:.4f}")
    
    # Test 데이터에 최종 룰 적용
    final_test_preds = np.where(test_class_preds > best_thresh, test_reg_preds, 0)
    
    return final_test_preds

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
    
    # 이전 단계에서 식별된 노이즈 완벽 제거
    noise_to_drop = [
        'intersection_wait_time_avg', 'storage_density_pct', 'racking_height_avg_m', 
        'quality_check_rate', 'kpi_otd_pct', 'backorder_ratio', 'sort_accuracy_pct', 
        'task_reassign_15m'
    ]
    target_col = 'avg_delay_minutes_next_30m'
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col] + noise_to_drop
    
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    final_predictions = run_two_stage_model(train_df_fe, test_df_fe, features)
    
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub[target_col] = final_predictions
    sub.to_csv('submissions/two_stage_submission.csv', index=False)
    print("\n[SUCCESS] Final Submission Created: submissions/two_stage_submission.csv")