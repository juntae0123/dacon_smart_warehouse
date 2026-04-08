# ==============================================================================
# [Module] Advanced Feature Selection Pipeline
# File: src/optimize.py
# Description: 타겟 셔플링(Permutation)을 통한 가짜 피처(Noise) 식별 및 제거
# ==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from src.features.build_features import generate_sota_features

def get_null_importance(X, y, groups, n_runs=3):
    """
    [SOTA] 타겟을 무작위로 섞어 의미 없는 상태를 만든 후 학습하여 가짜 중요도를 판별합니다.
    """
    print(f"\n--- Running Null Importance (Target Shuffling) {n_runs} Iterations ---")
    null_importances = pd.DataFrame()
    
    # 가벼운 검증을 위해 파라미터 축소
    params = {
        'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
        'subsample': 0.8, 'random_state': 42, 'verbose': -1
    }
    
    for run in range(n_runs):
        # 핵심: 타겟 변수 y의 순서를 무작위로 섞어버림 (인과관계 파괴)
        y_shuffled = np.random.permutation(y)
        
        gkf = GroupKFold(n_splits=3)
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y_shuffled, groups)):
            X_train, y_train = X.iloc[trn_idx], y_shuffled[trn_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            fold_imp = pd.DataFrame()
            fold_imp['feature'] = X.columns
            fold_imp['importance'] = model.feature_importances_
            fold_imp['run'] = run
            null_importances = pd.concat([null_importances, fold_imp], axis=0)
            
    return null_importances.groupby('feature')['importance'].mean().reset_index()

if __name__ == "__main__":
    train_df = pd.read_csv('data/raw/train.csv')
    train_df_fe = generate_sota_features(train_df)
    
    target_col = 'avg_delay_minutes_next_30m'
    
    # 1. 0.0의 Importance를 보인 명백한 쓰레기 피처 즉각 제외 (Efficiency)
    known_garbage = ['task_reassign_15m']
    drop_cols = ['ID', 'scenario_id', 'layout_id', target_col] + known_garbage
    features = [c for c in train_df_fe.columns if c not in drop_cols]
    
    X = train_df_fe[features]
    y = train_df_fe[target_col]
    groups = train_df_fe['scenario_id']
    
    # 2. 진짜 Importance 계산
    real_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    real_model.fit(X, y)
    real_imp = pd.DataFrame({'feature': features, 'real_importance': real_model.feature_importances_})
    
    # 3. 가짜 Importance (Null Importance) 계산
    null_imp = get_null_importance(X, y, groups)
    
    # 4. 검증 결과 병합 및 판별
    result_df = pd.merge(real_imp, null_imp, on='feature')
    result_df.rename(columns={'importance': 'null_importance'}, inplace=True)
    
    # 실제 중요도가 가짜 중요도보다 작거나 같으면, 그 피처는 우연히 노이즈를 학습한 것
    result_df['is_noise'] = result_df['real_importance'] <= result_df['null_importance']
    
    print("\n[Permutation Importance Check Result]")
    noise_features = result_df[result_df['is_noise'] == True]
    
    if len(noise_features) > 0:
        print("!!! 다음 변수들은 타겟과 인과관계가 없는 노이즈(Noise)로 판명되었습니다. 즉시 제거하십시오 !!!")
        print(noise_features[['feature', 'real_importance', 'null_importance']])
    else:
        print("모든 피처가 검증을 통과했습니다.")