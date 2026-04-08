# ==============================================================================
# [Module] 1st Place Magic: Rule-Based Post Processing
# File: src/post_process.py
# Description: 앙상블의 물리적 모순(음수) 제거 및 극단적 병목 구간 강제 보정
# ==============================================================================

import numpy as np
import pandas as pd
from src.features.build_features import generate_sota_features

def apply_domain_rules():
    print("--- Running Domain Rule Post-Processing ---")
    
    # 1. 스태킹 앙상블이 뱉어낸 원본 제출 파일 로드
    sub_path = 'submissions/final_stacking_submission.csv'
    sub_df = pd.read_csv(sub_path)
    
    # 2. Test 데이터의 피처 로드 (어떤 상황인지 파악하기 위해)
    test_df = pd.read_csv('data/raw/test.csv')
    test_df_fe = generate_sota_features(test_df)
    
    target_col = 'avg_delay_minutes_next_30m'
    original_preds = sub_df[target_col].copy()
    
    # =====================================================================
    # [Rule 1] Physical Constraint Clipping (음수 지연시간 방지)
    # =====================================================================
    sub_df[target_col] = np.clip(sub_df[target_col], a_min=0, a_max=None)
    clipped_count = (original_preds < 0).sum()
    print(f"[Rule 1] 음수 예측값 0으로 보정 완료: {clipped_count}건 수정됨")
    
    # =====================================================================
    # [Rule 2] Extreme Bottleneck Boosting (극단적 병목 과소예측 보정)
    # 트리 모델이 예측하지 못하는 상위 1%의 극악의 병목 상황은 강제로 지연시간을 15% 가중함
    # =====================================================================
    # 우리가 만든 핵심 피처 활용
    pressure_idx = test_df_fe['bottleneck_pressure_idx_roll3_max']
    
    # 상위 1%의 임계치 계산
    threshold_99th = pressure_idx.quantile(0.99)
    
    # 임계치를 초과하는 악성 병목 시나리오의 인덱스 추출
    extreme_idx = test_df_fe[pressure_idx >= threshold_99th].index
    
    # 해당 시나리오들의 예측 지연시간에 1.15배의 Multiplier 적용
    sub_df.loc[extreme_idx, target_col] = sub_df.loc[extreme_idx, target_col] * 1.15
    print(f"[Rule 2] 상위 1% 악성 병목 구간(압력지수 > {threshold_99th:.2f}) 15% 가중 보정 완료: {len(extreme_idx)}건 수정됨")
    
    # 최종 보정된 파일 저장
    final_path = 'submissions/post_processed_submission.csv'
    sub_df.to_csv(final_path, index=False)
    print(f"\n[SUCCESS] Final Post-Processed Submission Created: {final_path}")

if __name__ == "__main__":
    apply_domain_rules()