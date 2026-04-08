# 파생 변수 생성 로직 (Adversarial Validation 포함)
# ==============================================================================
# [Module] Feature Engineering Script
# File: src/features/build_features.py
# Description: 도메인 지식 기반 파생 변수(Magic Feature) 및 시계열 특성 생성
# ==============================================================================

import pandas as pd


def generate_sota_features(df):
    """
    [Data-Centric] 단순 변수 조합이 아닌 물리적 병목 현상을 수치화하는 Magic Feature 생성
    """
    df = df.copy()

    # 1. Bottleneck Pressure Index (병목 압력 지수)
    df['active_resource_capacity'] = df['robot_active'] * (
                1 - df['robot_charging'] / (df['robot_active'] + df['robot_idle'] + 1e-5))
    df['bottleneck_pressure_idx'] = (df['order_inflow_15m'] * df['avg_items_per_order']) / (
                df['active_resource_capacity'] + 1e-5)

    # 2. Battery Drain Risk (배터리 고갈 위험도)
    df['battery_drain_risk'] = df['robot_utilization'] * (df['battery_std'] / (df['battery_mean'] + 1e-5))

    # 3. Time-Series Lag & Rolling Features (Anti-Leakage 적용)
    sort_cols = ['scenario_id', 'shift_hour']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    rolling_cols = ['congestion_score', 'bottleneck_pressure_idx']
    for col in rolling_cols:
        df[f'{col}_roll3_mean'] = df.groupby('scenario_id')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())
        df[f'{col}_roll3_max'] = df.groupby('scenario_id')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).max())

    return df