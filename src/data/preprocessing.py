import os
import gc
import pandas as pd
import numpy as np

def clean_and_impute_master_dataset(df, snapshot_date_str='2017-02-28'):
    """
    Executes a hard-coded, zero-leakage imputation matrix across member demographics,
    transaction histories, and streaming user logs with conflict-free sentinels.
    Returns the cleaned DataFrame completely in-memory.
    """
    print(f"Starting master dataset cleaning pipeline (Input Shape: {df.shape})...")
    snapshot_date = pd.Timestamp(snapshot_date_str)

    # =========================================================================
    # PHASE 1: MEMBER PROFILE FEATURES & FIXED TIME_ACTIVE
    # =========================================================================
    print("Executing Phase 1: Cleaning member demographics...")

    if 'registration_init_time' in df.columns:
        clean_date_strings = pd.to_numeric(df['registration_init_time'], errors='coerce') \
                               .astype('Int64') \
                               .astype(str) \
                               .str.replace('<NA>', '', regex=False)

        reg_dates = pd.to_datetime(clean_date_strings, format='%Y%m%d', errors='coerce')
        df['time_active'] = (snapshot_date - reg_dates).dt.days.astype('float32')
        del clean_date_strings, reg_dates
    else:
        df['time_active'] = np.nan

    member_numerical_categoricals = ['city', 'bd', 'registered_via', 'registration_init_time', 'time_active']
    for col in member_numerical_categoricals:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if col in ['city', 'registered_via']:
                df[col] = df[col].astype('int8')
            elif col in ['bd']:
                df[col] = df[col].astype('int16')
            elif col in ['registration_init_time', 'time_active']:
                df[col] = df[col].astype('float32')

    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('not_spec').astype(str)

    # =========================================================================
    # PHASE 2: TRANSACTION & BILLING FEATURES (CONFLICT-FREE SENTINELS)
    # =========================================================================
    print("Executing Phase 2: Imputing transaction matrix with validated sentinels...")

    # Mode fallback logic from historical latest transactions
    mode_fallbacks = {
        'hist_ex_latest_mode_payment_plan_days': 'hist_latest_payment_plan_days',
        'hist_ex_latest_mode_plan_list_price': 'hist_latest_plan_list_price',
        'hist_ex_latest_mode_actual_amount_paid': 'hist_latest_actual_amount_paid',
        'hist_ex_latest_mode_is_auto_renew': 'hist_latest_is_auto_renew'
    }
    for mode_col, latest_col in mode_fallbacks.items():
        if mode_col in df.columns and latest_col in df.columns:
            df[mode_col] = df[mode_col].fillna(df[latest_col])

    # Validated explicit sentinels (Guaranteed no collision with valid feature values)
    explicit_sentinel_fills = {
        'recent_autorenew_intent_trend': -9,       # Valid values in {-1, 0, 1}
        'latest_auto_renew_dropped': -9,           # Valid values in {-1, 0, 1}
        'hist_latest_is_auto_renew': -1,           # Valid values in {0, 1}
        'hist_ex_latest_mode_is_auto_renew': -1,   # Valid values in {0, 1}
        'hist_latest_is_cancel': -1,               # Valid values in {0, 1}
        'payment_method_mode': '-1',               # Valid IDs are strictly positive integers
        'hist_recency_days': -999,                 # Valid recency days >= 0
        'hist_expiration_recency_days': -9999,     # Shifted to -9999 to prevent clashes with multi-year future expirations
        'replenishment_consistency_variance': -1.0 # True variance is strictly >= 0.0
    }
    for col, fill_val in explicit_sentinel_fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    # Non-negative numeric counts, amounts, rates, and means defaulted to 0.0
    absolute_zero_fills = [
        'hist_transaction_count', 'hist_total_cancels', 'hist_payment_method_change',
        'hist_tenure_days', 'hist_cancel_rate', 'payment_toggle_state_flips',
        'payment_elasticity_vector', 'payment_discount_amount', 'payment_discount_ratio',
        'is_promo_tier_subscriber', 'hist_ex_latest_mean_payment_plan_days',
        'hist_ex_latest_mean_actual_amount_paid', 'hist_ex_latest_mean_plan_list_price',
        'hist_ex_latest_mean_coverage_days', 'hist_ex_latest_mean_discount_amount',
        'hist_latest_payment_plan_days', 'hist_latest_plan_list_price',
        'hist_latest_actual_amount_paid', 'hist_latest_coverage_days',
        'hist_ex_latest_mode_payment_plan_days', 'hist_ex_latest_mode_plan_list_price',
        'hist_ex_latest_mode_actual_amount_paid'
    ]
    for col in absolute_zero_fills:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # =========================================================================
    # PHASE 3: USER LOG STREAMING METRICS (UPDATED TO M1, M2, M3 NAMING)
    # =========================================================================
    print("Executing Phase 3: Zero-filling user streaming telemetry logs...")

    user_log_cols = [col for col in df.columns if col.startswith('hist_snap_') or
                     col.startswith('m1_') or col.startswith('m2_') or col.startswith('m3_') or
                     '_w0' in col or '_w1' in col or '_w2' in col or '_w3' in col or
                     'last_3d' in col or 'songs' in col or 'secs' in col or 'active_days' in col or
                     'trend_3m_' in col or 'hist_prior_2m_' in col or 'last2wk_vs_' in col or
                     col in ['coefficient_of_daily_variation_m3', 'inactivity_chunkiness_index_m3',
                             'true_relative_performance_ratio_complete', 'active_day_engagement_density_m3',
                             'extreme_stream_compression_ratio_w0', 'raw_m3_log_rows',
                             'last2wk_vs_last_month_numunq_avg_diff']]

    for col in user_log_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(0.0)

    remaining_nans = df.isna().sum().sum()
    print("\n========================================= SYSTEM AUDIT =========================================")
    if remaining_nans == 0:
        print(f"🎉 SUCCESS! The dataset is 100% complete. Final matrix scale: {df.shape[0]:,} rows x {df.shape[1]:,} cols.")
    else:
        print(f"⚠️ ATTENTION: {remaining_nans:,} unhandled NaNs remain in the dataset.")
        print(df.isna().sum()[df.isna().sum() > 0])
    print("================================================================================================\n")

    gc.collect()
    return df

def main():
    """Execution wrapper to load the parquet dataset, clean it in-memory, and save to output parquet."""
    input_path = 'data/processed/train_df.parquet'
    output_path = 'data/processed/train_df_processed.parquet'

    if not os.path.exists(input_path):
        print(f"File path error: '{input_path}' not found.")
        return None

    print(f"Loading raw dataset into memory from {input_path}...")
    df = pd.read_parquet(input_path)

    df_cleaned = clean_and_impute_master_dataset(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving processed dataset to {output_path}...")
    df_cleaned.to_parquet(output_path, index=False)
    print("Dataset successfully saved!")

    return df_cleaned

if __name__ == '__main__':
    main()
