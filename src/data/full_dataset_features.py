import pandas as pd
import numpy as np
import gc

def build_interactions(df,reference_date, low_percentile=0.15):
    """
    Computes anti-drowning vector features in-place, fixes legacy ghost mapping,
    and drops useless features to prevent model noise.
    """

    print("Initializing Memory-Insulated Feature Engineering Map...")

    # --------------------------------------------------------------------
    # SECTION 0: PURGE DISRUPTIVE HEADERS IN-PLACE & CAST PRIMITIVES
    # --------------------------------------------------------------------
    if 'total_secs_w0' in df.columns:
        is_corrupt = df['total_secs_w0'].astype(str).str.strip() == 'total_secs_w0'
        corrupt_count = is_corrupt.sum()
        if corrupt_count > 0:
            df.drop(df[is_corrupt].index, inplace=True)
            df.reset_index(drop=True, inplace=True)

    if 'payment_method_mode' in df.columns:
        df['payment_method_mode'] = pd.to_numeric(df['payment_method_mode'], errors='coerce').fillna(-1).astype('int16')

    log_numeric_cols = [
        'last_3d_secs', 'last_3d_songs', 'last_3d_unq', 'last_3d_active_days',
        'total_secs_w0', 'total_songs_w0', 'num_unq_w0', 'hist_snap_m3_total_songs'
    ]
    for col in log_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float32')

    transaction_numeric_casts = [
        'hist_tenure_days', 'hist_transaction_count', 'hist_latest_is_auto_renew',
        'hist_latest_plan_list_price', 'hist_latest_actual_amount_paid', 'hist_ex_latest_mode_actual_amount_paid'
    ]
    for col in transaction_numeric_casts:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')

    if 'inactivity_chunkiness_index_m3' in df.columns:
        df['inactivity_chunkiness_index_m3'] = pd.to_numeric(df['inactivity_chunkiness_index_m3'], errors='coerce').fillna(30.0).astype('float32')

    # --------------------------------------------------------------------
    # SECTION 1: CALCULATE PERCENTILE ANCHORS
    # --------------------------------------------------------------------
    print("Mapping percentile boundaries...")
    def get_nonzero_quantile(series, q):
        active = series[series > 0]
        return active.quantile(q) if not active.empty else 0.0

    thresh_last_3d_secs = get_nonzero_quantile(df['last_3d_secs'], low_percentile)
    thresh_last_3d_songs = get_nonzero_quantile(df['last_3d_songs'], low_percentile)
    thresh_last_3d_unq = get_nonzero_quantile(df['last_3d_unq'], low_percentile)
    thresh_w0_secs = get_nonzero_quantile(df['total_secs_w0'], low_percentile)
    thresh_w0_songs = get_nonzero_quantile(df['total_songs_w0'], low_percentile)
    thresh_w0_unq = get_nonzero_quantile(df['num_unq_w0'], low_percentile)

    # --------------------------------------------------------------------
    # SECTION 2: SEQUENTIAL IN-PLACE VECTOR INTERACTION INJECTIONS
    # --------------------------------------------------------------------
    print("Injecting mathematical interaction series vectors...")
    auto_renew = df['hist_latest_is_auto_renew'].fillna(0).astype('int8')

    low_3d_habit = ((df['last_3d_active_days'] <= 1) | (df['last_3d_secs'] <= thresh_last_3d_secs) | (df['last_3d_songs'] <= thresh_last_3d_songs))
    df['inter_auto_renew_x_fading_3d_stats'] = (auto_renew * low_3d_habit).astype('int8')
    del low_3d_habit

    low_3d_unq_flag = ((df['last_3d_unq'] == 0) | (df['last_3d_unq'] <= thresh_last_3d_unq))
    df['inter_auto_renew_x_last_3d_low_unq'] = (auto_renew * low_3d_unq_flag).astype('int8')

    low_w0_habit = ((df['total_secs_w0'] <= thresh_w0_secs) | (df['total_songs_w0'] <= thresh_w0_songs))
    df['inter_auto_renew_x_w0_structural_slump'] = (auto_renew * low_w0_habit).astype('int8')
    del low_w0_habit

    past_songs_denom = df['hist_snap_m3_total_songs']
    recent_to_past_ratio = np.where(past_songs_denom > 0, (df['last_3d_songs'] / 3.0) / (past_songs_denom / 28).replace(0, np.nan), np.nan)
    ratio_series = pd.Series(recent_to_past_ratio).dropna()
    thresh_ratio = ratio_series.quantile(low_percentile) if not ratio_series.empty else 0.0
    del ratio_series

    decay_flag = np.where(pd.isna(recent_to_past_ratio), np.where(df['last_3d_songs'] > 0, 0, 1), (recent_to_past_ratio <= thresh_ratio).astype(int))
    df['inter_auto_renew_x_multi_window_decay'] = (auto_renew * decay_flag).astype('int8')
    del recent_to_past_ratio, decay_flag

    low_unq_w0_flag = ((df['num_unq_w0'] == 0) | (df['num_unq_w0'] <= thresh_w0_unq))
    df['inter_auto_renew_x_engagement_desert'] = (auto_renew * low_unq_w0_flag).astype('int8')

    df['inter_auto_renew_x_promo_wall_cross'] = (auto_renew * (df['hist_latest_plan_list_price'] < 99.0)).astype('int8')
    df['inter_auto_renew_x_ghost_billing_disconnect'] = (auto_renew * (df['hist_latest_actual_amount_paid'] == 0.0)).astype('int8')

    print("Injecting loyalty cross-classifications...")
    tenure_cutoff = df['hist_tenure_days'][df['hist_tenure_days'] > 0].quantile(0.7)
    tx_cutoff = df['hist_transaction_count'][df['hist_transaction_count'] > 0].quantile(0.7)
    tenure_cutoff = tenure_cutoff if pd.notna(tenure_cutoff) else 365.0
    tx_cutoff = tx_cutoff if pd.notna(tx_cutoff) else 12.0

    is_loyal = ((df['hist_tenure_days'] >= tenure_cutoff) | (df['hist_transaction_count'] >= tx_cutoff))
    is_low_tier = (df['hist_ex_latest_mode_actual_amount_paid'] <= 129.0)

    df['inter_loyal_x_low_3d_unq'] = (is_loyal & low_3d_unq_flag).astype('int8')
    df['inter_loyal_x_low_tier_payer'] = (is_loyal & is_low_tier).astype('int8')
    df['inter_autorenew_x_desert_payer'] = (auto_renew.astype(bool) & low_unq_w0_flag).astype('int8')
    df['inter_autorenew_x_low_tier_payer'] = (auto_renew.astype(bool) & is_low_tier).astype('int8')

    del is_loyal, is_low_tier, low_3d_unq_flag, low_unq_w0_flag, auto_renew
    gc.collect()

    # --------------------------------------------------------------------
    # SECTION 3: REBUILT LEGACY GHOST ACCOUNT MAPPING
    # --------------------------------------------------------------------
    print("Executing precise ghost account isolation based on profile metadata vacuums...")
    cond_ghost = pd.isna(df['city']) & (df['total_secs_w0'] == 0)

    df['is_legacy_ghost'] = np.where(cond_ghost, 1, 0).astype('int8')
    df['ghost_group_base_rate'] = np.where(df['is_legacy_ghost'] == 1, 0.0542, 0.0).astype('float32')
    print(f"--> Successfully isolated {df['is_legacy_ghost'].sum():,} ghost profiles.")

    # --------------------------------------------------------------------
    # SECTION 3.5: CALCULATE TIME ACTIVE (LEAVING MISSING VALUES AS RAW NaN FOR LATER IMPUTATION)
    # --------------------------------------------------------------------
    print("Calculating system timeline metrics (time_active)...")
    snapshot_date = pd.Timestamp(reference_date)

    if 'registered_init_time' in df.columns:
        # 1. Parse the raw dates to true datetime, turning invalid/missing rows into NaT
        reg_dates = pd.to_datetime(df['registered_init_time'], format='%Y%m%d', errors='coerce')

        # 2. Compute the exact day delta and cast to float32 to naturally preserve missing slots as NaN
        df['time_active'] = (snapshot_date - reg_dates).dt.days.astype('float32')
        del reg_dates
    else:
        # Fallback to empty if the registration column is missing entirely
        df['time_active'] = np.nan

    # --------------------------------------------------------------------
    # SECTION 4: EXPLICIT REMOVAL OF UNWANTED/REDUNDANT COLUMNS
    # --------------------------------------------------------------------
    cols_to_drop = [c for c in ['immediate_contract_expiration_wall', 'multi_year_prepaid_runway'] if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    print("Feature processing complete.")
    gc.collect()
    return df
