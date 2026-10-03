import os
import gc
import pandas as pd

def create_purified_dataset():
    t1_path = 'data/raw/transactions.csv'
    t2_path = 'data/raw/transactions_v2.csv'
    train_v2_path = 'data/raw/train_v2.csv'
    output_path = 'data/processed/train_v2_purified.csv'

    print("📦 Loading datasets...")
    use_cols = ['msno', 'transaction_date', 'membership_expire_date', 'is_cancel']
    transactions = pd.concat([
        pd.read_csv(t1_path, usecols=use_cols),
        pd.read_csv(t2_path, usecols=use_cols)
    ], ignore_index=True)

    train_v2 = pd.read_csv(train_v2_path)

    # Sort history chronologically
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d', errors='coerce')
    transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d', errors='coerce')
    transactions = transactions.sort_values(['msno', 'transaction_date', 'membership_expire_date']).reset_index(drop=True)

    print("🔍 Calculating true mathematical timeline...")
    feb_history = transactions[transactions['transaction_date'] < pd.Timestamp('2017-03-01')]
    last_feb_record = feb_history.groupby('msno').last().reset_index()

    march_history = transactions[(transactions['transaction_date'] >= pd.Timestamp('2017-03-01')) & (transactions['transaction_date'] <= pd.Timestamp('2017-03-31'))]
    first_march_record = march_history.groupby('msno').first().reset_index()

    # Merge timelines
    audit_base = pd.merge(last_feb_record, first_march_record, on='msno', suffixes=('_feb', '_march'))
    audit_base['gap_days'] = (audit_base['transaction_date_march'] - audit_base['membership_expire_date_feb']).dt.days

    print("🛠️ Flipping incorrect labels...")
    # Map the official labels in
    merged_targets = pd.merge(train_v2, audit_base[['msno', 'gap_days', 'is_cancel_march']], on='msno', how='left')

    # Create the corrected column, defaulting to the original label
    merged_targets['is_churn_purified'] = merged_targets['is_churn']

    # Fix 1: Labeled Churn (1) but actually renewed within 30 days (Set to 0)
    invisible_renew_mask = (merged_targets['is_churn'] == 1) & (merged_targets['gap_days'] >= 0) & (merged_targets['gap_days'] < 30) & (merged_targets['is_cancel_march'] == 0)
    merged_targets.loc[invisible_renew_mask, 'is_churn_purified'] = 0

    # Fix 2: Labeled Active (0) but actually has a gap >= 30 days (Set to 1)
    ghost_active_mask = (merged_targets['is_churn'] == 0) & (merged_targets['gap_days'] >= 30)
    merged_targets.loc[ghost_active_mask, 'is_churn_purified'] = 1

    # Keep the original 970,960 structure perfectly intact (including the 47k silent drop-offs)
    final_output = pd.DataFrame({
        'msno': merged_targets['msno'],
        'is_churn': merged_targets['is_churn_purified']
    })

    final_output.to_csv(output_path, index=False)
    print(f"✅ Success! Purified targets saved to {output_path}")
    print(f"📊 Fixed {invisible_renew_mask.sum():,} invisible renewals.")
    print(f"📊 Fixed {ghost_active_mask.sum():,} ghost actives.")

if __name__ == "__main__":
    create_purified_dataset()
