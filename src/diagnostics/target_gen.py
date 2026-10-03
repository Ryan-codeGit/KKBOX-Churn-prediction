import os
import gc
import pandas as pd
import numpy as np

def audit_official_labels():
    t1_path = 'data/raw/transactions.csv'
    t2_path = 'data/raw/transactions_v2.csv'
    train_v2_path = 'data/raw/train_v2.csv'

    print("📦 Loading transaction archives...")
    use_cols = ['msno', 'transaction_date', 'membership_expire_date', 'is_cancel']
    df1 = pd.read_csv(t1_path, usecols=use_cols)
    df2 = pd.read_csv(t2_path, usecols=use_cols)
    transactions = pd.concat([df1, df2], ignore_index=True)
    del df1, df2; gc.collect()

    # Format and sort strictly by timeline sequence
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d', errors='coerce')
    transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d', errors='coerce')
    transactions = transactions.sort_values(['msno', 'transaction_date', 'membership_expire_date']).reset_index(drop=True)

    print("🔍 Reading official train_v2 targets...")
    train_v2 = pd.read_csv(train_v2_path).rename(columns={'is_churn': 'official_label'})

    # Filter transaction histories down STRICTLY to the users present in train_v2
    # This ensures we only look at records the organizers deliberately included
    transactions = transactions[transactions['msno'].isin(train_v2['msno'])].copy()

    print("⚖️ Analyzing user behaviors for logical defects in train_v2...")
    # Step 1: Find every user's absolute latest transaction that occurred *before* March 1st
    feb_history = transactions[transactions['transaction_date'] < pd.Timestamp('2017-03-01')]
    last_feb_record = feb_history.groupby('msno').last().reset_index()

    # Step 2: Extract March transactions to find renewal events
    march_history = transactions[
        (transactions['transaction_date'] >= pd.Timestamp('2017-03-01')) &
        (transactions['transaction_date'] <= pd.Timestamp('2017-03-31'))
    ]

    # Find the earliest renewal event in March for each user
    first_march_record = march_history.groupby('msno').first().reset_index()

    # Step 3: Map them together to measure the true mathematical gap
    audit_base = pd.merge(last_feb_record, first_march_record, on='msno', suffixes=('_feb', '_march'))
    audit_base['gap_days'] = (audit_base['transaction_date_march'] - audit_base['membership_expire_date_feb']).dt.days

    # Step 4: Bring in the official labels to hunt for explicit contradictions
    final_audit = pd.merge(audit_base, train_v2, on='msno')

    # --- BUG SIGHTING 1: The "Invisible Renewal" Error ---
    # User renewed within 30 days, is_cancel is 0, but train_v2 labeled them as CHURNED (1)
    invisible_renewals = final_audit[
        (final_audit['gap_days'] >= 0) &
        (final_audit['gap_days'] < 30) &
        (final_audit['is_cancel_march'] == 0) &
        (final_audit['official_label'] == 1)
    ]

    # --- BUG SIGHTING 2: The "Ghost Active" Error ---
    # User had absolutely no transactions in March, or their transaction gap was > 30 days,
    # but train_v2 labeled them as NOT CHURNED (0)
    ghost_actives = final_audit[
        (final_audit['gap_days'] >= 30) &
        (final_audit['official_label'] == 0)
    ]

    print("\n=======================================================================")
    print("🚨 TRAIN_V2 GROUND-TRUTH DEFECT REPORT")
    print("=======================================================================")
    print(f"Total overlapping users audited      : {len(final_audit):,}")
    print(f"❌ Blatant Errors (Renewed < 30 days but labeled Churn=1) : {len(invisible_renewals):,}")
    print(f"❌ Blatant Errors (Gap >= 30 days but labeled Churn=0)    : {len(ghost_actives):,}")
    print("=======================================================================\n")

    if len(invisible_renewals) > 0:
        print("👀 Smoking Gun Examples (Users who renewed safely but were labeled as Churn):")
        for idx, row in invisible_renewals.head(3).iterrows():
            print(f"\nUser MSNO: {row['msno']}")
            print(f" -> Feb Expire Date: {row['membership_expire_date_feb'].strftime('%Y-%m-%d')}")
            print(f" -> March Trans Date: {row['transaction_date_march'].strftime('%Y-%m-%d')}")
            print(f" -> Calculated Gap  : {row['gap_days']} days")
            print(f" -> train_v2 Label  : {row['official_label']} (Claims Churned!)")

if __name__ == "__main__":
    audit_official_labels()
