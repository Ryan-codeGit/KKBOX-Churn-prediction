import gc
import os
import numpy as np
import pandas as pd
import yaml

from src.data.full_dataset_features import build_interactions
from src.data.process_transaction_data_test import process_transactions


def transaction_features(path, reference_date='2017-03-31'):
    print(f"Running deep chunked history parsing on Transactions Ledger (Ref Date: {reference_date})...")
    transactions = process_transactions(path, reference_date=reference_date)
    return transactions


def main():
    # Load paths from config.yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Standardize test pipeline path keys with fallback support
    paths = config.get('paths', config)
    target_path = paths['test_targets']
    user_log_path = paths['user_log_test']
    output_path = paths['test_dataset']
    member_path = paths['members']
    transaction_path = paths['test_transaction']

    test_ref_date = '2017-03-31'

    print('Loading test metadata baseline layers...')
    member = pd.read_csv(member_path)
    target = pd.read_csv(target_path)

    target_users = set(target['msno'].unique())
    print(f"-> Test target cohort initialized with {len(target_users):,} unique users.")

    print('Processing core transaction metrics ledger for test cohort...')
    transactions = transaction_features(transaction_path, reference_date=test_ref_date)
    gc.collect()

    print("-> Slicing transactions matrix down to target pool...")
    transactions = transactions[transactions['msno'].isin(target_users)].reset_index(drop=True)
    gc.collect()

    if os.path.exists(user_log_path):
        if user_log_path.endswith('.parquet'):
            print(f"-> Loading memory-efficient parquet user logs from {user_log_path}...")
            user_log = pd.read_csv(user_log_path)
            if 'msno' in user_log.columns:
                user_log = user_log[user_log['msno'] != 'msno']
            user_log = user_log[user_log['msno'].isin(target_users)].reset_index(drop=True)
            gc.collect()
        else:
            print("-> Stream-processing user logs via chunked iteration to defend RAM...")
            user_log_filtered_chunks = []
            for chunk in pd.read_csv(user_log_path, chunksize=500000, dtype={'msno': str}, low_memory=False):
                if 'msno' in chunk.columns:
                    chunk = chunk[chunk['msno'] != 'msno']

                chunk_filtered = chunk[chunk['msno'].isin(target_users)]
                if not chunk_filtered.empty:
                    user_log_filtered_chunks.append(chunk_filtered)

                del chunk
                gc.collect()

            print("-> Consolidating memory-safe filtered user log records...")
            if user_log_filtered_chunks:
                user_log = pd.concat(user_log_filtered_chunks, ignore_index=True)
            else:
                user_log = pd.DataFrame(columns=['msno'])
            user_log_filtered_chunks.clear()
            gc.collect()
    else:
        print(f"-> WARNING: {user_log_path} not found! Initializing empty user_log frame...")
        user_log = pd.DataFrame(columns=['msno'])

    print('\nExecuting Structural Master Left-Merge Sequence...')
    member = member[member['msno'].isin(target_users)].reset_index(drop=True)

    df = target
    del target
    gc.collect()

    print("-> Merging member layer...")
    df = pd.merge(df, member, on='msno', how='left')
    del member
    gc.collect()

    print("-> Merging transaction statistics ledger...")
    df = pd.merge(df, transactions, on='msno', how='left')
    del transactions
    gc.collect()

    print("-> Merging user log metrics layer...")
    df = pd.merge(df, user_log, on='msno', how='left')
    del user_log
    gc.collect()

    print(f"-> Verified Final Test Matrix Shape: {df.shape[0]:,} rows.")

    print('Executing anti-drowning logic map & feature corrections for test cohort...')
    final_processed_df = build_interactions(df, reference_date=test_ref_date, low_percentile=0.15)
    del df
    gc.collect()

    print(f'Saving final test dataset to {output_path}...')
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    final_processed_df.to_parquet(output_path, index=False)
    print("Test pipeline run finalized successfully!")

    del final_processed_df
    gc.collect()


if __name__ == '__main__':
    main()
