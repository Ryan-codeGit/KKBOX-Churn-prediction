import pandas as pd
import numpy as np
import gc
import os
import yaml

from src.data.process_transaction_data import process_transactions
from src.data.full_dataset_features import build_interactions

def transaction_features(path):
    print("Running deep chunked history parsing on V1 Transactions Ledger...")
    transactions = process_transactions(path,reference_date='2017-02-28')
    return transactions

def main():
    # Load paths from config.yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    target_path = config['paths']['targets']
    user_log_path = config['paths']['user_log_train']
    output_path = config['paths']['train_dataset']
    member_path = config['paths']['members']
    transaction_path = config['paths']['transaction_train']

    print('Loading metadata baseline layers...')
    member = pd.read_csv(member_path)
    target = pd.read_csv(target_path)

    target_users = set(target['msno'].unique())
    print(f"-> Target cohort initialized with {len(target_users):,} unique users.")

    print('Processing core transaction metrics ledger...')
    transactions = transaction_features(transaction_path)
    gc.collect()

    print("-> Slicing transactions matrix down to target pool...")
    transactions = transactions[transactions['msno'].isin(target_users)].reset_index(drop=True)
    gc.collect()

    if os.path.exists(user_log_path):
        # Handle Parquet format from config vs Legacy CSV chunking
        if user_log_path.endswith('.parquet'):
            print(f"-> Loading memory-efficient parquet user logs from {user_log_path}...")
            user_log = pd.read_parquet(user_log_path)
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
        print(f"-> WARNING: {user_log_path} not found! Template empty...")
        user_log = pd.DataFrame(columns=['msno'])

    print('\nExecuting Structural Master Left-Merge Sequence...')
    member = member[member['msno'].isin(target_users)].reset_index(drop=True)

    # Left-join onto the target cohort table template
    df = target
    del target
    gc.collect()

    print("-> Merging member layer...")
    df = pd.merge(df, member, on='msno', how='left')
    del member
    gc.collect()

    print("-> Merging transaction statistics ledger (GHOST TRANSACTIONS PRESERVED)...")
    df = pd.merge(df, transactions, on='msno', how='left')
    del transactions
    gc.collect()

    print("-> Merging user log metrics layer...")
    df = pd.merge(df, user_log, on='msno', how='left')
    del user_log
    gc.collect()

    print(f"-> Verified Final Shape: {df.shape[0]:,} rows.")

    print('Executing anti-drowning logic map & feature corrections...')
    final_processed_df = build_interactions(df,'2017-02-28', low_percentile=0.15)
    del df
    gc.collect()

    print('Saving final unpolluted dataset...')
    out_dir = os.path.dirname(output_path); if out_dir: os.makedirs(out_dir, exist_ok=True)
    final_processed_df.to_parquet(output_path, index=False)
    print("🎉 Pipeline run finalized successfully!")

    del final_processed_df
    gc.collect()

if __name__ == '__main__':
    main()
