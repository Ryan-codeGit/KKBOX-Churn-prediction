import os
import shutil
import duckdb
import yaml


def join_transaction_files():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    transaction_train = config['paths']['transaction_train']
    transaction_v2 = config['paths']['transaction_v2']
    output_path = 'data/raw/transaction_test.parquet'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Executing RAM-safe disk-backed sort via DuckDB...")

    # Establish temp directory for disk-spilling
    temp_dir = "temp_duckdb_sort"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    con = duckdb.connect()
    try:
        # Direct DuckDB to use disk-spilling when RAM limit is reached
        con.execute(f"PRAGMA temp_directory='{temp_dir}'")

        query = f"""
            COPY (
                SELECT
                    msno,
                    payment_method_id,
                    payment_plan_days,
                    plan_list_price,
                    actual_amount_paid,
                    is_auto_renew,
                    strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d') AS transaction_date,
                    strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS membership_expire_date,
                    is_cancel
                FROM read_csv_auto(['{transaction_train}', '{transaction_v2}'])
                ORDER BY transaction_date ASC
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
        """
        con.execute(query)
        print(f"🎉 Success! Disk-sorted dataset saved to {output_path}")

    finally:
        con.close()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    join_transaction_files()
