import os
import gc
import shutil
from typing import Optional
import duckdb
import numpy as np
import pandas as pd


def _read_chunks(path: str, usecols: Optional[list] = None, chunk_size: int = 1000000):
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk_size, columns=usecols):
            yield batch.to_pandas()
    else:
        for chunk in pd.read_csv(
            path, usecols=usecols, chunksize=chunk_size, low_memory=False
        ):
            yield chunk


def process_transactions(
    input_path: str, chunk_size: int = 1000000, reference_date: str = None
) -> pd.DataFrame:
    print(f"Opening Ledger for Pass 1: {input_path}...")

    if reference_date is None:
        raise ValueError('reference_date is required')

    user_latest_dates = {}
    chunk_idx = 1
    for chunk in _read_chunks(
        input_path, usecols=['msno', 'transaction_date'], chunk_size=chunk_size
    ):
        print(f"Pass 1/2: Scanning dates in Chunk #{chunk_idx}...")
        chunk['transaction_date'] = pd.to_datetime(
            chunk['transaction_date'], format='%Y%m%d', errors='coerce'
        )

        local_max = chunk.groupby('msno')['transaction_date'].max()
        for msno, max_date in local_max.items():
            if pd.notna(max_date):
                if msno not in user_latest_dates or max_date > user_latest_dates[msno]:
                    user_latest_dates[msno] = max_date

        del chunk, local_max
        gc.collect()
        chunk_idx += 1

    print(f'\nPass 1 complete. Identified {len(user_latest_dates):,} unique users.')

    df_latest_mapping = pd.DataFrame(
        list(user_latest_dates.items()),
        columns=['msno', 'latest_transaction_date'],
    )
    user_latest_dates.clear()
    gc.collect()

    print(f'\nOpening Ledger for Pass 2: {input_path}...')

    temp_dir = 'temp_processing_db'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    db_path = os.path.join(temp_dir, 'pipeline.duckdb')
    con = duckdb.connect(db_path)

    try:
        # Set PRAGMAs inside try block for guaranteed cleanup on failure
        con.execute(f"PRAGMA temp_directory='{temp_dir}/duckdb_temp'")
        con.execute("PRAGMA max_memory='1.5GB'")
        con.execute('PRAGMA threads=2')

        con.execute("""
            CREATE TABLE seq_chunks (msno VARCHAR, transaction_date TIMESTAMP, is_auto_renew UTINYINT, payment_method_id INTEGER);
            CREATE TABLE mode_counts (msno VARCHAR, feature_name VARCHAR, feature_val DOUBLE, count BIGINT);
            CREATE TABLE hist_aggs (
                msno VARCHAR, sum_plan_days DOUBLE, sum_actual_paid DOUBLE, sum_list_price DOUBLE,
                sum_coverage_days DOUBLE, sum_discount_amount DOUBLE, hist_total_cancels BIGINT,
                hist_transaction_count BIGINT, min_tx_date TIMESTAMP, max_tx_date TIMESTAMP
            );
        """)

        global_latest_df = None
        mode_columns = [
            'payment_plan_days',
            'actual_amount_paid',
            'plan_list_price',
            'payment_method_id',
            'is_auto_renew',
        ]

        chunk_idx = 1
        for chunk in _read_chunks(input_path, chunk_size=chunk_size):
            print(f'Pass 2/2: Processing Chunk #{chunk_idx}...')

            chunk = chunk.drop_duplicates(
                subset=[
                    'msno',
                    'transaction_date',
                    'membership_expire_date',
                    'plan_list_price',
                    'actual_amount_paid',
                ]
            )

            chunk['transaction_date'] = pd.to_datetime(
                chunk['transaction_date'], format='%Y%m%d', errors='coerce'
            )
            chunk['membership_expire_date'] = pd.to_datetime(
                chunk['membership_expire_date'], format='%Y%m%d', errors='coerce'
            )
            chunk['coverage_days'] = (
                (chunk['membership_expire_date'] - chunk['transaction_date'])
                .dt.days.astype('float32')
            )

            chunk['plan_list_price'] = chunk['plan_list_price'].astype('float32')
            chunk['actual_amount_paid'] = chunk['actual_amount_paid'].astype(
                'float32'
            )
            chunk['payment_plan_days'] = chunk['payment_plan_days'].astype('float32')
            chunk['is_cancel'] = chunk['is_cancel'].astype('int8')
            chunk['is_auto_renew'] = chunk['is_auto_renew'].astype('int8')
            chunk['payment_method_id'] = chunk['payment_method_id'].astype('int32')
            chunk['discount_amount'] = (
                chunk['plan_list_price'] - chunk['actual_amount_paid']
            ).astype('float32')

            seq_chunk = chunk[[
                'msno',
                'transaction_date',
                'is_auto_renew',
                'payment_method_id',
            ]]
            con.register('df_seq', seq_chunk)
            con.execute('INSERT INTO seq_chunks SELECT * FROM df_seq')
            con.unregister('df_seq')
            del seq_chunk

            chunk = pd.merge(chunk, df_latest_mapping, on='msno', how='inner')
            is_latest_mask = (
                chunk['transaction_date'] == chunk['latest_transaction_date']
            )

            chunk_latest = chunk[is_latest_mask].drop_duplicates(
                subset=['msno'], keep='last'
            )
            if global_latest_df is None:
                global_latest_df = chunk_latest
            else:
                global_latest_df = pd.concat(
                    [global_latest_df, chunk_latest], ignore_index=True
                ).drop_duplicates(subset=['msno'], keep='last')

            chunk_historical = chunk[~is_latest_mask]

            if not chunk_historical.empty:
                hist_grouped = chunk_historical.groupby('msno', as_index=False).agg(
                    sum_plan_days=('payment_plan_days', 'sum'),
                    sum_actual_paid=('actual_amount_paid', 'sum'),
                    sum_list_price=('plan_list_price', 'sum'),
                    sum_coverage_days=('coverage_days', 'sum'),
                    sum_discount_amount=('discount_amount', 'sum'),
                    hist_total_cancels=('is_cancel', 'sum'),
                    hist_transaction_count=('msno', 'count'),
                    min_tx_date=('transaction_date', 'min'),
                    max_tx_date=('transaction_date', 'max'),
                )
                con.register('df_hist', hist_grouped)
                con.execute('INSERT INTO hist_aggs SELECT * FROM df_hist')
                con.unregister('df_hist')
                del hist_grouped

                for col in mode_columns:
                    mc = (
                        chunk_historical.groupby(['msno', col])
                        .size()
                        .reset_index(name='count')
                    )
                    mc.columns = ['msno', 'feature_val', 'count']
                    mc['feature_name'] = col
                    con.register('df_mc', mc)
                    con.execute(
                        'INSERT INTO mode_counts SELECT msno, feature_name, feature_val,'
                        ' count FROM df_mc'
                    )
                    con.unregister('df_mc')
                    del mc

            del chunk, chunk_latest, chunk_historical
            gc.collect()
            chunk_idx += 1

        del df_latest_mapping
        gc.collect()

        print('\nConsolidating historical aggregates on disk...')
        global_hist_aggregates = con.execute("""
            SELECT
                msno,
                SUM(sum_plan_days) as sum_plan_days,
                SUM(sum_actual_paid) as sum_actual_paid,
                SUM(sum_list_price) as sum_list_price,
                SUM(sum_coverage_days) as sum_coverage_days,
                SUM(sum_discount_amount) as sum_discount_amount,
                SUM(hist_total_cancels) as hist_total_cancels,
                SUM(hist_transaction_count) as hist_transaction_count,
                MIN(min_tx_date) as min_tx_date,
                MAX(max_tx_date) as max_tx_date
            FROM hist_aggs
            GROUP BY msno
        """).df()

        print('Computing global sequential features directly in DuckDB...')

        con.execute("""
            CREATE TEMP TABLE seq_metrics AS
            WITH seq_dedup AS (
                SELECT DISTINCT msno, transaction_date, is_auto_renew, payment_method_id
                FROM seq_chunks
            ),
            ordered_seq AS (
                SELECT msno, transaction_date, is_auto_renew, payment_method_id,
                       LAG(transaction_date) OVER(PARTITION BY msno ORDER BY transaction_date) as prev_date,
                       LAG(is_auto_renew) OVER(PARTITION BY msno ORDER BY transaction_date) as prev_renew,
                       ROW_NUMBER() OVER(PARTITION BY msno ORDER BY transaction_date DESC) as rev_rank
                FROM seq_dedup
            )
            SELECT
                msno,
                transaction_date,
                is_auto_renew,
                payment_method_id,
                DATEDIFF('day', prev_date, transaction_date) as interval_days,
                CASE
                    WHEN prev_renew IS NOT NULL AND is_auto_renew != prev_renew THEN 1
                    ELSE 0
                END as is_flip,
                rev_rank
            FROM ordered_seq;
        """)

        pay_methods = con.execute("""
            SELECT msno, COUNT(DISTINCT payment_method_id) AS hist_payment_method_change
            FROM seq_chunks
            GROUP BY msno
        """).df()

        variance_df = con.execute("""
            SELECT msno, COALESCE(VAR_POP(interval_days), 0.0) AS replenishment_consistency_variance
            FROM seq_metrics
            WHERE interval_days IS NOT NULL
            GROUP BY msno
        """).df()

        flips_df = con.execute("""
            SELECT msno, CAST(SUM(is_flip) AS UTINYINT) AS payment_toggle_state_flips
            FROM seq_metrics
            GROUP BY msno
        """).df()

        trends_df = con.execute("""
            WITH tail_3 AS (
                SELECT msno, is_auto_renew, rev_rank
                FROM seq_metrics
                WHERE rev_rank <= 3
            ),
            stats AS (
                SELECT
                    msno,
                    COUNT(*) as cnt,
                    MAX(CASE WHEN rev_rank = 1 THEN is_auto_renew END) as newest_val,
                    MAX(CASE WHEN rev_rank = 2 THEN is_auto_renew END) as mid_val,
                    MAX(CASE WHEN rev_rank = 3 THEN is_auto_renew END) as oldest_val,
                    MAX(rev_rank) as max_rank
                FROM tail_3
                GROUP BY msno
            )
            SELECT
                msno,
                CASE
                    WHEN cnt < 2 THEN -9
                    WHEN (max_rank = 2 AND newest_val > mid_val) OR (max_rank = 3 AND newest_val > oldest_val) THEN 1
                    WHEN (max_rank = 2 AND newest_val < mid_val) OR (max_rank = 3 AND newest_val < oldest_val) THEN -1
                    ELSE 0
                END AS recent_autorenew_intent_trend
            FROM stats
        """).df()

        con.execute('DROP TABLE IF EXISTS seq_metrics;')

        print('Computing independent true mode values...')
        mode_col_renames = {
            'payment_plan_days': 'hist_ex_latest_mode_payment_plan_days',
            'actual_amount_paid': 'hist_ex_latest_mode_actual_amount_paid',
            'plan_list_price': 'hist_ex_latest_mode_plan_list_price',
            'payment_method_id': 'payment_method_mode',
            'is_auto_renew': 'hist_ex_latest_mode_is_auto_renew',
        }

        df_modes = None
        for col, new_name in mode_col_renames.items():
            mode_sql_df = con.execute(f"""
                WITH aggregated AS (
                    SELECT msno, feature_val, SUM(count) as total_cnt
                    FROM mode_counts
                    WHERE feature_name = '{col}'
                    GROUP BY msno, feature_val
                ),
                ranked AS (
                    SELECT msno, feature_val,
                           ROW_NUMBER() OVER(PARTITION BY msno ORDER BY total_cnt DESC, feature_val ASC) as rank
                    FROM aggregated
                )
                SELECT msno, feature_val AS {new_name}
                FROM ranked
                WHERE rank = 1
            """).df()

            if df_modes is None:
                df_modes = mode_sql_df
            else:
                df_modes = pd.merge(df_modes, mode_sql_df, on='msno', how='outer')
            del mode_sql_df
            gc.collect()

    finally:
        con.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print('Compiling final engineering matrix columns...')
    df_final = pd.merge(
        global_latest_df, global_hist_aggregates, on='msno', how='left'
    )
    df_final = pd.merge(df_final, df_modes, on='msno', how='left')
    df_final = pd.merge(df_final, pay_methods, on='msno', how='left').fillna(
        {'hist_payment_method_change': 0}
    )
    df_final = pd.merge(df_final, variance_df, on='msno', how='left').fillna(
        {'replenishment_consistency_variance': -1.0}
    )
    df_final = pd.merge(df_final, flips_df, on='msno', how='left').fillna(
        {'payment_toggle_state_flips': 0}
    )
    df_final = pd.merge(df_final, trends_df, on='msno', how='left').fillna(
        {'recent_autorenew_intent_trend': -9}
    )

    del (
        global_latest_df,
        global_hist_aggregates,
        df_modes,
        pay_methods,
        variance_df,
        flips_df,
        trends_df,
    )
    gc.collect()

    ref_dt = pd.Timestamp(reference_date)

    tenure_days = (
        (df_final['transaction_date'] - df_final['min_tx_date']).dt.days.fillna(0)
    )
    recency_days = (ref_dt - df_final['transaction_date']).dt.days.fillna(999)
    expiration_recency_days = (
        (ref_dt - df_final['membership_expire_date']).dt.days.fillna(999)
    )

    mean_plan = (
        df_final['sum_plan_days'] / df_final['hist_transaction_count']
    )
    mean_paid = (
        df_final['sum_actual_paid'] / df_final['hist_transaction_count']
    )
    mean_price = (
        df_final['sum_list_price'] / df_final['hist_transaction_count']
    )
    mean_cov = (
        df_final['sum_coverage_days'] / df_final['hist_transaction_count']
    )
    mean_discount = (
        df_final['sum_discount_amount'] / df_final['hist_transaction_count']
    )

    mode_renew_clean = (
        df_final['hist_ex_latest_mode_is_auto_renew'].fillna(-1).astype('int8')
    )
    latest_auto_renew_dropped = np.where(
        mode_renew_clean == -1, -9, mode_renew_clean - df_final['is_auto_renew']
    ).astype('int8')

    elasticity_denom = df_final['hist_ex_latest_mode_plan_list_price'].fillna(
        0.0
    )
    computed_elasticity = np.where(
        elasticity_denom > 0,
        df_final['actual_amount_paid']
        / elasticity_denom.replace(0, np.nan),
        0.0,
    ).astype('float32')

    latest_discount_ratio = np.where(
        df_final['plan_list_price'] > 0,
        df_final['discount_amount']
        / df_final['plan_list_price'].replace(0, np.nan),
        0.0,
    ).astype('float32')
    is_promo_tier = (df_final['plan_list_price'] < 99.0).astype('int8')

    safe_pm_mode = (
        pd.to_numeric(df_final['payment_method_mode'], errors='coerce')
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    features = pd.DataFrame({
        'msno': df_final['msno'],
        'hist_ex_latest_mode_payment_plan_days': df_final[
            'hist_ex_latest_mode_payment_plan_days'
        ].astype('float32'),
        'hist_ex_latest_mode_actual_amount_paid': df_final[
            'hist_ex_latest_mode_actual_amount_paid'
        ].astype('float32'),
        'hist_ex_latest_mode_plan_list_price': df_final[
            'hist_ex_latest_mode_plan_list_price'
        ].astype('float32'),
        'hist_ex_latest_mean_payment_plan_days': mean_plan.fillna(30).astype(
            'float32'
        ),
        'hist_ex_latest_mean_actual_amount_paid': mean_paid.fillna(0).astype(
            'float32'
        ),
        'hist_ex_latest_mean_plan_list_price': mean_price.fillna(0).astype(
            'float32'
        ),
        'hist_ex_latest_mean_coverage_days': mean_cov.fillna(30).astype(
            'float32'
        ),
        'hist_ex_latest_mean_discount_amount': mean_discount.fillna(0).astype(
            'float32'
        ),
        'hist_latest_payment_plan_days': df_final['payment_plan_days'].astype(
            'float32'
        ),
        'hist_latest_plan_list_price': df_final['plan_list_price'].astype(
            'float32'
        ),
        'hist_latest_actual_amount_paid': df_final['actual_amount_paid'].astype(
            'float32'
        ),
        'hist_latest_coverage_days': df_final['coverage_days'].astype('float32'),
        'hist_latest_is_cancel': df_final['is_cancel'].astype('int8'),
        'hist_latest_is_auto_renew': df_final['is_auto_renew'].astype('int8'),
        'hist_total_cancels': df_final['hist_total_cancels']
        .fillna(0)
        .astype('int32'),
        'hist_transaction_count': (
            df_final['hist_transaction_count'].fillna(0) + 1
        ).astype('int32'),
        'hist_payment_method_change': df_final[
            'hist_payment_method_change'
        ].astype('int8'),
        'hist_tenure_days': tenure_days.astype('int32'),
        'hist_recency_days': recency_days.astype('int32'),
        'hist_cancel_rate': (
            df_final['hist_total_cancels'].fillna(0)
            / (df_final['hist_transaction_count'].fillna(0) + 1)
        ).astype('float32'),
        'payment_method_mode': safe_pm_mode,
        'hist_ex_latest_mode_is_auto_renew': mode_renew_clean,
        'latest_auto_renew_dropped': latest_auto_renew_dropped,
        'recent_autorenew_intent_trend': df_final[
            'recent_autorenew_intent_trend'
        ].astype('int8'),
        'hist_expiration_recency_days': expiration_recency_days.astype('int32'),
        'payment_toggle_state_flips': df_final[
            'payment_toggle_state_flips'
        ].astype('int8'),
        'payment_elasticity_vector': computed_elasticity,
        'replenishment_consistency_variance': df_final[
            'replenishment_consistency_variance'
        ].astype('float32'),
        'payment_discount_amount': df_final['discount_amount'].astype('float32'),
        'payment_discount_ratio': latest_discount_ratio,
        'is_promo_tier_subscriber': is_promo_tier,
    })

    del df_final
    gc.collect()
    return features
