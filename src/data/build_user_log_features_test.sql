```sql
CREATE OR REPLACE TABLE `<GCP_PROJECT_ID>.<DATASET_ID>.user_comprehensive_features_test` AS

WITH combined_user_logs AS (
    -- 1. Stack v1 and v2 log tables to maintain complete temporal sequence
    SELECT msno, date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs
    FROM `<GCP_PROJECT_ID>.<DATASET_ID>.user_logs_test`

    UNION ALL

    SELECT msno, date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs
    FROM `<GCP_PROJECT_ID>.<DATASET_ID>.user_logs_train`
),
TargetCohort AS (
    -- 2. Select distinct msno from your test submission table
    SELECT DISTINCT msno
    FROM `<GCP_PROJECT_ID>.<DATASET_ID>.sample_submission_v2`
),
DailyUserLogs AS (
    -- 3. Pre-aggregate stacked logs for users in TargetCohort
    SELECT
        l.msno,
        PARSE_DATE('%Y%m%d', CAST(l.date AS STRING)) AS log_date,
        EXTRACT(MONTH FROM PARSE_DATE('%Y%m%d', CAST(l.date AS STRING))) AS month_idx,
        EXTRACT(DAY FROM PARSE_DATE('%Y%m%d', CAST(l.date AS STRING))) AS day_idx,
        SUM(l.total_secs) AS daily_secs,
        SUM(l.num_25) AS daily_num_25,
        SUM(l.num_50) AS daily_num_50,
        SUM(l.num_75) AS daily_num_75,
        SUM(l.num_985) AS daily_num_985,
        SUM(l.num_100) AS daily_num_100,
        SUM(l.num_unq) AS daily_num_unq,
        SUM(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100) AS daily_songs,
        COUNT(*) AS daily_log_row_count
    FROM combined_user_logs l
    INNER JOIN TargetCohort t ON l.msno = t.msno
    WHERE l.date BETWEEN 20170101 AND 20170331
    GROUP BY l.msno, l.date
),

Pool2DailyCore AS (
    SELECT
        msno,
        GREATEST(
            COALESCE(SAFE_DIVIDE(
                STDDEV_POP(CASE WHEN log_date >= '2017-03-01' THEN daily_songs END),
                AVG(CASE WHEN log_date >= '2017-03-01' THEN daily_songs END)
            ), -1.0)
        ) AS coefficient_of_daily_variation_m3,

        MAX(CASE WHEN log_date BETWEEN '2017-03-22' AND '2017-03-28' THEN daily_songs ELSE 0 END) AS max_single_day_songs_w0
    FROM DailyUserLogs
    WHERE log_date >= '2017-03-01'
    GROUP BY msno
),

Pool2CliffTracking AS (
    SELECT
        msno,
        GREATEST(
            COALESCE(MAX(gap_before_first), 0),
            COALESCE(MAX(internal_gap), 0),
            COALESCE(MAX(gap_after_last), 0)
        ) AS inactivity_chunkiness_index_m3
    FROM (
        SELECT
            msno,
            DATE_DIFF(MIN(log_date) OVER(PARTITION BY msno), DATE '2017-03-01', DAY) AS gap_before_first,
            DATE_DIFF(DATE '2017-03-31', MAX(log_date) OVER(PARTITION BY msno), DAY) AS gap_after_last,
            DATE_DIFF(log_date, LAG(log_date, 1) OVER (PARTITION BY msno ORDER BY log_date ASC), DAY) - 1 AS internal_gap
        FROM DailyUserLogs
        WHERE log_date BETWEEN '2017-03-01' AND '2017-03-31'
    )
    GROUP BY msno
),

MonthlyRollups AS (
    SELECT
        msno,
        -- m1 Summaries (Jan)
        SUM(CASE WHEN month_idx = 1 THEN daily_secs ELSE 0 END) AS m1_total_secs,
        SUM(CASE WHEN month_idx = 1 THEN daily_num_25 ELSE 0 END) AS m1_num_25,
        SUM(CASE WHEN month_idx = 1 THEN daily_num_50 ELSE 0 END) AS m1_num_50,
        SUM(CASE WHEN month_idx = 1 THEN daily_num_75 ELSE 0 END) AS m1_num_75,
        SUM(CASE WHEN month_idx = 1 THEN daily_num_985 ELSE 0 END) AS m1_num_985,
        SUM(CASE WHEN month_idx = 1 THEN daily_num_100 ELSE 0 END) AS m1_num_100,
        SUM(CASE WHEN month_idx = 1 THEN daily_num_unq ELSE 0 END) AS m1_num_unq,
        SUM(CASE WHEN month_idx = 1 THEN daily_songs ELSE 0 END) AS m1_total_songs,
        COUNT(DISTINCT CASE WHEN month_idx = 1 THEN log_date END) AS m1_days_active,
        SUM(CASE WHEN month_idx = 1 THEN daily_log_row_count ELSE 0 END) AS m1_raw_log_rows,

        -- m2 Summaries (Feb)
        SUM(CASE WHEN month_idx = 2 THEN daily_secs ELSE 0 END) AS m2_total_secs,
        SUM(CASE WHEN month_idx = 2 THEN daily_num_25 ELSE 0 END) AS m2_num_25,
        SUM(CASE WHEN month_idx = 2 THEN daily_num_50 ELSE 0 END) AS m2_num_50,
        SUM(CASE WHEN month_idx = 2 THEN daily_num_75 ELSE 0 END) AS m2_num_75,
        SUM(CASE WHEN month_idx = 2 THEN daily_num_985 ELSE 0 END) AS m2_num_985,
        SUM(CASE WHEN month_idx = 2 THEN daily_num_100 ELSE 0 END) AS m2_num_100,
        SUM(CASE WHEN month_idx = 2 THEN daily_num_unq ELSE 0 END) AS m2_num_unq,
        SUM(CASE WHEN month_idx = 2 THEN daily_songs ELSE 0 END) AS m2_total_songs,
        COUNT(DISTINCT CASE WHEN month_idx = 2 THEN log_date END) AS m2_days_active,
        SUM(CASE WHEN month_idx = 2 THEN daily_log_row_count ELSE 0 END) AS m2_raw_log_rows,

        -- Pristine Pre-W3 slice
        SUM(CASE WHEN month_idx = 2 AND log_date <= '2017-02-28' THEN daily_num_unq ELSE 0 END) AS m2_pre_w3_num_unq,
        COUNT(DISTINCT CASE WHEN month_idx = 2 AND log_date <= '2017-02-28' THEN log_date END) AS m2_pre_w3_days_active,

        -- m3 Summaries (Mar)
        SUM(CASE WHEN month_idx = 3 THEN daily_secs ELSE 0 END) AS m3_total_secs,
        SUM(CASE WHEN month_idx = 3 THEN daily_num_25 ELSE 0 END) AS m3_num_25,
        SUM(CASE WHEN month_idx = 3 THEN daily_num_50 ELSE 0 END) AS m3_num_50,
        SUM(CASE WHEN month_idx = 3 THEN daily_num_75 ELSE 0 END) AS m3_num_75,
        SUM(CASE WHEN month_idx = 3 THEN daily_num_985 ELSE 0 END) AS m3_num_985,
        SUM(CASE WHEN month_idx = 3 THEN daily_num_100 ELSE 0 END) AS m3_num_100,
        SUM(CASE WHEN month_idx = 3 THEN daily_num_unq ELSE 0 END) AS m3_num_unq,
        SUM(CASE WHEN month_idx = 3 THEN daily_songs ELSE 0 END) AS m3_total_songs,
        COUNT(DISTINCT CASE WHEN month_idx = 3 THEN log_date END) AS m3_days_active,
        SUM(CASE WHEN month_idx = 3 THEN daily_log_row_count ELSE 0 END) AS m3_raw_log_rows
    FROM DailyUserLogs
    GROUP BY msno
),

M3Granular AS (
    SELECT
        msno,
        -- Micro-Trigger Window (3 Days)
        SUM(CASE WHEN log_date BETWEEN '2017-03-29' AND '2017-03-31' THEN daily_secs ELSE 0 END) AS last_3d_secs,
        SUM(CASE WHEN log_date BETWEEN '2017-03-29' AND '2017-03-31' THEN daily_songs ELSE 0 END) AS last_3d_songs,
        SUM(CASE WHEN log_date BETWEEN '2017-03-29' AND '2017-03-31' THEN daily_num_unq ELSE 0 END) AS last_3d_unq,
        COUNT(DISTINCT CASE WHEN log_date BETWEEN '2017-03-29' AND '2017-03-31' THEN log_date END) AS last_3d_active_days,
        SUM(CASE WHEN log_date BETWEEN '2017-03-29' AND '2017-03-31' THEN daily_log_row_count ELSE 0 END) AS last_3d_raw_rows,

        -- Week 0 (7 Days)
        SUM(CASE WHEN log_date BETWEEN '2017-03-22' AND '2017-03-28' THEN daily_secs ELSE 0 END) AS total_secs_w0,
        SUM(CASE WHEN log_date BETWEEN '2017-03-22' AND '2017-03-28' THEN daily_songs ELSE 0 END) AS total_songs_w0,
        SUM(CASE WHEN log_date BETWEEN '2017-03-22' AND '2017-03-28' THEN daily_num_100 ELSE 0 END) AS num_100_w0,
        SUM(CASE WHEN log_date BETWEEN '2017-03-22' AND '2017-03-28' THEN daily_num_unq ELSE 0 END) AS num_unq_w0,
        COUNT(DISTINCT CASE WHEN log_date BETWEEN '2017-03-22' AND '2017-03-28' THEN log_date END) AS active_days_w0,

        -- Week 1 (7 Days)
        SUM(CASE WHEN log_date BETWEEN '2017-03-15' AND '2017-03-21' THEN daily_secs ELSE 0 END) AS total_secs_w1,
        SUM(CASE WHEN log_date BETWEEN '2017-03-15' AND '2017-03-21' THEN daily_songs ELSE 0 END) AS total_songs_w1,
        SUM(CASE WHEN log_date BETWEEN '2017-03-15' AND '2017-03-21' THEN daily_num_100 ELSE 0 END) AS num_100_w1,
        SUM(CASE WHEN log_date BETWEEN '2017-03-15' AND '2017-03-21' THEN daily_num_unq ELSE 0 END) AS num_unq_w1,
        COUNT(DISTINCT CASE WHEN log_date BETWEEN '2017-03-15' AND '2017-03-21' THEN log_date END) AS active_days_w1,

        -- Week 2 (7 Days)
        SUM(CASE WHEN log_date BETWEEN '2017-03-08' AND '2017-03-14' THEN daily_secs ELSE 0 END) AS total_secs_w2,
        SUM(CASE WHEN log_date BETWEEN '2017-03-08' AND '2017-03-14' THEN daily_num_100 ELSE 0 END) AS num_100_w2,
        SUM(CASE WHEN log_date BETWEEN '2017-03-08' AND '2017-03-14' THEN daily_num_unq ELSE 0 END) AS num_unq_w2,
        COUNT(DISTINCT CASE WHEN log_date BETWEEN '2017-03-08' AND '2017-03-14' THEN log_date END) AS active_days_w2,

        -- Week 3 (7 Days)
        SUM(CASE WHEN log_date BETWEEN '2017-03-01' AND '2017-03-07' THEN daily_secs ELSE 0 END) AS total_secs_w3,
        SUM(CASE WHEN log_date BETWEEN '2017-03-01' AND '2017-03-07' THEN daily_songs ELSE 0 END) AS total_songs_w3,
        SUM(CASE WHEN log_date BETWEEN '2017-03-01' AND '2017-03-07' THEN daily_num_100 ELSE 0 END) AS num_100_w3,
        SUM(CASE WHEN log_date BETWEEN '2017-03-01' AND '2017-03-07' THEN daily_num_unq ELSE 0 END) AS num_unq_w3,
        COUNT(DISTINCT CASE WHEN log_date BETWEEN '2017-03-01' AND '2017-03-07' THEN log_date END) AS active_days_w3
    FROM DailyUserLogs
    WHERE log_date >= '2017-03-01'
    GROUP BY msno
),

BasePrep AS (
    SELECT
        r.msno,
        r.m1_total_secs AS hist_snap_m1_total_secs,
        r.m1_num_25 AS hist_snap_m1_num_25,
        r.m1_num_50 AS hist_snap_m1_num_50,
        r.m1_num_75 AS hist_snap_m1_num_75,
        r.m1_num_985 AS hist_snap_m1_num_985,
        r.m1_num_100 AS hist_snap_m1_num_100,
        r.m1_num_unq AS hist_snap_m1_num_unq,
        r.m1_total_songs AS hist_snap_m1_total_songs,
        r.m1_days_active AS hist_snap_m1_days_active,
        (r.m1_total_songs - r.m1_num_unq) AS hist_snap_m1_repeat_songs,

        r.m2_total_secs AS hist_snap_m2_total_secs,
        r.m2_num_25 AS hist_snap_m2_num_25,
        r.m2_num_50 AS hist_snap_m2_num_50,
        r.m2_num_75 AS hist_snap_m2_num_75,
        r.m2_num_985 AS hist_snap_m2_num_985,
        r.m2_num_100 AS hist_snap_m2_num_100,
        r.m2_num_unq AS hist_snap_m2_num_unq,
        r.m2_total_songs AS hist_snap_m2_total_songs,
        r.m2_days_active AS hist_snap_m2_days_active,
        (r.m2_total_songs - r.m2_num_unq) AS hist_snap_m2_repeat_songs,

        r.m2_pre_w3_num_unq,
        r.m2_pre_w3_days_active,

        r.m3_total_secs AS hist_snap_m3_total_secs,
        r.m3_num_25 AS hist_snap_m3_num_25,
        r.m3_num_50 AS hist_snap_m3_num_50,
        r.m3_num_75 AS hist_snap_m3_num_75,
        r.m3_num_985 AS hist_snap_m3_num_985,
        r.m3_num_100 AS hist_snap_m3_num_100,
        r.m3_num_unq AS hist_snap_m3_num_unq,
        r.m3_total_songs AS hist_snap_m3_total_songs,
        r.m3_days_active AS hist_snap_m3_days_active,
        (r.m3_total_songs - r.m3_num_unq) AS hist_snap_m3_repeat_songs,

        r.m1_raw_log_rows, r.m2_raw_log_rows, r.m3_raw_log_rows,

        COALESCE(g.total_songs_w0, 0) AS total_songs_w0, COALESCE(g.total_secs_w0, 0) AS total_secs_w0,
        COALESCE(g.num_100_w0, 0) AS num_100_w0, COALESCE(g.num_unq_w0, 0) AS num_unq_w0,
        COALESCE(g.total_songs_w1, 0) AS total_songs_w1, COALESCE(g.total_secs_w1, 0) AS total_secs_w1,
        COALESCE(g.num_100_w1, 0) AS num_100_w1, COALESCE(g.num_unq_w1, 0) AS num_unq_w1,
        COALESCE(g.total_songs_w2, 0) AS total_songs_w2, COALESCE(g.total_secs_w2, 0) AS total_secs_w2,
        COALESCE(g.num_100_w2, 0) AS num_100_w2, COALESCE(g.num_unq_w2, 0) AS num_unq_w2,
        COALESCE(g.total_songs_w3, 0) AS total_songs_w3, COALESCE(g.total_secs_w3, 0) AS total_secs_w3,
        COALESCE(g.num_100_w3, 0) AS num_100_w3, COALESCE(g.num_unq_w3, 0) AS num_unq_w3,

        COALESCE(g.active_days_w0, 0) AS active_days_w0,
        COALESCE(g.active_days_w1, 0) AS active_days_w1,
        COALESCE(g.active_days_w2, 0) AS active_days_w2,
        COALESCE(g.active_days_w3, 0) AS active_days_w3,

        COALESCE(g.last_3d_secs, 0) AS last_3d_secs, COALESCE(g.last_3d_songs, 0) AS last_3d_songs,
        COALESCE(g.last_3d_unq, 0) AS last_3d_unq, COALESCE(g.last_3d_active_days, 0) AS last_3d_active_days,
        COALESCE(g.last_3d_raw_rows, 0) AS last_3d_raw_rows,

        COALESCE(p2.coefficient_of_daily_variation_m3, -1.0) AS coefficient_of_daily_variation_m3,
        COALESCE(p2.max_single_day_songs_w0, 0) AS max_single_day_songs_w0,
        COALESCE(cf.inactivity_chunkiness_index_m3, 30) AS inactivity_chunkiness_index_m3
    FROM MonthlyRollups r
    LEFT JOIN M3Granular g ON r.msno = g.msno
    LEFT JOIN Pool2DailyCore p2 ON r.msno = p2.msno
    LEFT JOIN Pool2CliffTracking cf ON r.msno = cf.msno
)

SELECT
    b.*,

    ((b.hist_snap_m1_total_songs + b.hist_snap_m2_total_songs) / 2.0) AS hist_prior_2m_avg_songs,
    ((b.hist_snap_m1_total_secs + b.hist_snap_m2_total_secs) / 2.0) AS hist_prior_2m_avg_secs,
    ((b.hist_snap_m1_days_active + b.hist_snap_m2_days_active) / 2.0) AS hist_prior_2m_avg_days_active,
    ((b.hist_snap_m1_num_unq + b.hist_snap_m2_num_unq) / 2.0) AS hist_prior_2m_avg_unq,
    (((b.hist_snap_m1_total_songs - b.hist_snap_m1_num_unq) + (b.hist_snap_m2_total_songs - b.hist_snap_m2_num_unq)) / 2.0) AS hist_prior_2m_avg_repeats,

    (b.hist_snap_m3_total_songs - ((b.hist_snap_m1_total_songs + b.hist_snap_m2_total_songs) / 2.0)) AS m3_songs_vs_prior_avg_delta,
    (b.hist_snap_m3_total_secs - ((b.hist_snap_m1_total_secs + b.hist_snap_m2_total_secs) / 2.0)) AS m3_secs_vs_prior_avg_delta,
    (b.hist_snap_m3_days_active - ((b.hist_snap_m1_days_active + b.hist_snap_m2_days_active) / 2.0)) AS m3_active_days_vs_prior_avg_delta,
    (b.hist_snap_m3_num_unq - ((b.hist_snap_m1_num_unq + b.hist_snap_m2_num_unq) / 2.0)) AS m3_unq_vs_prior_avg_delta,
    (b.hist_snap_m3_repeat_songs - (((b.hist_snap_m1_total_songs - b.hist_snap_m1_num_unq) + (b.hist_snap_m2_total_songs - b.hist_snap_m2_num_unq)) / 2.0)) AS m3_repeats_vs_prior_avg_delta,

    (b.hist_snap_m3_total_songs - b.hist_snap_m2_total_songs) AS m3_vs_m2_songs_delta,
    (b.hist_snap_m3_total_secs - b.hist_snap_m2_total_secs) AS m3_vs_m2_secs_delta,
    (b.hist_snap_m3_days_active - b.hist_snap_m2_days_active) AS m3_vs_m2_active_days_delta,
    (b.hist_snap_m3_num_unq - b.hist_snap_m2_num_unq) AS m3_vs_m2_unq_delta,
    (b.hist_snap_m3_repeat_songs - b.hist_snap_m2_repeat_songs) AS m3_vs_m2_repeats_delta,

    (b.hist_snap_m2_total_songs - b.hist_snap_m1_total_songs) AS m2_vs_m1_songs_delta,
    (b.hist_snap_m2_total_secs - b.hist_snap_m1_total_secs) AS m2_vs_m1_secs_delta,
    (b.hist_snap_m2_num_unq - b.hist_snap_m1_num_unq) AS m2_vs_m1_unq_delta,
    (b.hist_snap_m2_repeat_songs - b.hist_snap_m1_repeat_songs) AS m2_vs_m1_repeats_delta,

    ((b.hist_snap_m3_total_secs - b.hist_snap_m1_total_secs) / 2.0) AS trend_3m_monthly_velocity_total_secs,
    ((b.hist_snap_m3_num_unq - b.hist_snap_m1_num_unq) / 2.0) AS trend_3m_monthly_velocity_num_unq,
    ((b.hist_snap_m3_num_100 - b.hist_snap_m1_num_100) / 2.0) AS trend_3m_monthly_velocity_num_100,
    ((CAST(b.hist_snap_m3_days_active AS FLOAT64) - CAST(b.hist_snap_m1_days_active AS FLOAT64)) / 2.0) AS trend_3m_monthly_velocity_days_active,
    ((b.hist_snap_m3_repeat_songs - b.hist_snap_m1_repeat_songs) / 2.0) AS trend_3m_monthly_velocity_repeat_songs,

    (b.total_songs_w0 - b.total_songs_w1) AS songs_final_week_vs_prior_week_delta,
    (b.total_secs_w0 - b.total_secs_w1) AS secs_final_week_vs_prior_week_delta,
    (b.num_100_w0 - b.num_100_w1) AS completed_songs_final_week_vs_prior_week_delta,
    (b.num_unq_w0 - b.num_unq_w1) AS unique_songs_final_week_vs_prior_week_delta,

    SAFE_DIVIDE(CAST(b.last_3d_songs AS FLOAT64), GREATEST(1.0, CAST(b.last_3d_active_days AS FLOAT64))) AS avg_daily_songs_last_3d,
    SAFE_DIVIDE(CAST(b.last_3d_secs AS FLOAT64), GREATEST(1.0, CAST(b.last_3d_active_days AS FLOAT64))) AS avg_daily_secs_last_3d,
    (
        SAFE_DIVIDE(CAST(b.last_3d_songs AS FLOAT64), GREATEST(1.0, CAST(b.last_3d_active_days AS FLOAT64))) -
        SAFE_DIVIDE(CAST(b.total_songs_w0 AS FLOAT64), GREATEST(1.0, CAST(b.active_days_w0 AS FLOAT64)))
    ) AS songs_last_3d_vs_w0_daily_delta,
    (
        SAFE_DIVIDE(CAST(b.last_3d_secs AS FLOAT64), GREATEST(1.0, CAST(b.last_3d_active_days AS FLOAT64))) -
        SAFE_DIVIDE(CAST(b.total_secs_w0 AS FLOAT64), GREATEST(1.0, CAST(b.active_days_w0 AS FLOAT64)))
    ) AS secs_last_3d_vs_w0_daily_delta,
    SAFE_DIVIDE(CAST(b.last_3d_active_days AS FLOAT64), 3.0) AS last_3d_active_rate,

    (
        SAFE_DIVIDE(
            CAST(b.num_unq_w0 + b.num_unq_w1 AS FLOAT64),
            GREATEST(1.0, CAST(b.active_days_w0 + b.active_days_w1 AS FLOAT64))
        ) -
        SAFE_DIVIDE(
            CAST(b.hist_snap_m1_num_unq + b.m2_pre_w3_num_unq + b.num_unq_w2 + b.num_unq_w3 AS FLOAT64),
            GREATEST(1.0, CAST(b.hist_snap_m1_days_active + b.m2_pre_w3_days_active + b.active_days_w2 + b.active_days_w3 AS FLOAT64))
        )
    ) AS last2wk_vs_last_month_numunq_avg_diff,

    SAFE_DIVIDE(CAST(b.m3_raw_log_rows AS FLOAT64), GREATEST(1.0, CAST(b.hist_snap_m3_days_active AS FLOAT64))) AS m3_daily_session_density,
    SAFE_DIVIDE(CAST(b.hist_snap_m3_num_unq AS FLOAT64), GREATEST(1.0, CAST(b.hist_snap_m3_days_active AS FLOAT64))) AS m3_daily_variety_density,
    SAFE_DIVIDE(CAST(b.last_3d_raw_rows AS FLOAT64), GREATEST(1.0, CAST(b.last_3d_active_days AS FLOAT64))) AS last_3d_micro_session_density,

    SAFE_DIVIDE(CAST(b.last_3d_songs AS FLOAT64), 3.0) AS raw_last_3d_daily_songs,
    SAFE_DIVIDE(CAST(b.total_songs_w0 AS FLOAT64), 7.0) AS raw_w0_daily_songs,
    b.last_3d_active_days AS raw_last_3d_active_days,
    b.active_days_w0 AS raw_w0_active_days,
    b.m3_raw_log_rows AS raw_m3_log_rows,
    b.hist_snap_m3_days_active AS m3_true_active_days,

    CASE
        WHEN b.hist_snap_m2_num_100 = 0 AND b.hist_snap_m3_num_100 = 0 THEN -1.0
        WHEN b.hist_snap_m2_num_100 = 0 AND b.hist_snap_m3_num_100 > 0 THEN -2.0
        ELSE SAFE_DIVIDE(CAST(b.hist_snap_m3_num_100 AS FLOAT64), CAST(b.hist_snap_m2_num_100 AS FLOAT64))
    END AS true_relative_performance_ratio_complete,
    COALESCE(SAFE_DIVIDE(CAST(b.hist_snap_m3_num_unq AS FLOAT64), CAST(b.hist_snap_m3_days_active AS FLOAT64)), 0.0) AS active_day_engagement_density_m3,
    COALESCE(SAFE_DIVIDE(CAST(b.max_single_day_songs_w0 AS FLOAT64), CAST(b.total_songs_w0 AS FLOAT64)), 0.0) AS extreme_stream_compression_ratio_w0
FROM BasePrep b;
```
