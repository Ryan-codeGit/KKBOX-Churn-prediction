import gc
import os
import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

def generate_oofs(
    config_path='config.yaml',
    lgb_params_path='models/artifacts/best_lgbm_params.yaml',
    xgb_params_path='models/artifacts/best_xgb_params.yaml',
    feature_manifest_path='models/artifacts/optimized_dropped_features.yaml',
    n_splits=5
):
    print("==================================================")
    print("=== GENERATING OUT-OF-FOLD (OOF) PREDICTIONS ===")
    print("==================================================\n")

    # 1. Load Setup
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_path = config.get('paths', {}).get('train_processed', config.get('train_processed'))

    with open(lgb_params_path, 'r') as f:
        lgb_params = yaml.safe_load(f)
        lgb_params.pop('best_log_loss', None)

    with open(xgb_params_path, 'r') as f:
        xgb_params = yaml.safe_load(f)
        xgb_params.pop('best_log_loss', None)

    with open(feature_manifest_path, 'r') as f:
        selected_features = yaml.safe_load(f)['selected_features']

    # 2. Load Data
    df = pd.read_parquet(train_path)
    target_col = 'is_churn'

    # Filter contradictions
    if 'hist_transaction_count' in df.columns and 'hist_ex_latest_mean_actual_amount_paid' in df.columns:
        contradiction_mask = (df['hist_transaction_count'] == 0) & (df['hist_ex_latest_mean_actual_amount_paid'] > 0)
        df = df[~contradiction_mask].reset_index(drop=True)

    # Enforce Categoricals
    cat_cols = ['city', 'registered_via', 'gender', 'payment_method_mode']
    active_cats = []
    for col in cat_cols:
        if col in selected_features and col in df.columns:
            df[col] = df[col].astype('category')
            active_cats.append(col)

    selected_features = [f for f in selected_features if f in df.columns]

    X = df[selected_features].copy()
    y = df[target_col].astype(int).values

    oof_lgb = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))

    # Stratified K-Fold Cross Validation Setup (Shuffled with Random Seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Starting Stratified {n_splits}-Fold CV to collect clean OOF predictions...\n")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Training Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        # LightGBM Fold
        trn_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=active_cats, free_raw_data=False)
        val_lgb = lgb.Dataset(X_va, label=y_va, reference=trn_lgb, categorical_feature=active_cats, free_raw_data=False)

        model_lgb = lgb.train(
            lgb_params,
            trn_lgb,
            num_boost_round=1500,
            valid_sets=[val_lgb],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_va)

        # XGBoost Fold
        model_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=1500, early_stopping_rounds=30)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]

        gc.collect()

    print(f"\nIndividual OOF Validation Scores:")
    print(f"LightGBM OOF Log Loss: {log_loss(y, oof_lgb):.5f}")
    print(f"XGBoost  OOF Log Loss: {log_loss(y, oof_xgb):.5f}")

    # Save OOF Predictions Matrix
    os.makedirs('models/artifacts', exist_ok=True)
    oof_df = pd.DataFrame({
        'target': y,
        'oof_lgb': oof_lgb,
        'oof_xgb': oof_xgb
    })
    oof_df.to_csv('models/artifacts/oof_predictions.csv', index=False)
    print("\nOOF Predictions successfully saved to 'models/artifacts/oof_predictions.csv'!")

if __name__ == '__main__':
    generate_oofs()
