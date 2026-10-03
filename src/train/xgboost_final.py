import gc
import os
import yaml
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def train_final_xgboost(
    config_path='config.yaml',
    params_path='models/artifacts/best_xgb_params.yaml',
    feature_manifest_path='models/artifacts/optimized_dropped_features.yaml'
):
    print("==================================================")
    print("=== FINAL XGBOOST: VALIDATE & RETRAIN ON FULL DATA ===")
    print("==================================================\n")

    # 1. Load Configurations & Artifacts
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_path = config.get('paths', {}).get('train_processed', config.get('train_processed'))

    print(f"Loading hyperparameters from: {params_path}")
    with open(params_path, 'r') as f:
        best_params = yaml.safe_load(f)

    print(f"Loading feature subset from: {feature_manifest_path}")
    with open(feature_manifest_path, 'r') as f:
        feature_manifest = yaml.safe_load(f)
    selected_features = feature_manifest['selected_features']

    # 2. Load Dataset
    print(f"Loading engineered dataset from: {train_path}")
    df = pd.read_parquet(train_path)
    target_col = 'is_churn'

    # Filter contradictions
    if 'hist_transaction_count' in df.columns and 'hist_ex_latest_mean_actual_amount_paid' in df.columns:
        initial_len = len(df)
        contradiction_mask = (df['hist_transaction_count'] == 0) & (df['hist_ex_latest_mean_actual_amount_paid'] > 0)
        df = df[~contradiction_mask].reset_index(drop=True)
        print(f"Filtered out {initial_len - len(df):,} contradiction rows.")

    # Enforce Categorical Types
    cat_cols = ['city', 'registered_via', 'gender', 'payment_method_mode']
    for col in cat_cols:
        if col in selected_features and col in df.columns:
            df[col] = df[col].astype('category')

    selected_features = [f for f in selected_features if f in df.columns]
    best_params.pop('best_log_loss', None)

    # 3. Stage 1: Stratified Split Validation to Find Optimal Boost Rounds
    print("\n--- STAGE 1: Validating on 80/20 Stratified Split ---")
    X_tr, X_val, y_tr, y_val = train_test_split(
        df[selected_features],
        df[target_col].astype(int),
        test_size=0.20,
        random_state=42,
        stratify=df[target_col].astype(int)
    )

    val_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1500,
        early_stopping_rounds=50
    )

    val_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iteration = val_model.best_iteration
    val_preds = val_model.predict_proba(X_val)[:, 1]
    val_loss = log_loss(y_val, val_preds)

    print(f"Validation Log Loss: {val_loss:.5f}")
    print(f"Optimal Stopping Iteration: {best_iteration} trees")

    del X_tr, y_tr, X_val, y_val, val_model
    gc.collect()

    # 4. Stage 2: Retrain on Full Dataset (100% Data)
    print("\n--- STAGE 2: Retraining XGBoost on 100% Full Dataset ---")
    X_full = df[selected_features].copy()
    y_full = df[target_col].astype(int)

    del df
    gc.collect()

    final_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=best_iteration
    )

    final_model.fit(X_full, y_full, verbose=False)

    # 5. Save Final Model Artifact
    os.makedirs('models/artifacts', exist_ok=True)
    save_path = 'models/artifacts/final_xgb_model.pkl'
    joblib.dump(final_model, save_path)
    print(f"\nFinal XGBoost model trained on full dataset and saved to '{save_path}'!")


if __name__ == '__main__':
    train_final_xgboost()
