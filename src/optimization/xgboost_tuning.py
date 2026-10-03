import gc
import os
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

optuna.logging.set_verbosity(optuna.logging.INFO)


def load_data_and_features(
    config_path='config.yaml',
    feature_manifest_path='models/artifacts/optimized_dropped_features.yaml',
):
    print(f"Loading configuration from '{config_path}'...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'paths' in config and 'train_processed' in config['paths']:
        train_path = config['paths']['train_processed']
    elif 'train_processed' in config:
        train_path = config['train_processed']
    else:
        raise KeyError("Could not find 'train_processed' in config.yaml")

    print(f"Loading engineered dataset from: {train_path}")
    df = pd.read_parquet(train_path)

    target_col = 'is_churn'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    if (
        'hist_transaction_count' in df.columns
        and 'hist_ex_latest_mean_actual_amount_paid' in df.columns
    ):
        initial_len = len(df)
        contradiction_mask = (df['hist_transaction_count'] == 0) & (
            df['hist_ex_latest_mean_actual_amount_paid'] > 0
        )
        df = df[~contradiction_mask].reset_index(drop=True)
        dropped_count = initial_len - len(df)
        print(f"Filtered out {dropped_count:,} contradiction rows.")

    print(f"Loading optimized features from: {feature_manifest_path}")
    with open(feature_manifest_path, 'r') as f:
        feature_manifest = yaml.safe_load(f)
    selected_features = feature_manifest['selected_features']

    cat_cols = ['city', 'registered_via', 'gender', 'payment_method_mode']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    selected_features = [f for f in selected_features if f in df.columns]

    return df, selected_features, target_col


def objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': 42,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = xgb.XGBClassifier(
        **params,
        n_estimators=1000,
        early_stopping_rounds=50,
        callbacks=[
            optuna.integration.XGBoostPruningCallback(trial, 'validation_0-logloss')
        ]
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, val_preds)
    return loss


def run_optuna_xgb_tuning(n_trials=50):
    print("==================================================")
    print("=== STARTING OPTUNA XGBOOST HYPERPARAMETER TUNER ===")
    print("==================================================\n")

    df, selected_features, target_col = load_data_and_features()

    # Stratified Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        df[selected_features],
        df[target_col].astype(int),
        test_size=0.20,
        random_state=42,
        stratify=df[target_col].astype(int)
    )

    del df
    gc.collect()

    print(f"\nDataset Configuration:")
    print(f"Features: {len(selected_features)}")
    print(f"Train set: {len(X_tr):,} rows | Validation set: {len(X_val):,} rows")
    print(f"Running Optuna tuning across {n_trials} trials...\n")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n==================================================")
    print("=== OPTUNA XGBOOST HYPERPARAMETER TUNING COMPLETE ===")
    print("==================================================")
    print(f"Best Validation Log Loss: {study.best_value:.5f}")
    print("\nBest Hyperparameters Found:")
    for key, val in study.best_params.items():
        print(f"  {key:<20}: {val}")
    print("==================================================")

    os.makedirs('models/artifacts', exist_ok=True)
    output_path = 'models/artifacts/best_xgb_params.yaml'

    best_config = study.best_params.copy()
    best_config.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': 42,
        'best_log_loss': float(study.best_value),
    })

    with open(output_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"\nBest XGBoost hyperparameter set saved to '{output_path}'!")


if __name__ == '__main__':
    run_optuna_xgb_tuning(n_trials=50)
