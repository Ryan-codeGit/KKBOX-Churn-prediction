import gc
import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def run_feature_optimization(config_path='config.yaml'):
    print("==================================================")
    print("=== STARTING 3-STEP FEATURE OPTIMIZATION ENGINE ===")
    print("==================================================\n")

    # 1. Load Configuration Path
    print(f"Loading configuration from '{config_path}'...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_path = config['paths']['train_processed']

    # 2. Load Processed Parquet Data (Features + Targets)
    print(f"Loading engineered dataset from: {train_path}")
    df = pd.read_parquet(train_path)

    target_col = 'is_churn'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    # 3. Explicitly Enforce 'category' Type for Categorical Features
    cat_cols = ['city', 'registered_via', 'gender', 'payment_method_mode']
    active_cats = []

    print("Enforcing 'category' pandas dtypes for categorical features...")
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            active_cats.append(col)

    # Define feature set excluding non-feature identifiers and targets
    exclude_cols = {'msno', target_col, 'registration_init_time', 'transaction_date', 'date'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Isolate Training & Validation Sets using STRATIFIED RANDOM SPLIT (Fixes account age bias)
    X_tr, X_val, y_tr, y_val = train_test_split(
        df[feature_cols],
        df[target_col].astype(int),
        test_size=0.20,
        random_state=42,
        stratify=df[target_col].astype(int)
    )

    del df
    gc.collect()

    initial_feature_count = len(feature_cols)
    current_features = list(feature_cols)

    # 4. Train Fast Baseline Model for Initial Diagnostics
    print("\n--- Training Fast Baseline Model for Initial Diagnostics ---")
    current_cats = [c for c in active_cats if c in current_features]

    train_data = lgb.Dataset(
        X_tr[current_features],
        label=y_tr,
        categorical_feature=current_cats,
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val[current_features],
        label=y_val,
        reference=train_data,
        categorical_feature=current_cats,
        free_raw_data=False,
    )

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 45,
        'max_depth': 8,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    base_val_preds = model.predict(X_val[current_features])
    base_loss = log_loss(y_val, base_val_preds)
    print(f"\nBaseline Full-Model Validation Log Loss: {base_loss:.5f}")

    # Top Features Diagnostic & Target Leakage Detection
    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'gain': model.feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False)

    print("\n--- TOP 10 FEATURES BY GAIN ---")
    print(importance_df.head(10).to_string(index=False))

    print("\n--- CHECKING SINGLE-FEATURE LEAKAGE ---")
    for feat in importance_df['feature'].head(10):
        single_train = lgb.Dataset(X_tr[[feat]], label=y_tr, free_raw_data=False)
        single_val = lgb.Dataset(X_val[[feat]], label=y_val, reference=single_train, free_raw_data=False)
        m = lgb.train(
            params,
            single_train,
            num_boost_round=100,
            valid_sets=[single_val],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        preds = m.predict(X_val[[feat]])
        loss = log_loss(y_val, preds)
        print(f"Feature: {feat:<40} | Standalone Log Loss: {loss:.5f}")

    # -------------------------------------------------------------
    # STEP 1: GAIN PRUNING (Drop Zero Gain)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("=== STEP 1: GAIN PRUNING (Zero-Gain Removal) ===")
    print("==================================================")

    zero_gain_cols = importance_df[importance_df['gain'] == 0]['feature'].tolist()
    current_features = [c for c in current_features if c not in zero_gain_cols]

    print(f"Features with 0 Gain: {len(zero_gain_cols)}")
    print(
        f"Step 1 Complete -> Remaining Features: {len(current_features)}"
        f" / {initial_feature_count}"
    )

    # -------------------------------------------------------------
    # STEP 2: CORRELATION PRUNING (> 0.98 Pearson Correlation)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("=== STEP 2: CORRELATION PRUNING (Threshold > 0.98) ===")
    print("==================================================")

    num_features_remaining = [
        c for c in current_features if c not in active_cats
    ]
    print(f"Computing correlation matrix across {len(num_features_remaining)} numerical features...")

    corr_matrix = X_tr[num_features_remaining].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    collinear_drops = [
        col for col in upper_tri.columns if any(upper_tri[col] > 0.98)
    ]
    current_features = [c for c in current_features if c not in collinear_drops]

    print(f"Identified Collinear Features (>0.98): {len(collinear_drops)}")
    print(
        f"Step 2 Complete -> Remaining Features: {len(current_features)}"
        f" / {initial_feature_count}"
    )

    # -------------------------------------------------------------
    # STEP 3: PERMUTATION IMPORTANCE CHECK (LOG LOSS)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("=== STEP 3: PERMUTATION IMPORTANCE CHECK (LOG LOSS) ===")
    print("==================================================")

    current_cats = [c for c in active_cats if c in current_features]

    train_data_pruned = lgb.Dataset(
        X_tr[current_features],
        label=y_tr,
        categorical_feature=current_cats,
        free_raw_data=False,
    )
    val_data_pruned = lgb.Dataset(
        X_val[current_features],
        label=y_val,
        reference=train_data_pruned,
        categorical_feature=current_cats,
        free_raw_data=False,
    )

    model_pruned = lgb.train(
        params,
        train_data_pruned,
        num_boost_round=300,
        valid_sets=[train_data_pruned, val_data_pruned],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    clean_base_preds = model_pruned.predict(X_val[current_features])
    clean_base_loss = log_loss(y_val, clean_base_preds)
    print(f"Pruned Baseline Validation Log Loss: {clean_base_loss:.5f}")

    noisy_features = []
    np.random.seed(42)

    print("Evaluating permutation impact per feature on validation set...")
    for idx, col in enumerate(current_features, start=1):
        X_val_permuted = X_val[current_features].copy()

        if col in current_cats:
            orig_type = X_val_permuted[col].dtype
            shuffled_vals = np.random.permutation(X_val_permuted[col].values)
            X_val_permuted[col] = pd.Series(shuffled_vals, index=X_val_permuted.index).astype(orig_type)
        else:
            X_val_permuted[col] = np.random.permutation(X_val_permuted[col].values)

        perm_preds = model_pruned.predict(X_val_permuted)
        perm_loss = log_loss(y_val, perm_preds)

        loss_increase = perm_loss - clean_base_loss

        if loss_increase <= 0:
            noisy_features.append(col)

        if idx % 50 == 0 or idx == len(current_features):
            print(
                f" Evaluated {idx}/{len(current_features)} features | Current Noisy"
                f" Count: {len(noisy_features)}"
            )

    final_selected_features = [
        c for c in current_features if c not in noisy_features
    ]

    print("\n==================================================")
    print("=== FEATURE OPTIMIZATION SUMMARY ===")
    print("==================================================")
    print(f"Initial Features:               {initial_feature_count}")
    print(f"Zero Gain Features Removed:     {len(zero_gain_cols)}")
    print(f"Collinear Features Removed:     {len(collinear_drops)}")
    print(f"Noisy Permutation Features:     {len(noisy_features)}")
    print(f"Final Selected Features:        {len(final_selected_features)}")
    print("==================================================")

    # Save optimized feature manifest
    os.makedirs('models/artifacts', exist_ok=True)
    output_yaml = 'models/artifacts/optimized_dropped_features.yaml'

    with open(output_yaml, 'w') as f:
        yaml.dump({'selected_features': final_selected_features}, f)

    print(f"\nOptimized feature list successfully saved to '{output_yaml}'!")


if __name__ == '__main__':
    run_feature_optimization()
