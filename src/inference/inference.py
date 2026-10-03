import os
import gc
import joblib
import numpy as np
import pandas as pd
import yaml

# Explicit list of categorical features in the dataset
CATEGORICAL_COLS = ['city', 'registered_via', 'gender', 'payment_method_mode']


def load_blend_weights(model_dir: str) -> tuple[float, float]:
    """Loads pre-calculated blend weights directly from models/artifacts/blend_config.yaml."""
    yaml_path = os.path.join(model_dir, "blend_config.yaml")
    print(f"Loading blend configuration from '{yaml_path}'...")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve nested dictionary if weights are wrapped under a sub-key
    weights = config
    for key in ["weights", "blend_weights", "model_weights"]:
        if isinstance(config, dict) and key in config and isinstance(config[key], dict):
            weights = config[key]
            break

    # Extract explicit keys: lgb_weight and xgb_weight
    lgb_w = weights.get("lgb_weight")
    xgb_w = weights.get("xgb_weight")

    if lgb_w is None or xgb_w is None:
        raise KeyError(
            f"Failed to find 'lgb_weight' or 'xgb_weight' in '{yaml_path}'. "
            f"Loaded structure keys: {list(weights.keys()) if isinstance(weights, dict) else weights}"
        )

    return float(lgb_w), float(xgb_w)


def load_selected_features(model_dir: str) -> list[str]:
    """Loads list of selected features from models/artifacts/optimized_dropped_features.yaml."""
    yaml_path = os.path.join(model_dir, "optimized_dropped_features.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Selected features manifest not found at '{yaml_path}'. Run feature pruning first.")

    print(f"Loading selected features list from '{yaml_path}'...")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    selected = config.get("selected_features", [])
    if not selected:
        raise KeyError(f"'selected_features' key is missing or empty in '{yaml_path}'.")

    print(f"Loaded {len(selected)} selected features.")
    return list(selected)


def load_processed_test_data(test_path: str) -> pd.DataFrame:
    """Loads the cleaned test dataset."""
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test dataset not found at '{test_path}'. Run preprocessing first."
        )

    print(f"Loading test data from '{test_path}'...")
    df = pd.read_parquet(test_path)
    print(f"Loaded test dataset shape: {df.shape}")
    return df


def prepare_features(X_test: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Selects target features and enforces category dtype on categorical columns."""
    # Retain only selected features present in test data
    cols_to_use = [c for c in selected_features if c in X_test.columns]
    X_prep = X_test[cols_to_use].copy()
    print(f"Features shape after applying feature manifest: {X_prep.shape}")

    # Enforce category dtype for categorical columns
    for col in CATEGORICAL_COLS:
        if col in X_prep.columns:
            X_prep[col] = X_prep[col].astype("category")

    return X_prep


def generate_predictions_and_submission(
    test_path: str = "data/processed/test_df_processed.parquet",
    model_dir: str = "models/artifacts",
    output_dir: str = "submissions",
    submission_filename: str = "submission.csv",
):
    """Loads test features, applies selected features and categoricals, runs LightGBM and XGBoost,
    and outputs the blended submission file.
    """
    df_test = load_processed_test_data(test_path)

    if "msno" not in df_test.columns:
        raise KeyError("'msno' column missing from test dataset.")

    msno = df_test["msno"]
    X_raw = df_test.drop(columns=["msno", "is_churn"], errors="ignore")

    # Load selected features and prepare test matrix
    selected_features = load_selected_features(model_dir)
    X_test = prepare_features(X_raw, selected_features)

    # Load deterministic blend ratio
    lgb_weight, xgb_weight = load_blend_weights(model_dir)

    print(
        f"Applying blend ratio -> LightGBM: {lgb_weight:.4f}, XGBoost: {xgb_weight:.4f}"
    )

    # -------------------------------------------------------------------------
    # MODEL INFERENCE & BLENDING
    # -------------------------------------------------------------------------
    lgb_model_path = os.path.join(model_dir, "final_lgbm_model.pkl")
    xgb_model_path = os.path.join(model_dir, "final_xgb_model.pkl")

    predictions = []
    weights = []

    # 1. LightGBM Prediction
    if os.path.exists(lgb_model_path):
        print(f"Generating predictions from LightGBM ({lgb_model_path})...")
        lgb_model = joblib.load(lgb_model_path)

        if hasattr(lgb_model, "predict_proba"):
            lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
        else:
            lgb_preds = lgb_model.predict(X_test)

        predictions.append(lgb_preds)
        weights.append(lgb_weight)
    else:
        print(f"Warning: LightGBM model not found at '{lgb_model_path}'.")

    # 2. XGBoost Prediction
    if os.path.exists(xgb_model_path):
        print(f"Generating predictions from XGBoost ({xgb_model_path})...")
        xgb_model = joblib.load(xgb_model_path)

        if hasattr(xgb_model, "predict_proba"):
            xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
        else:
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test, enable_categorical=True)
            xgb_preds = xgb_model.predict(dtest)

        predictions.append(xgb_preds)
        weights.append(xgb_weight)
    else:
        print(f"Warning: XGBoost model not found at '{xgb_model_path}'.")

    if not predictions:
        raise FileNotFoundError(
            f"Neither LightGBM nor XGBoost models were found in '{model_dir}'."
        )

    # Normalize weights across available models
    normalized_weights = np.array(weights) / np.sum(weights)
    final_preds = np.zeros(len(X_test), dtype=np.float32)

    for pred, weight in zip(predictions, normalized_weights):
        final_preds += pred * weight

    # -------------------------------------------------------------------------
    # SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    print("Formatting submission matrix...")
    sub_df = pd.DataFrame({"msno": msno, "is_churn": final_preds})

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, submission_filename)

    print(f"Saving final submission to '{out_file}'...")
    sub_df.to_csv(out_file, index=False)

    print(
        "\n========================================= SUBMISSION SUMMARY"
        " ========================================="
    )
    print(f"Rows outputted : {len(sub_df):,}")
    print(f"Churn mean     : {sub_df['is_churn'].mean():.4f}")
    print(
        f"Churn min/max  : {sub_df['is_churn'].min():.4f} /"
        f" {sub_df['is_churn'].max():.4f}"
    )
    print(
        "======================================================================================================\n"
    )

    gc.collect()
    return sub_df


if __name__ == "__main__":
    generate_predictions_and_submission()
