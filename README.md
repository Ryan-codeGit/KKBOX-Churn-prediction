# KKBox Churn Prediction

End-to-end machine learning pipeline for predicting customer churn in the **KKBox Churn Prediction Challenge**.

The project combines transaction history, user activity aggregates, and member/account information to estimate the probability that a subscriber will churn after their membership expires.

The final solution uses **XGBoost + LightGBM**, with engineered user-level features, feature pruning, stratified validation, out-of-fold predictions, and probability blending.

---

## Results

| Metric              |                         Result |
| ------------------- | -----------------------------: |
| Public leaderboard  |                      **0.117** |
| Private leaderboard |                      **0.116** |
| Final ensemble      | **50% LightGBM + 50% XGBoost** |

The competition evaluates predicted churn probabilities using **Log Loss**.

---

## Project Objective

The task is to predict whether a KKBox subscriber will churn after their current membership expires.

For this implementation, churn is evaluated using the 30-day renewal window after membership expiration. The competition's train/test cohorts are separated by membership-expiration month.

This project builds a complete pipeline around that prediction problem:

```text
Raw KKBox Data
      │
      ├── Member Information
      ├── Transaction History
      ├── User Activity Logs
      └── Competition Targets
             │
             ▼
      Target Auditing / Purification
             │
             ▼
      User-Level Feature Engineering
             │
             ▼
      Dataset Construction
             │
             ▼
      Feature Pruning
             │
             ▼
      ┌───────────────────────┐
      │                       │
      ▼                       ▼
   XGBoost                LightGBM
      │                       │
      └───────────┬───────────┘
                  ▼
          OOF Predictions
                  │
                  ▼
        Blend Weight Optimization
                  │
                  ▼
          50/50 Final Ensemble
                  │
                  ▼
          April Cohort Predictions
                  │
                  ▼
           Competition Submission
```

---

## Dataset

The project uses the main KKBox competition data sources:

* **`members_v3.csv`** — member/account information such as city, age, gender, registration method, and registration date.
* **`transactions.csv`** — historical subscription transactions and payment information.
* **`transactions_v2.csv`** — additional transaction history.
* **User activity logs** — daily listening behaviour, reduced to user-level aggregate features before model training.
* **`train_v2.csv`** — competition training labels.
* **`sample_submission_v2.csv`** — competition submission format.

The raw datasets are not included in this repository.

---

## Target Construction

The project does not rely blindly on the supplied target labels.

`src/data/build_targets.py` audits the competition labels against transaction history and constructs a purified target.

The process examines:

* Previous membership expiration
* Subsequent transaction dates
* The gap between expiration and the next transaction
* Cancellation status
* The 30-day renewal window

Two types of contradictory observations are corrected.

### Invisible renewals

Users originally labelled as churned are changed to retained when transaction history shows a qualifying renewal within the 30-day window and the corresponding cancellation condition indicates an active renewal.

### Ghost active users

Users originally labelled as retained are changed to churned when their transaction history indicates that the gap after membership expiration is at least 30 days.

The resulting target is saved locally as:

```text
data/processed/train_v2_purified.csv
```

This target-auditing step was important because the KKBox subscription structure makes cancellation and churn distinct concepts. A cancellation does not necessarily mean that a customer has churned.

---

## Cohort Construction

The project builds separate training and test datasets around the competition's monthly expiration cohorts.

### Training

The training cohort represents users whose relevant membership expiration falls in **March 2017** in this implementation.

Training features are constructed using information available through:

```text
2017-02-28
```

### Test

The test cohort represents the subsequent **April 2017** prediction cohort.

Test features are constructed using information available through:

```text
2017-03-31
```

This creates a chronological feature-building process in which the test feature set is generated separately from the training feature set.

---

## Feature Engineering

The feature pipeline combines three major information sources.

### 1. Transaction Features

Transaction history is processed into user-level historical statistics.

These capture aspects of:

* Subscription history
* Payment behaviour
* Renewal behaviour
* Cancellation behaviour
* Historical transaction volume
* Membership lifecycle
* Payment amounts
* Plan characteristics
* Recency and historical activity

Transaction processing is implemented in:

```text
src/data/process_transaction_data.py
src/data/process_transaction_data_test.py
```

---

### 2. User Activity Features

The original user activity data is too large to use directly as a raw modelling table.

Instead, activity information is aggregated into user-level behavioural features before being joined to the modelling dataset.

The resulting features capture aspects of:

* Listening activity
* Active days
* Usage volume
* Song engagement
* Recent activity
* Historical activity
* Behavioural changes over time

The feature construction is handled through the user-log SQL and feature-building pipeline.

---

### 3. Member Features

Member/account information is merged into the cohort dataset.

The modelling pipeline explicitly handles the following categorical features:

```text
city
registered_via
gender
payment_method_mode
```

These are passed to the tree-based models using their appropriate categorical representations.

---

## Dataset Construction

The training and test pipelines independently perform the following sequence:

```text
Target Cohort
     │
     ├── Member Data
     ├── Transaction Features
     └── User Activity Features
             │
             ▼
       User-Level Dataset
             │
             ▼
       Feature Interactions
             │
             ▼
       Processed Dataset
```

Training:

```text
src/pipelines/build_dataset_train.py
```

Test:

```text
src/pipelines/build_dataset_test.py
```

The resulting datasets are generated locally under:

```text
data/processed/
```

They are excluded from version control because of their size.

---

## Feature Selection

Feature pruning is performed after dataset construction.

The process is implemented in:

```text
src/optimization/feature_pruning.py
```

The pruning pipeline uses a LightGBM validation model to identify unnecessary features.

### Step 1 — Zero-Gain Removal

Features producing no useful model gain are removed.

### Step 2 — Correlation Filtering

Highly correlated features are evaluated using Pearson correlation.

Features exceeding the configured correlation threshold are candidates for removal to reduce redundancy.

### Step 3 — Permutation Importance

Remaining features are evaluated using permutation importance on the validation set.

Features that do not improve validation log loss when retained are treated as noise and removed.

The resulting feature manifest is saved locally as:

```text
models/artifacts/optimized_dropped_features.yaml
```

Both final models and inference use this same feature manifest.

---

## Validation Strategy

Two different validation concepts are used in the project.

### Internal Model Validation

For model development, the training data is split using a stratified:

```text
80% training
20% validation
```

with:

```text
random_state = 42
```

This split is used to:

* evaluate model configurations
* determine suitable boosting rounds
* perform early stopping
* compare model behaviour

### Out-of-Fold Validation

For ensemble construction, the project uses:

```text
5-fold StratifiedKFold
shuffle=True
random_state=42
```

Each fold produces validation predictions that are stored as OOF predictions.

The OOF predictions are generated locally during the ensemble stage and are not included in the repository because of their size.
The final 50/50 LightGBM–XGBoost blend achieved an OOF log loss of 0.13861 on the internal 5-fold validation setup.
---

## Final Models

The final ensemble consists of two gradient-boosted tree models.

### XGBoost

Implemented in:

```text
src/train/xgboost_final.py
```

The training process:

1. Loads optimized hyperparameters.
2. Loads the selected feature manifest.
3. Loads the processed training dataset.
4. Performs an 80/20 stratified validation split.
5. Uses early stopping to determine the appropriate number of boosting rounds.
6. Retrains on the complete training dataset.
7. Saves the final model locally.

The resulting model artifact is excluded from version control.

---

### LightGBM

Implemented in:

```text
src/train/lightgbm_final.py
```

The training process follows the same overall structure:

1. Load optimized hyperparameters.
2. Load selected features.
3. Perform an 80/20 stratified validation split.
4. Use early stopping to determine the boosting horizon.
5. Retrain using the complete training dataset.
6. Save the final model locally.

The resulting model artifact is excluded from version control.

---

## Out-of-Fold Ensemble Construction

OOF predictions are generated by:

```text
src/train/oof_predictions.py
```

Each fold trains both:

* LightGBM
* XGBoost

and produces predictions for the held-out fold.

The resulting prediction matrix contains:

```text
target
oof_lgb
oof_xgb
```

These predictions are used to determine how the two models should be combined without evaluating the blend on predictions generated from models trained directly on those same observations.

---

## Blend Optimization

Blend optimization is implemented in:

```text
src/optimization/blend_optimiser.py
```

The optimizer minimizes OOF Log Loss subject to:

```text
0 <= weight <= 1
```

and:

```text
LightGBM weight + XGBoost weight = 1
```

The resulting configuration is stored in:

```text
models/artifacts/blend_config.yaml
```

The final configuration is:

```yaml
lgb_weight: 0.5
xgb_weight: 0.5
oof_log_loss: 0.13861391437121212
```

Therefore the final submission uses an equal-weight ensemble:

```text
Prediction =
    0.50 × LightGBM probability
  + 0.50 × XGBoost probability
```

---

## Inference

The final inference pipeline is:

```text
src/inference/inference.py
```

It performs the following steps:

1. Load the processed April test dataset.
2. Load the selected feature manifest.
3. Apply the same categorical feature handling used during training.
4. Load the final LightGBM model.
5. Load the final XGBoost model.
6. Generate churn probabilities from both models.
7. Apply the stored blend weights.
8. Produce the final submission dataframe.
9. Save the submission CSV.

The output format is:

```text
msno,is_churn
user_id,predicted_probability
```

The submission is generated locally at:

```text
submissions/submission.csv
```

The generated submission is excluded from version control because of its size.

---

## Repository Structure

The GitHub repository contains the source code, configuration, modelling metadata, SQL feature-engineering scripts, and analysis notebooks.

```text
.
├── .gitignore
├── README.md
├── config.yaml
├── requirements.txt
│
├── models
│   └── artifacts
│       ├── best_lgbm_params.yaml
│       ├── best_xgb_params.yaml
│       ├── blend_config.yaml
│       ├── lgbm_model_results.yaml
│       ├── optimized_dropped_features.yaml
│       └── xgboost_model_results.yaml
│
├── notebooks
│   ├── feature_importance.ipynb
│   ├── full_dataset_sanity_check.ipynb
│   ├── member.ipynb
│   ├── residual_analysis.ipynb
│   ├── train_dataset_check.ipynb
│   ├── transactions.ipynb
│   └── user_logs.ipynb
│
├── src
│   ├── data
│   │   ├── build_targets.py
│   │   ├── build_user_log_features_test.sql
│   │   ├── build_user_log_features_train.sql
│   │   ├── full_dataset_features.py
│   │   ├── preprocessing.py
│   │   ├── preprocessing_test.py
│   │   ├── process_transaction_data.py
│   │   ├── process_transaction_data_test.py
│   │   └── transaction_join.py
│   │
│   ├── diagnostics
│   │   └── target_gen.py
│   │
│   ├── inference
│   │   └── inference.py
│   │
│   ├── optimization
│   │   ├── blend_optimiser.py
│   │   ├── feature_pruning.py
│   │   ├── lightgbm_tuning.py
│   │   └── xgboost_tuning.py
│   │
│   ├── pipelines
│   │   ├── build_dataset_test.py
│   │   └── build_dataset_train.py
│   │
│   └── train
│       ├── lightgbm_final.py
│       ├── oof_predictions.py
│       └── xgboost_final.py
│
└── notebooks/
```

Raw datasets, processed datasets, trained model binaries, OOF prediction files, and the final competition submission are intentionally excluded from version control.

---

## Key Engineering Decisions

### User-level aggregation

Large-scale activity data is transformed into compact user-level features before entering the modelling stage rather than attempting to train directly on raw event records.

### Target auditing

The supplied labels are checked against transaction history so that the target-generation logic is consistent with the 30-day renewal definition.

### Separate cohort pipelines

Training and test datasets are constructed independently with different reference dates, reflecting the temporal structure of the competition.

### Feature pruning

Feature selection is performed systematically rather than relying solely on model feature importance.

### Out-of-fold blending

The final ensemble weights are determined from out-of-fold predictions rather than simply averaging arbitrary model outputs.

### Full-data retraining

After validation determines the boosting horizon, each final model is retrained using the complete available training dataset.

---

## Running the Pipeline

The project is structured around separate stages.

### Build the training dataset

```bash
python src/pipelines/build_dataset_train.py
```

### Build the test dataset

```bash
python src/pipelines/build_dataset_test.py
```

### Run feature pruning

```bash
python src/optimization/feature_pruning.py
```

### Train final XGBoost model

```bash
python src/train/xgboost_final.py
```

### Train final LightGBM model

```bash
python src/train/lightgbm_final.py
```

### Generate OOF predictions

```bash
python src/train/oof_predictions.py
```

### Optimize ensemble weights

```bash
python src/optimization/blend_optimiser.py
```

### Generate competition submission

```bash
python src/inference/inference.py
```

The generated submission is written locally to:

```text
submissions/submission.csv
```

---

## Outcome

This project resulted in a complete competition-oriented churn prediction workflow covering:

* target reconstruction and auditing
* large-scale user-level feature engineering
* transaction history processing
* behavioural feature aggregation
* feature pruning
* hyperparameter optimization
* stratified validation
* out-of-fold prediction generation
* ensemble weight optimization
* full-data model retraining
* reproducible inference
* competition submission generation

Final leaderboard performance:

**Public Log Loss: 0.117**

**Private Log Loss: 0.116**
