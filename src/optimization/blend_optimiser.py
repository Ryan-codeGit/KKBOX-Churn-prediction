import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def find_best_blend():
    print("==================================================")
    print("=== OPTIMIZING ENSEMBLE BLEND RATIO ===")
    print("==================================================\n")

    oof_df = pd.read_csv('models/artifacts/oof_predictions.csv')
    y_true = oof_df['target'].values
    p_lgb = oof_df['oof_lgb'].values
    p_xgb = oof_df['oof_xgb'].values

    # Objective Function: Minimize Log Loss with sum(weights) = 1 constraint
    def loss_func(weights):
        w_lgb, w_xgb = weights
        blend_preds = (w_lgb * p_lgb) + (w_xgb * p_xgb)
        return log_loss(y_true, blend_preds)

    # Initial guess & constraints
    initial_weights = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]
    constraints = {'type': 'eq', 'fun': lambda w: 1.0 - sum(w)}

    res = minimize(
        loss_func,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    w_lgb_opt, w_xgb_opt = res.x
    best_blend_loss = res.fun

    print("Optimization Complete!")
    print(f"Optimal LightGBM Weight: {w_lgb_opt:.4f}")
    print(f"Optimal XGBoost Weight:  {w_xgb_opt:.4f}")
    print(f"\nFinal Blended OOF Log Loss: {best_blend_loss:.5f}")

    # Save Blend Configuration
    blend_config = {
        'lgb_weight': float(w_lgb_opt),
        'xgb_weight': float(w_xgb_opt),
        'oof_log_loss': float(best_blend_loss)
    }

    with open('models/artifacts/blend_config.yaml', 'w') as f:
        yaml.dump(blend_config, f, default_flow_style=False)

    print("Blend metadata saved to 'models/artifacts/blend_config.yaml'!")

if __name__ == '__main__':
    find_best_blend()
