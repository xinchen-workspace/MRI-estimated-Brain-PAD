import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

__author__ = "Xinjie Chen, Po-Jui Lu (primary developer)"
__email__ = "xinjie.chen@unibas.ch, p.lu@unibas.ch"

# -----------------------------------------------------------------------------
# 1. Setup Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. Data Processing Functions
# -----------------------------------------------------------------------------
def preproc_data(file_path, feature_columns=None, return_ID=False):
    """Load and preprocess the CSV data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df_data = pd.read_csv(file_path)
    
    # Default features: All columns from index 5 onwards
    if not feature_columns:
        feature_columns = df_data.columns.to_list()[5:]

    # Clean missing values
    df_data.dropna(axis=1, how="all", inplace=True)
    df_data.dropna(axis=0, how="all", inplace=True)
    df_data.fillna(0, inplace=True)

    target = "Age"
    if "batch_vector" in df_data.columns:
        df_data.rename(columns={"batch_vector": "site_ind"}, inplace=True)
    
    # Create Stratification Labels (Age + Site + Sex)
    try:
        df_data["Age_category"] = df_data["Age"] // 10
        strat_label = (df_data["Age_category"].values.astype(int) + 
                       df_data["site_ind"].values.astype(int) * 10 + 
                       (df_data["Sex"].values.astype(int) + 1) * 100)
    except KeyError as e:
        logger.error(f"Missing column required for stratification: {e}")
        raise

    X_data = df_data.loc[:, feature_columns].values
    Y_data = df_data.loc[:, target].values

    if return_ID:
        return feature_columns, strat_label, X_data, Y_data, df_data["ID"].values
    return feature_columns, strat_label, X_data, Y_data

def correct_age(Y_data, Y_predict):
    """
    Correct predicted ages based on the relationship between actual and predicted ages.

    Parameters:
    Y_data (ndarray): Array of actual ages.
    Y_predict (ndarray): Array of predicted ages.

    Returns:
    The uncorrected residue, correction term coefficients, and corrected ages.
    """
    # Calculate Brain Age Gap (BAG) as the difference between predicted ages and actual ages.
    BAG = Y_predict - Y_data
    # Fit a linear model to the relationship between actual ages and BAG.
    z = np.polyfit(Y_data, BAG, 1)
    # Calculate the uncorrected residue by removing the linear trend from BAG.
    Y_uncorrected_residue = BAG - (z[0] * Y_data + z[1])
    # Fit a linear model to the relationship between actual ages and predicted ages.
    correction_term = np.polyfit(Y_data, Y_predict, 1)
    # Correct the predicted ages using the correction term.
    Y_corrected = Y_predict + (Y_data - (correction_term[0] * Y_data + correction_term[1]))
    return Y_uncorrected_residue,correction_term,Y_corrected

# -----------------------------------------------------------------------------
# 3. Training & Search Logic
# -----------------------------------------------------------------------------

def run_grid_search(file_path, grid_params, feature_columns=None):
    """Performs Grid Search to find best hyperparameters."""
    logger.info("--- Starting Grid Search ---")
    _, strat_label, X_data, Y_data = preproc_data(file_path, feature_columns=feature_columns)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_ind_list = [a for a in skf.split(strat_label, strat_label)]

    bst = XGBRegressor(objective='reg:squarederror', seed=42)
    
    search = GridSearchCV(
        estimator=bst,
        param_grid=grid_params,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, 
        cv=cv_ind_list,
        refit=False,
        verbose=1
    )
    search.fit(X_data, Y_data)
    
    logger.info(f"Grid Search Best Params: {search.best_params_}")
    logger.info(f"Grid Search Best Score: {search.best_score_}")
    return search.best_params_

def train_final_models(file_path, output_dir, param_dict, feature_columns=None):
    """Trains the final model using 5-fold CV and saves results."""
    logger.info("--- Starting Final Training Loop ---")
    feature_columns, strat_label, X_data, Y_data, ID_data = preproc_data(file_path, feature_columns, return_ID=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_ind_list = [a for a in skf.split(strat_label, strat_label)]

    rmse_arr = np.zeros(5)
    mae_arr = np.zeros(5)
    
    # Directories
    model_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    all_val_ids = []
    all_val_actual = []
    all_val_preds = []
    # Cross Validation Loop
    for cv_ind, (tr_ind, val_ind) in enumerate(cv_ind_list):
        X_train, Y_train = X_data[tr_ind, :], Y_data[tr_ind]
        X_val, Y_val = X_data[val_ind, :], Y_data[val_ind]
        split_info = {
            "train_ids": ID_data[tr_ind].tolist(),
            "val_ids": ID_data[val_ind].tolist()
        }
        split_file = os.path.join(splits_dir, f"cv{cv_ind}_ids.json")
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=4)
        logger.info(f"Fold {cv_ind}: Split IDs saved to {split_file}")

        # Train
        bst = XGBRegressor(objective='reg:squarederror', seed=42, **param_dict)
        bst.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
        
        # Save
        bst.save_model(os.path.join(model_dir, f"best_model_cv{cv_ind}.json"))
        
        # Evaluate
        preds = bst.predict(X_val)
        
        all_val_ids.extend(ID_data[val_ind])
        all_val_actual.extend(Y_val)
        all_val_preds.extend(preds)

        rmse_arr[cv_ind] = ((preds - Y_val)**2).mean()**(1/2)
        mae_arr[cv_ind] = np.abs(preds - Y_val).mean()
        
        logger.info(f"Fold {cv_ind}: RMSE={rmse_arr[cv_ind]:.4f}, MAE={mae_arr[cv_ind]:.4f}")
        
        # Permutation Importance (Simplified for speed in this example)
        # Note: In production, consider moving this outside the main loop if it's slow
        results = permutation_importance(bst, X_train, Y_train, n_repeats=5, random_state=123, scoring="neg_mean_squared_error")
        perm_df = pd.DataFrame(
            data=np.stack([results.importances_mean, results.importances_std], axis=0),
            columns=feature_columns, index=["mean", "std"]
        ).sort_values(axis=1, by="mean", ascending=False)
        perm_df.to_csv(os.path.join(results_dir, f"permutation_cv{cv_ind}.csv"))

    logger.info("--- Performing Global Age Correction ---")
    
    Y_full_actual = np.array(all_val_actual)
    Y_full_predict = np.array(all_val_preds)
    
    # Calculate correction based on all folds combined
    _, correction_term, Y_full_corrected = correct_age(Y_full_actual, Y_full_predict)
    
    # Save Correction Parameters to JSON
    correction_params = {
        "correction_slope": float(correction_term[0]),
        "correction_intercept": float(correction_term[1])
    }
    
    corr_file_path = os.path.join(results_dir, "correction_params.json")
    with open(corr_file_path, "w") as f:
        json.dump(correction_params, f, indent=4)    
    
    # Save the actual, predicted, and corrected ages to a CSV
    pd.DataFrame({
        "ID": all_val_ids,
        "Age_Actual": Y_full_actual,
        "Age_Predicted": Y_full_predict,
        "Age_Corrected": Y_full_corrected
    }).to_csv(os.path.join(results_dir, "final_predictions.csv"), index=False)

    # Save Summary
    pd.DataFrame({"CV": range(5), "RMSE": rmse_arr, "MAE": mae_arr}).to_csv(
        os.path.join(results_dir, "performance_summary.csv"), index=False
    )
    logger.info("Training finished.")

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="XGBoost Brain Age: Grid Search & Train")
    
    # Main Flags
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', type=str, default="./output", help="Directory to save results")
    parser.add_argument('--search', action='store_true', help="If set, run Grid Search first. If not, use provided args directly.")

    # Hyperparameters (accepts 1 or more values)
    parser.add_argument('--n_estimators', type=int, nargs='+', default=[450], help="Value(s) for n_estimators")
    parser.add_argument('--max_depth', type=int, nargs='+', default=[2], help="Value(s) for max_depth")
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[0.05], help="Value(s) for learning_rate")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Determine Parameters (Search vs Manual)
    # ---------------------------------------------------------
    final_params = {}

    if args.search:
        # 
        # Collect lists for grid search
        grid_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate
        }
        # Run Search
        best_found_params = run_grid_search(args.input, grid_params)
        
        # Update final params with best found results
        final_params.update(best_found_params)
        
        # Save these for record
        with open(os.path.join(args.output_dir, "best_search_params.json"), "w") as f:
            json.dump(final_params, f, indent=4)
            
    else:
        # User did NOT request search.
        # We must ensure they provided only SINGLE values for training.
        if any(len(x) > 1 for x in [args.n_estimators, args.max_depth, args.learning_rate]):
            logger.error("ERROR: You provided multiple parameter values but did not use the --search flag.")
            logger.error("Action: Add --search to run a grid search, or provide single values for direct training.")
            sys.exit(1)
            
        final_params['n_estimators'] = args.n_estimators[0]
        final_params['max_depth'] = args.max_depth[0]
        final_params['learning_rate'] = args.learning_rate[0]
        logger.info(f"Using manual parameters: {final_params}")

    # Add static parameters required for training
    final_params['early_stopping_rounds'] = 100

    # ---------------------------------------------------------
    # STEP 2: Train Model with Final Parameters
    # ---------------------------------------------------------
    train_final_models(args.input, args.output_dir, final_params)

    # ---------------------------------------------------------
    

if __name__ == "__main__":
    main()