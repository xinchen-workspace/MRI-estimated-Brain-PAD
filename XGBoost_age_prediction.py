# coding: utf-8

"""
Brain Age Prediction and Correction Script.

This script loads brain morphometric data, predicts brain age using a pre-trained
XGBoost model, applies a linear bias correction derived from healthy controls,
visualizes the results, and saves the predictions to a CSV file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

__author__ = "Xinjie Chen, Po-Jui Lu (primary developer)"
__email__ = "xinjie.chen@unibas.ch, p.lu@unibas.ch"

def load_json(file_path: str) -> Dict[str, Any]:
    """Safely loads a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: The configuration file {file_path} was not found.")


def preproc_data(
    file_path: str, 
    feature_columns: Optional[List[str]] = None, 
    return_ID: bool = False):
    """
    Loads and preprocesses the brain morphometry data.
    """
    df_data = pd.read_csv(file_path)
    if feature_columns is None:
        if df_data.shape[1] > 5:
            feature_columns = df_data.columns.to_list()[5:]
        else:
            sys.exit("Error: Input CSV has fewer than 6 columns.")

    df_data.dropna(axis=1, how="all", inplace=True)
    df_data.dropna(axis=0, how="all", inplace=True)
    df_data.fillna(0, inplace=True)

    if "Age" not in df_data.columns:
        sys.exit("Error: Column 'Age' missing from input CSV.")
    target = "Age"
    X_data = df_data.loc[:, feature_columns].values
    Y_data = df_data.loc[:, target].values

    if return_ID:
        if "ID" not in df_data.columns:
            sys.exit("Error: Column 'ID' missing from input CSV.")
        return X_data, Y_data, df_data["ID"].values
    return  X_data, Y_data


def prediction_brain_age_MS(
    file_path,
    best_model_json_file,
    HC_prediction_json_file,
    output_csv_filename,
    param_dict,
):
    """
    Main pipeline: Loads data, runs inference, applies correction, plots, and saves.
    """
    selected_feature_list = pd.read_csv(file_path).columns.tolist()[5:]

    X_data, Y_data, ID_data = preproc_data(
        file_path, feature_columns=selected_feature_list, return_ID=True
    )

    Y_predict = np.zeros_like(Y_data)
    Y_corrected = np.zeros_like(Y_data)

    #Setup Model
    bst = XGBRegressor(objective="reg:squarederror", seed=42, **param_dict)
    bst.load_model(best_model_json_file)
    bst.set_params(n_estimators=bst.best_iteration)
    
    #Predict
    Y_predict_cv = bst.predict(X_data)
    correction_dict = load_json(HC_prediction_json_file)

    #Correct Bias
    Y_corrected_cv = Y_predict_cv + (
        Y_data
        - (correction_dict["correction_slope"] * Y_data + correction_dict["correction_intercept"])
    )

    Y_predict += Y_predict_cv
    Y_corrected += Y_corrected_cv

    #Visualize and SAVE
    fig, ax = plt.subplots(1, 2, figsize=(30, 20))
    ax = ax.ravel()
    ax[0].scatter(Y_data, Y_predict)
    ax[0].plot(np.arange(90), np.arange(90), c="black")
    ax[0].set_xlim(0, 90)
    ax[0].set_ylim(0, 90)

    ax[1].scatter(Y_data, Y_corrected)
    ax[1].plot(np.arange(90), np.arange(90), c="black")
    ax[1].set_xlim(0, 90)
    ax[1].set_ylim(0, 90)
    output_plot_filename = Path(output_csv_filename).with_suffix('.png')
    plt.savefig(output_plot_filename)
    print(f"Plot saved to {output_plot_filename}")
    plt.close()

    df_age = pd.DataFrame(
        {"ID": ID_data, "Age": Y_data, "Prediction": Y_predict, "Corrected_Age": Y_corrected}
    )
    df_age.to_csv(output_csv_filename, index=False)
    print(f"Results saved to {output_csv_filename}")


def build_argparser():
    p = argparse.ArgumentParser(description="Brain Age Prediction Script")
    p.add_argument("--file_path", help="CSV file containing 'ID', 'Age', and morphometrics (starting col 6).", required=True)
    p.add_argument("--best_model_json_file",help="JSON file of the best trained XGBoost model in cross-validation", required=True)
    p.add_argument("--HC_prediction_json_file", help="JSON file containing the slope and the intercept of the correction based on healthy controls", required=True)
    p.add_argument("--output_csv_filename", hep="Filename for the output CSV.",required=True)

    p.add_argument("--max_depth", type=int, default=2,help="XGBoost max_depth parameter.")
    p.add_argument("--learning_rate", type=float, default=0.05, help="XGBoost learning_rate parameter.")

    return p


def main():
    args = build_argparser().parse_args()

    param_dict = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }

    prediction_brain_age_MS(
        file_path=args.file_path,
        best_model_json_file=args.best_model_json_file,
        HC_prediction_json_file=args.HC_prediction_json_file,
        output_csv_filename=args.output_csv_filename,
        param_dict=param_dict,
    )


if __name__ == "__main__":
    main()
