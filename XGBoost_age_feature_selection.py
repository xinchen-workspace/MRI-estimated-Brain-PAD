#!/usr/bin/env python3
# coding: utf-8

"""
Feature Selection Script (Direct Path Version).

This script identifies the best features based on a specific importance file
and then filters a raw data CSV to include only metadata and those features.
"""

import argparse
import sys
import pandas as pd

__author__ = "Xinjie Chen, Po-Jui Lu (primary developer)"
__email__ = "xinjie.chen@unibas.ch, p.lu@unibas.ch"

def get_selected_feature_list(importance_file_path):
    """
    Determines the list of important features based on a specific 
    importance/permutation results CSV file.
    """
    try:
        # Load the specific importance file
        print(f"Loading importance data from: {importance_file_path}")
        df_XG_results = pd.read_csv(importance_file_path)

        # Logic from original script: Remove first column (assumed to be index/names)
        df_XG_results = df_XG_results.iloc[:, 1:]

        # Calculate threshold (1% of the maximum importance found at index 0,0)
        max_importance = df_XG_results.iloc[0, 0]
        thr = 0.01
        imp_thr = max_importance * thr

        # Filter features where importance is >= threshold
        selected_feature_list = df_XG_results.loc[
            :, df_XG_results.iloc[0, :] >= imp_thr
        ].columns.to_list()
        
        print(f"Selected {len(selected_feature_list)} features based on threshold {imp_thr:.4f}")
        return selected_feature_list

    except FileNotFoundError:
        sys.exit(f"Error: Importance file '{importance_file_path}' not found.")
    except Exception as e:
        sys.exit(f"An error occurred during feature selection: {e}")


def filter_and_save_data(input_csv, output_csv, selected_features):
    """
    Loads raw data, keeps metadata + selected features, and saves to new CSV.
    """
    try:
        df_data = pd.read_csv(input_csv)

        # Assuming first 5 columns are metadata (ID, Age, Sex, Site, etc.)
        metadata_columns = df_data.columns.tolist()[:5]

        # Verify selected features exist in the data file
        valid_features = [f for f in selected_features if f in df_data.columns]
        
        missing_count = len(selected_features) - len(valid_features)
        if missing_count > 0:
            print(f"Warning: {missing_count} selected features were not found in the data file.")

        # Combine metadata and valid features
        final_columns = metadata_columns + valid_features
        df_filtered = df_data[final_columns]

        # Save
        df_filtered.to_csv(output_csv, index=False)
        print(f"Successfully saved filtered data to: {output_csv}")

    except FileNotFoundError:
        sys.exit(f"Error: Input data file '{input_csv}' not found.")


def build_argparser():
    p = argparse.ArgumentParser(description="Filter brain morphometry data using a specific importance file.")
    
    p.add_argument("--data_file", required=True, 
                   help="The raw csv file containing ID, Age, and all brain morphometrics.")
    
    p.add_argument("--imp_file", required=True, 
                   help="The specific importance/permutation CSV file to use for selection.")
    
    p.add_argument("--output_file", required=True, 
                   help="The filename for the new filtered output csv file.")

    return p


def main():
    args = build_argparser().parse_args()

    # 1. Get list of features directly from the importance file
    features = get_selected_feature_list(args.imp_file)

    # 2. Filter data file and save
    filter_and_save_data(args.data_file, args.output_file, features)


if __name__ == "__main__":
    main()