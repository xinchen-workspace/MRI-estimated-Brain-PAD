# MRI-estimated-Brain-PAD

This repository hosts the training pipeline for MRI(T1w)-based brain-age predictive model as reported in:
Chen, X., Lu, P. J., Ocampo-Pineda, M., Cagol, A., Schaedelin, S. A., Ruberte, E., … & Granziera, C. (2025). Unraveling microstructural and macrostructural brain age dynamics in multiple sclerosis. Neurology: Neuroimmunology & Neuroinflammation, 12(5), e200459.
The code covers data/feature setup, model fitting and validation, and the generation of predicted brain age and Brain-PAD (predicted age difference) calculation for downstream analyses.

## Prerequisites
* Python 3.x
* `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`

## Grid Search and Model Training
This script performs 5-fold cross-validation and can optionally search for the best hyperparameters.
```
python XGBoost_age_model_modelling.py \
    --input "filtered_features.csv" \
    --output_dir "./training_output" \
    --search \
    --n_estimators 100 300 500 \
    --max_depth 2 4 6 \
    --learning_rate 0.01 0.05
```
## Or Model Training with Specific Parameters
```
python XGBoost_age_model_modelling.py \
    --input "filtered_features.csv" \
    --output_dir "./final_model" \
    --n_estimators 450 \
    --max_depth 2 \
    --learning_rate 0.05
```
## Brain Age Prediction
Apply your trained model and bias correction to a new dataset (like a patient cohort).
```
python XGBoost_age_prediction.py \
    --file_path "patient_data.csv" \
    --best_model_json_file "./final_model/models/best_model_cv0.json" \
    --HC_prediction_json_file "./final_model/results/correction_params.json" \
    --output_csv_filename "patient_predictions.csv" \
    --max_depth 2 \
    --learning_rate 0.05
```
## Important Feature Identification for Re-training the Model from Scratch
Use this script to filter your raw data based on an importance file (e.g., from a preliminary run). It keeps the top features and the first 5 metadata columns.
```
python XGBoost_age_feature_selection.py \
    --data_file "raw_morphometry.csv" \
    --imp_file "permutation_results.csv" \
    --output_file "filtered_features.csv"
```
