import pandas as pd
from sklearn.model_selection import train_test_split
from MAEImputer import ReMaskerStep
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from math import sqrt
import os

import joblib
import xgboost

def load_xgboost_model(filename):
    # Load the model from the file
    loaded_model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return loaded_model

def predict_in_batches(model, X, batch_size=64):
    """Perform batch predictions using the provided model."""
    predictions = []
    # Split the data into batches
    for X_batch in np.array_split(X, len(X) // batch_size + 1):
        # Perform prediction on each batch
        predictions.append(model.predict(X_batch))
    # Concatenate all batch predictions into a single array
    return np.concatenate(predictions, axis=0)


################ Read Dataset ################
df_test = pd.read_csv('../data/X_test.csv')
print(f'Test values shape: {df_test.shape}')


################ Clean Missing Data ################
def clean_missing(df, threshold=20 + 3, missing_per_col=100, cols_to_remove=None):
    # Remove rows with less than 20 values
    df = df.dropna(thresh=threshold)
    print(f"DataFrame after removing rows with at least 20 missing values: {df.shape}")
    
    if type(cols_to_remove) != list:
        if missing_per_col and not cols_to_remove:
            # Get columns where at least 100 values are not missing
            columns_all_nan = df.columns[df.notna().sum() < missing_per_col].tolist()
            # Identify columns that end with a number after the last underscore
            ids = ['_' + col.split('_')[-1] for col in columns_all_nan]

            def ids_in_string(value_list, target_string):
                for value in value_list:
                    if value in target_string:
                        return True
                return False

            cols_to_remove = []
            for column in df.columns:
                if ids_in_string(ids, column):
                    cols_to_remove.append(column)

    print(f'Removing columns: {cols_to_remove}')

    df.drop(columns=cols_to_remove, inplace=True)
    
    return df, cols_to_remove

missing_per_row = 20 + 3 # + 3 because of: first_race, chartyear, hadm_id
missing_per_col = 500

df_test, _ = clean_missing(df_test, missing_per_row, cols_to_remove=[])

df_test = df_test[:100000]


# Create a list of columns to ignore
columns_ignore = ['first_race', 'chartyear', 'hadm_id']


################ Create Imputer Instance ################
columns = df_test.shape[1] - 3 # + 3 because of: first_race, chartyear, hadm_id
mask_ratio = 0.25
max_epochs = 300
save_path = '100_Labs_Train_0.25Mask_L_V3'
weigths = '100_Labs_Train_0.25Mask_L_V3/epoch390_checkpoint'


batch_size=256 
embed_dim=64
depth=8
decoder_depth=4
num_heads=8
mlp_ratio=4.0


imputer = ReMaskerStep(dim=columns, mask_ratio=mask_ratio, max_epochs=max_epochs, save_path=save_path, batch_size=batch_size,
                      embed_dim=embed_dim, depth=depth, decoder_depth=decoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      weigths=weigths)


with open('100_Labs_Train_0.25Mask_L_V3/norm_parameters.pkl', 'rb') as file:
    loaded_norm_parameters = pickle.load(file)
    
imputer.norm_parameters = loaded_norm_parameters


################ Test the model ################
def test_model(imputer, df_test, exclude_columns=[], eval_batch_size=32):
    print('Testing the MAE Model')
    epoch_validation_results = []
    
    for column, column_name in enumerate(df_test.columns):
        
        # Columns with time
        if 'time' in column_name:
            continue
        if 'last' in column_name:
            continue
        # Columns to ignore
        if column_name in exclude_columns:
            continue

        # Only evaluate if the column contains values
        X_test_real = df_test[df_test[column_name].notna()]
        if len(X_test_real) < 1:
            print(f'The sampling size of test with in column: {column_name}, is only {len(X_test_real)}')
            continue

        X_test_masked = X_test_real.copy()
        # Mask all values in that column with NaN
        X_test_masked.iloc[:,column]=np.nan

        # Impute the values:
        X_test_imputed =  pd.DataFrame(imputer.transform(X_test_masked, eval_batch_size=eval_batch_size).cpu().numpy())
        
        print(f'calculating metrics for {column_name}')
        try:
            # Calculate RMSE, MAE, and R2
            rmse = sqrt(mean_squared_error(df_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna()))
            mae = mean_absolute_error(df_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
            r2 = r2_score(df_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
            err = 0
        except:
            print(f'Error for {column_name}')
            rmse = 0
            mae = 0
            r2 = 1
            err = 1

        # Construct the output string
        output_str = f"Evaluation for {column_name}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}\n"

        """ Here if we wanna se the outputs per test: """
        print(output_str)

        epoch_validation_results.append({
            'Column': column_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Err': err,
            'Model': 'MAE'
        })

    results_df = pd.DataFrame(epoch_validation_results)

    return results_df



################ Test the model ################
def test_model_xgb(df_test, exclude_columns=[]):
    print('Testing the XGB Model')
    epoch_validation_results = []
    ItemID = 0
    
    for column, column_name in enumerate(df_test.columns):
        
        # Columns with time
        if 'time' in column_name:
            continue
        if 'last' in column_name:
            continue
        # Columns to ignore
        if column_name in exclude_columns:
            continue

        # Only evaluate if the column contains values
        X_test_real = df_test[df_test[column_name].notna()]
        if len(X_test_real) < 1:
            print(f'The sampling size of test with in column: {column_name}, is only {len(X_test_real)}')
            continue

        X_test_masked = X_test_real.copy()
        
        print(f'Imputing column: {X_test_masked.columns[ItemID*2]} using the xgboost model')
        baseModel = load_xgboost_model(f"/Users/davidrestrepo/MAE Tabular/xgboost/{ItemID}best_model.pkl")
        
        # Drop the column to be imputed
        X_test_masked_copy = X_test_masked.drop(X_test_masked.columns[ItemID*2], axis=1).copy()
        
        # Perform batch predictions using the batch size
        X_test_imputed = baseModel.predict(X_test_masked_copy)
        
        ItemID += 1
        
        print(f'calculating metrics for {column_name}')
        try:
            # Calculate RMSE, MAE, and R2
            rmse = sqrt(mean_squared_error(df_test.iloc[:, column].dropna(), X_test_imputed))
            mae = mean_absolute_error(df_test.iloc[:, column].dropna(), X_test_imputed)
            r2 = r2_score(df_test.iloc[:, column].dropna(), X_test_imputed)
            err = 0
        except:
            print(f'Error for {column_name}')
            rmse = 0
            mae = 0
            r2 = 1
            err = 1

        # Construct the output string
        output_str = f"Evaluation for {column_name}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}\n"

        """ Here if we wanna se the outputs per test: """
        print(output_str)

        epoch_validation_results.append({
            'Column': column_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Err': err,
            'Model': 'XGB'
        })

    results_df = pd.DataFrame(epoch_validation_results)

    return results_df


df_xgb = test_model_xgb(df_test.drop(columns_ignore, axis=1))
df_MAE = test_model(imputer, df_test.drop(columns_ignore, axis=1))

df = pd.concat([df_xgb, df_MAE])
df.to_csv('results_test_mae_xgb.csv', index=False)
print("Overall metrics calculated and saved.")


### Calculate the metrics per race ###

# Function to calculate metrics per race group
def calculate_metrics_per_race(df_test, imputer, races_column='first_race', exclude_columns=[], eval_batch_size=32):
    race_metrics = []
    races = ['White', 'Black', 'Hispanic', 'Asian', 'Others']  # The 5 race groups

    # Loop through each race and calculate metrics for each group
    for race in races:
        print(f"\nCalculating metrics for race: {race}")
        race_df_test = df_test[df_test[races_column] == race]  # Filter the data for the current race
        
        if race_df_test.empty:
            print(f"No data for race: {race}")
            continue
        
        race_results = test_model(imputer, race_df_test.drop(columns_ignore, axis=1), exclude_columns, eval_batch_size)
        race_results['Race'] = race  # Add the race column to the results
        
        race_metrics.append(race_results)

    # Concatenate all race metrics into one DataFrame
    all_race_metrics = pd.concat(race_metrics, ignore_index=True)
    return all_race_metrics

# Function to calculate metrics per race for XGBoost
def calculate_metrics_per_race_xgb(df_test, races_column='first_race', exclude_columns=[]):
    race_metrics_xgb = []
    races = ['White', 'Black', 'Hispanic', 'Asian', 'Others']  # The 5 race groups

    # Loop through each race and calculate metrics for each group
    for race in races:
        print(f"\nCalculating XGBoost metrics for race: {race}")
        race_df_test = df_test[df_test[races_column] == race]  # Filter the data for the current race
        
        if race_df_test.empty:
            print(f"No data for race: {race}")
            continue
        
        race_results_xgb = test_model_xgb(race_df_test.drop(columns_ignore, axis=1), exclude_columns)
        race_results_xgb['Race'] = race  # Add the race column to the results
        
        race_metrics_xgb.append(race_results_xgb)

    # Concatenate all race metrics into one DataFrame
    all_race_metrics_xgb = pd.concat(race_metrics_xgb, ignore_index=True)
    return all_race_metrics_xgb


# Now calculate metrics per race
df_race_metrics = calculate_metrics_per_race(df_test, imputer, exclude_columns=columns_ignore)
df_race_metrics_xgb = calculate_metrics_per_race_xgb(df_test, exclude_columns=columns_ignore)

# Combine the results for both models
df_combined_race_metrics = pd.concat([df_race_metrics, df_race_metrics_xgb])

# Save results to a CSV file
df_combined_race_metrics.to_csv('results_test_mae_xgb_per_race.csv', index=False)

print("Metrics per race calculated and saved.")
