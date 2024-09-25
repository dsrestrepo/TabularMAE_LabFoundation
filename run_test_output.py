import pandas as pd
from MAEImputer import ReMaskerStep
import numpy as np
import pickle
import os
import csv
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

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

# get a subset of the first 100000 rows
df_test = df_test[:100000]


# Create a list of columns to ignore
columns_ignore = ['first_race', 'chartyear', 'hadm_id']


################ Create Imputer Instance ################
columns = df_test.shape[1] - 3 # + 3 because of: first_race, chartyear, hadm_id
mask_ratio = 0.25
max_epochs = 300
save_path = '100_Labs_Train_0.25Mask_L_V3'
weights = '100_Labs_Train_0.25Mask_L_V3/epoch390_checkpoint'


batch_size=256 
embed_dim=64
depth=8
decoder_depth=4
num_heads=8
mlp_ratio=4.0
device = 'mps'

imputer = ReMaskerStep(dim=columns, mask_ratio=mask_ratio, max_epochs=max_epochs, save_path=save_path, batch_size=batch_size,
                      embed_dim=embed_dim, depth=depth, decoder_depth=decoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      weights=weights, device=device)


with open('100_Labs_Train_0.25Mask_L_V3/norm_parameters.pkl', 'rb') as file:
    loaded_norm_parameters = pickle.load(file)
    
imputer.norm_parameters = loaded_norm_parameters


################ Test the model ################
def test_model(imputer, df_test, exclude_columns=[], eval_batch_size=32, predictions_file='results_test_predictions_test_100k.csv', ignore_previous=True, baseline=True):
    # Initialize the DataFrame to store predictions
    df_predictions = df_test.copy()
    
    ItemID = 0
    for column, column_name in enumerate(tqdm(df_test.columns, desc="Processing columns")):
        # Columns with time
        if 'time' in column_name:
            continue
        
        # Ignore columns with '_last' in the name
        if ignore_previous:
            if '_last' in column_name:
                continue
        
        # Columns to ignore
        if column_name in exclude_columns:
            continue

        # Only evaluate if the column contains values
        X_test_real = df_test[df_test[column_name].notna()]
        if len(X_test_real) < 1:
            print(f'The sampling size of test within column: {column_name}, is only {len(X_test_real)}')
            continue

        X_test_masked = X_test_real.copy()
        # Mask all values in that column with NaN
        X_test_masked.iloc[:, column] = np.nan

        ################ Impute the values using MAE ################
        print(f'Imputing column: {column_name} using MAE')
        
        # Impute the values:
        X_test_imputed = imputer.transform(X_test_masked, eval_batch_size=eval_batch_size).cpu().numpy()
        
        # Add a column to the DataFrame with the imputed values
        imputed_column_name = column_name + '_imputed_MAE'
        df_predictions[imputed_column_name] = np.nan
        
        # Create a new column with the imputed values from the MAE model
        df_predictions.loc[df_test[column_name].notna(), imputed_column_name] = X_test_imputed[:, column]
        
        
        
        # Save the predictions column to the CSV file
        #df_predictions[[imputed_column_name]].to_csv(predictions_file, mode='a', header=False, index=False)
        
        ################ Impute the values using the baseline ################
        # Impute the values:
        if baseline:
            print(f'Imputing column: {X_test_masked.columns[ItemID*2]} using the xgboost model')
            baseModel = load_xgboost_model(f"/Users/davidrestrepo/MAE Tabular/xgboost/{ItemID}best_model.pkl")
            
            # Drop the column to be imputed
            X_test_masked_copy = X_test_masked.drop(X_test_masked.columns[ItemID*2], axis=1).copy()
            
            # Perform batch predictions using the batch size
            #X_test_imputed = predict_in_batches(baseModel, X_test_masked_copy, batch_size=eval_batch_size)
            X_test_imputed = baseModel.predict(X_test_masked_copy)
            
            # Add a column to the DataFrame with the imputed values
            imputed_column_name = column_name + '_imputed_xgb'
            df_predictions[imputed_column_name] = np.nan
            
            # Create a new column with the imputed values from the MAE model
            df_predictions.loc[df_test[column_name].notna(), imputed_column_name] = X_test_imputed
            
            # Save the predictions column to the CSV file
            #df_predictions[[imputed_column_name]].to_csv(predictions_file, mode='a', header=False, index=False)
            
            ItemID += 1
        
        # Fill the masked column with the imputed values
        #df_predictions.loc[df_test[column_name].notna(), column_name] = X_test_imputed[:, column]
        # Save the predictions column to the CSV file
        #df_predictions[[column_name]].to_csv(predictions_file, mode='a', header=False, index=False)
        
    # Save the final DataFrame to CSV
    df_predictions.to_csv(predictions_file, index=False)


# Evaluate and save results
test_model(imputer, df_test.drop(columns_ignore, axis=1))
