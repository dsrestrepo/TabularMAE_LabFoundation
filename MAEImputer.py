# stdlib
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import math, sys, argparse
import pandas as pd
import torch
from torch import nn
from functools import partial
import time, os, json
from utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset
#import MAE
from MAE import MaskedAutoencoder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import sys
import timm.optim.optim_factory as optim_factory
from utils import get_args_parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.datasets import load_iris
from tqdm import tqdm
eps = 1e-8
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from math import sqrt
import os 
import pickle

class ReMaskerStep:

    def __init__(self, dim=16, mask_ratio=0.5, max_epochs=300, warmup_epochs=20, save_path=None, model=None, device=None, weigths=None, eps = 1e-7, normalize=True, nan=-1,
                batch_size=64, accum_iter=1, min_lr=1e-5, norm_field_loss=False, 
                 weight_decay=0.05, lr=None, blr=1e-3, embed_dim=32, depth=6, 
                 decoder_depth=4, num_heads=4, mlp_ratio=4.0, encode_func='linear', **kwargs):
        #args = get_args_parser().parse_args()

        self.batch_size = batch_size
        self.accum_iter = accum_iter
        self.min_lr = min_lr
        self.norm_field_loss = norm_field_loss
        self.weight_decay = weight_decay
        self.lr = lr
        self.blr = blr
        self.warmup_epochs = warmup_epochs
        self.weigths = weigths
        self.dim = dim
        self.eps = 1e-7
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_epochs = max_epochs
        self.mask_ratio = mask_ratio
        self.encode_func = encode_func
        self.nan = nan
        
        if not save_path:
            self.save_path = f'./checkpoints_{self.mask_ratio}'
        else:
            self.save_path = save_path
            
        os.makedirs(save_path, exist_ok=True)
            
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if not(model):
            ### Model ###
            self.model = MaskedAutoencoder(
                rec_len=self.dim,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                decoder_embed_dim=self.embed_dim,
                decoder_depth=self.decoder_depth,
                decoder_num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                norm_layer=partial(nn.LayerNorm, eps=self.eps),
                norm_field_loss=self.norm_field_loss,
                encode_func=self.encode_func
            )
        else:
            self.model = model
            
        
        # Load Checkpoint if any
        if weigths and os.path.exists(weigths):
            print('loading model weigths...')
            self.model.load_state_dict(torch.load(weigths, map_location=torch.device(self.device)))
            
            
        if torch.cuda.device_count() > 1:  # Checks for multiple GPUs
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        self.model.to(self.device)
        
        #self.normalize_vals = normalize
        self.norm_parameters = None
        

    def calculate_norm_parameters(self, X: pd.DataFrame):
        
        min_val = np.zeros(self.dim)
        max_val = np.zeros(self.dim)
        
        for i in range(self.dim):
            # Use .iloc to access the DataFrame by integer-location
            min_val[i] = np.nanmin(X.iloc[:, i])
            max_val[i] = np.nanmax(X.iloc[:, i])
        
        self.norm_parameters = {"min": min_val, "max": max_val}
        
    def normalize(self, X_raw: pd.DataFrame, return_format='torch'):
        X = X_raw.copy()
        
        if not(self.norm_parameters):
            print('calculating norm parameters...')
            self.calculate_norm_parameters(X)
            
            # Save the norm_parameters to a file
            with open(os.path.join(self.save_path, 'norm_parameters.pkl'), 'wb') as file:
                pickle.dump(self.norm_parameters, file)
            
        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        ### Normalization:
        for i in range(self.dim):
            # Perform the operation and update the column
            X.iloc[:, i] = (X.iloc[:, i] - min_val[i]) / (max_val[i] - min_val[i] + self.eps)

        self.norm_parameters = {"min": min_val, "max": max_val}
        
        if return_format == 'numpy':
            np_array = X.to_numpy()
            return np_array
        elif return_format == 'torch': 
            np_array = X.to_numpy()
            # Convert NumPy array to PyTorch tensor
            X = torch.tensor(np_array, dtype=torch.float32)
            return X
        else:
            return X
        
    def denormalize(self, imputed_data):
    
        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]
        
        # Renormalize
        for i in range(self.dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + self.eps) + min_val[i]
            
        return imputed_data
        
        

    def fit(self, X_raw: pd.DataFrame, X_val=None, exclude_columns=[]):
        
        #if self.normalize:
        X = self.normalize(X_raw)
            
        # Set missing mask
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(self.device)

        X = torch.nan_to_num(X, nan=self.nan)
        X = X.to(self.device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        
        # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        
        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )
        
        # To store validation results
        results_csv_path = os.path.join(self.save_path, 'validation_results.csv')
        
        ############ Train Loop ############
        for epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            eight = True

            for iter, (samples, masks) in tqdm(enumerate(dataloader), total = len(dataloader)):
                
                # we use a per iteration (instead of per epoch) lr scheduler
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
                                         self.max_epochs, self.warmup_epochs)
                
                
                # Add 1 dimension and send to device
                samples = samples.unsqueeze(dim=1)
                samples = samples.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # Calculate the loss
                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio, exclude_columns=exclude_columns)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
                
                loss /= self.accum_iter
                
                # Calculate the gradient and backpropagate
                loss_scaler(loss, optimizer, parameters=self.model.parameters(),
                            update_grad=(iter + 1) % self.accum_iter == 0)
                
                # Set gradients to 0 each accum_iter iterations
                if (iter + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()

            total_loss = (total_loss / (iter + 1)) ** 0.5
            
            
            ############ Validation ############
            self.model.eval()
            eight_str = str(eight)
            if epoch % 30 == 0 and X_val is not None and not X_val.empty:

                # Get a subset of data
                if epoch != (self.max_epochs-1):
                    X_test = X_val[:10000]
                else: 
                    X_test = X_val
                    
                epoch_validation_results = []
                
                print(f'Evaluation of epoch {epoch}...')
                # Evaluate each lab value:
                for column, column_name in enumerate(X_test.columns):
                    
                    if 'time' in column_name:
                        continue
                        
                    # Ignore the time columns
                    if column in exclude_columns:
                        continue  
                    
                    # Only evaluate if the column contains values
                    X_test_real = X_test[X_test[column_name].notna()]
                    
                    if len(X_test_real) < 1:
                        print(f'The sampling size of test with in column: {column_name}, is only {len(X_test_real)}')
                        continue
                    
                    X_test_masked = X_test_real.copy()
                    # Mask all values in that column with NaN
                    X_test_masked.iloc[:,column]=np.nan

                    # Impute the values:
                    X_test_imputed =  pd.DataFrame(self.transform(X_test_masked).cpu().numpy())
                    
                    try:
                        # Calculate RMSE, MAE, and R2
                        rmse = sqrt(mean_squared_error(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna()))
                        mae = mean_absolute_error(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
                        r2 = r2_score(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
                        err = 0
                    except:
                        print(f'Error for {column_name}')
                        rmse = 0
                        mae = 0
                        r2 = 1
                        err = 1
                        
                    # Construct the output string
                    #output_str = f"Epoch{epoch} Evaluation for {column_name}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}, Confusion Matrix: {cm.tolist()}\n"
                    output_str = f"Epoch{epoch} Evaluation for {column_name}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}\n"
                    
                    """ Here if we wanna se the outputs per test: """
                    print(output_str)
                    
                    epoch_validation_results.append({
                        'Epoch': epoch,
                        'Column': column_name,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'Err': err
                    })

                results_df = pd.DataFrame(epoch_validation_results)

                # Check if file exists to determine if we need to write headers
                if not os.path.exists(results_csv_path):
                    results_df.to_csv(results_csv_path, index=False)  # Include header
                else:
                    results_df.to_csv(results_csv_path, mode='a', header=False, index=False)  # Append without header

          
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print((epoch+1),',', total_loss)
                # Check if dir exists, if not, create the dir
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), f'{self.save_path}/epoch{epoch+1}_checkpoint')
        
        return self

    def transform(self, X_raw: pd.DataFrame, eval_batch_size=None):
        
        no = X_raw.shape[0]
        
        #if self.normalize:
        X = self.normalize(X_raw)
            
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(self.device)
        
        X = torch.nan_to_num(X, nan=self.nan)
        X = X.to(self.device)
        
        dataset = MAEDataset(X, M)
        if eval_batch_size:
            dataloader = DataLoader(
                dataset, sampler=SequentialSampler(dataset),
                batch_size=eval_batch_size, 
                drop_last=False
            )
        else:
            dataloader = DataLoader(
                dataset, sampler=SequentialSampler(dataset),
                batch_size=self.batch_size, 
                drop_last=False
            )

        self.model.eval()

        # Imputed data
        imputed_data_list = []
        with torch.no_grad():
            for sample, mask in dataloader:
                sample = sample.unsqueeze(1)
                sample.to(self.device)
                mask.to(self.device)
                _, pred, _, _ = self.model(sample, mask, mask_ratio=0.0)
                pred = pred.squeeze(dim=2)
                imputed_data_list.append(pred)

        imputed_data = torch.cat(imputed_data_list, 0)
        imputed_data = self.denormalize(imputed_data)


        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        
        if not torch.is_tensor(X_raw):
            X_raw = torch.tensor(X_raw.values) 

        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data
    
    def extract_embeddings(self, X_raw: pd.DataFrame, eval_batch_size=None):
        
        no = X_raw.shape[0]
        
        #if self.normalize:
        X = self.normalize(X_raw)
            
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(self.device)
        
        X = torch.nan_to_num(X, nan=self.nan)
        X = X.to(self.device)
        
        dataset = MAEDataset(X, M)
        if eval_batch_size:
            dataloader = DataLoader(
                dataset, sampler=SequentialSampler(dataset),
                batch_size=eval_batch_size, 
                drop_last=False
            )
        else:
            dataloader = DataLoader(
                dataset, sampler=SequentialSampler(dataset),
                batch_size=self.batch_size, 
                drop_last=False
            )

        self.model.eval()

        # generate the embeddings
        embeddings_list = []
        with torch.no_grad():
            for sample, mask in dataloader:
                sample = sample.unsqueeze(1)
                sample.to(self.device)
                mask.to(self.device)
                embedding_batch, _ = self.model.extract_embeddings(sample, mask, mask_ratio=0.0)
                #embeddings = embeddings.squeeze(dim=2)
                embeddings_list.append(embedding_batch)

        embeddings = torch.cat(embeddings_list, 0)

        embeddings = embeddings.detach().cpu()
        
        return embeddings


    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Imputes the provided dataset using the GAIN strategy.
        Args:
            X: np.ndarray
                A dataset with missing values.
        Returns:
            Xhat: The imputed dataset.
        """
        X = torch.tensor(X.values, dtype=torch.float32)
        return self.fit(X).transform(X).detach().cpu().numpy()
