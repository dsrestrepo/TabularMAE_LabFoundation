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
import model_mae
from torch.utils.data import DataLoader, RandomSampler
import sys
import timm.optim.optim_factory as optim_factory
from utils import get_args_parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
# hyperimpute absolute
# from hyperimpute.plugins.imputers import ImputerPlugin
from sklearn.datasets import load_iris
# from hyperimpute.utils.benchmarks import compare_models
# from hyperimpute.plugins.imputers import Imputers
from tqdm import tqdm
eps = 1e-8
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from math import sqrt
import os 

class ReMaskerStep:

    def __init__(self, mask_ratio=0.5, max_epochs=300, warmup_epochs=20, save_path=None, model=None, device=None, weigths=None):
        args = get_args_parser().parse_args()

        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = warmup_epochs
        self.norm_parameters = None
        self.weigths = None
        self.model = model

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = max_epochs
        self.mask_ratio = mask_ratio
        self.encode_func = args.encode_func
        
        if not save_path:
            self.save_path = f'./checkpoints_yesterdaymodel_{self.mask_ratio}'
        else:
            self.save_path = save_path
            
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        
            
    def fit(self, X_raw: pd.DataFrame, weigths=None):
        X = X_raw.copy()

        # Parameters
        no = len(X)
        dim = X.shape[1]

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # Assuming X is a DataFrame and dim is the number of columns
        dim = X.shape[1]
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        eps = 1e-7
        
        ### Model ###
        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )
        
        # Load Checkpoint if any
        if weigths and os.path.exists(weigths):
            self.model.load_state_dict(torch.load(weigths))
        
        self.model.to(self.device)

        ### Normalization:
        for i in range(dim):
            # Use .iloc to access the DataFrame by integer-location
            min_val[i] = np.nanmin(X.iloc[:, i])
            max_val[i] = np.nanmax(X.iloc[:, i])
            # Perform the operation and update the column
            X.iloc[:, i] = (X.iloc[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        self.norm_parameters = {"min": min_val, "max": max_val}
        np_array = X.to_numpy()

        # Convert NumPy array to PyTorch tensor
        X = torch.tensor(np_array, dtype=torch.float32)
        
        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(self.device)

        X = torch.nan_to_num(X)
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
            dataset, sampler = RandomSampler(dataset),
            batch_size=self.batch_size,
        )
        
        ############ Train Loop ############
        for epoch in range(self.max_epochs):
            self.model.train()
            print(epoch)
            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            eight = True

            for iter, (samples, masks) in tqdm(enumerate(dataloader), total = len(dataloader)):
                
                # Check if we are using 8 or 16 values:
                if iter == 0:          
                    if samples.shape[1]<16:
                        eight = True
                    else:
                        eight = False
                
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
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
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
            if epoch % 30 == 0:
                # columns    50882	50912	50971	50983	51006	51222	51265	51301
                # normalranges = {
                #     "50912": test2,
                #     "51265": test7
                #     "51222": test6
                #     "51301": test8
                #     "51006": test5
                #     "50983": test4
                #     "50882": test1
                #     "50971": test3
                # }
                normal_ranges = {
                    "test2": (0.7, 1.3),#test2
                    "test7": (150, 450),#test7
                    "test6": (12, 18),#test6
                    "test8": (4, 11),#test8
                    "test5": (8, 20),#test5
                    "test4": (136, 145),#test4
                    "test1": (23, 28), #test1
                    "test3": (3.5, 5),#test3
                }
                def classify_value(value, low, high):
                    if pd.isna(value):
                        return 'missing'  # Optional, to handle NaN values
                    if value < low:
                        return 'under'
                    elif value > high:
                        return 'over'
                    else:
                        return 'within'
    
                with open(f'Results_mask{self.mask_ratio}_eight{eight_str}.txt', 'a') as file:
                    # Read the test file
                    if eight:
                        X_test = pd.read_csv('X_test8.csv')
                    else:
                        X_test = pd.read_csv('X_test16.csv')
                        
                    # Get a subset of data
                    if epoch != (self.max_epochs-1):
                        X_test = X_test[:5000]
                    
                    # Evaluate each lab value:
                    for column, column_name in enumerate(X_test.columns[:8]):
                        X_test_masked = X_test.copy()
                        # Mask all values in that column with NaN
                        X_test_masked.iloc[:,column]=np.nan
                        
                        # Impute the values:
                        X_test_imputed =  pd.DataFrame(self.transform(X_test_masked).cpu().numpy())
                        
                        # Classify into normal abnormal
                        actual_classes = X_test.iloc[:, column].apply(classify_value, args=normal_ranges[column_name])
                        predicted_classes = X_test_imputed.iloc[:, column].apply(classify_value, args=normal_ranges[column_name])
                        
                        # Calculate the metrics:
                        cm = confusion_matrix(actual_classes, predicted_classes, labels=['under', 'within', 'over'])

                        # Calculate RMSE, MAE, and R2
                        rmse = sqrt(mean_squared_error(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna()))
                        mae = mean_absolute_error(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
                        r2 = r2_score(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())

                        # Construct the output string
                        output_str = f"Epoch{epoch} Evaluation for {column_name}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}, Confusion Matrix: {cm.tolist()}\n"
                        print(output_str)
                        # Write to file and print
                        file.write(output_str)
                        file.flush()
                    
          
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print((epoch+1),',', total_loss)
                # Check if dir exists, if not, create the dir
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), f'{self.save_path}/epoch{epoch+1}_checkpoint')
        
        return self

    def transform(self, X_raw: torch.Tensor, weigths=None):
        
                
        # Load Checkpoint if any
        if weigths and os.path.exists(weigths):
            self.model.load_state_dict(torch.load(weigths))
            
        if not torch.is_tensor(X_raw):
            X_raw = torch.tensor(X_raw.values) 
        X = X_raw.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = X.shape
        X = X.cpu()

        # MinMaxScaler normalization     
        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(self.device).float()
        M = M.to(self.device).float()

        self.model.eval()

        # Imputed data
        with torch.no_grad():
            for i in tqdm(range(no),total=no):
                sample = torch.reshape(X[i], (1, 1, -1))
                mask = torch.reshape(M[i], (1, -1))
                _, pred, _, _ = self.model(sample, mask)
                pred = pred.squeeze(dim=2)
                if i == 0:
                    imputed_data = pred
                else:
                    imputed_data = torch.cat((imputed_data, pred), 0)

        # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()

        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data

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
