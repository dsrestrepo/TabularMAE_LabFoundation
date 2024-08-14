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

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin
from sklearn.datasets import load_iris
from hyperimpute.utils.benchmarks import compare_models
from hyperimpute.plugins.imputers import Imputers
from tqdm import tqdm
eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReMasker:

    def __init__(self):
        args = get_args_parser().parse_args()

        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = 20
        self.model = None
        self.norm_parameters = None

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = 300
        self.mask_ratio = 0.5
        self.encode_func = args.encode_func
        
    def set_params(self, X_raw: pd.DataFrame):
        X = X_raw.copy()

        # Parameters
        no = len(X)
        dim = X.shape[1]

        # X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

# import numpy as np
# import pandas as pd

# Assuming X is a DataFrame and dim is the number of columns
        dim = X.shape[1]
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        eps = 1e-7

        for i in range(dim):
            # Use .iloc to access the DataFrame by integer-location
            min_val[i] = np.nanmin(X.iloc[:, i])
            max_val[i] = np.nanmax(X.iloc[:, i])
            # Perform the operation and update the column
            X.iloc[:, i] = (X.iloc[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        self.norm_parameters = {"min": min_val, "max": max_val}
    def load(self, dim, path):
        eps = 1e-7
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
        self.model.load_state_dict(torch.load(path))
        self.model.to(device)
        self.model.eval()
        
    def fit(self,  X_raw: pd.DataFrame):
        X = X_raw.copy()

        # Parameters
        no = len(X)
        dim = X.shape[1]

        # X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

# import numpy as np
# import pandas as pd

# Assuming X is a DataFrame and dim is the number of columns
        dim = X.shape[1]
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        eps = 1e-7

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
        M = M.float().to(device)

        X = torch.nan_to_num(X)
        X = X.to(device)

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
        print(self.embed_dim, self.depth, self.num_heads, self.decoder_depth)
        # if self.improve and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.model.to(device)
        #     return self

        self.model.to(device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        # optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        # if self.resume and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.lr *= 0.5

        self.model.train()
        best_loss = 99999
        for epoch in range(self.max_epochs):
            print(epoch)
            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            for iter, (samples, masks) in tqdm(enumerate(dataloader), total = len(dataloader)):

                # we use a per iteration (instead of per epoch) lr scheduler
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
                                         self.max_epochs, self.warmup_epochs)

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                # print(samples, masks)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= self.accum_iter
                loss_scaler(loss, optimizer, parameters=self.model.parameters(),
                            update_grad=(iter + 1) % self.accum_iter == 0)

                if (iter + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()

            total_loss = (total_loss / (iter + 1)) ** 0.5
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.model.state_dict(), f'./yesterdaymodel/epoch{epoch+1} yesterdaymodel best_loss{best_loss}')
                # torch.save(self.model.state_dict(), f'./todaymodel/epoch{epoch+1} todaymodel best_loss{best_loss}')
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print((epoch+1),',', total_loss)
                torch.save(self.model.state_dict(), f'./yesterdaymodel/epoch{epoch+1} yesterdaymodel regular checkpoint')
                # torch.save(self.model.state_dict(), f'./todaymodel/epoch{epoch+1} todaymodel regular checkpoint')
        return self
    # def load_transform(self, X_train, X_test):
        
    def transform(self, X_raw: torch.Tensor):
        
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

        X = torch.from_numpy(X).to(device).float()
        M = M.to(device).float()

        self.model.eval()
        # print("debug")
        # Imputed data
        # print(no)
        with torch.no_grad():
            for i in range(no):
                sample = torch.reshape(X[i], (1, 1, -1))
                mask = torch.reshape(M[i], (1, -1))
                # print("S1",sample.shape,mask.shape)
                # print(
                _, pred, _, _ = self.model(sample, mask)
                # print("S2")
                pred = pred.squeeze(dim=2)
                if i == 0:
                    imputed_data = pred
                else:
                    imputed_data = torch.cat((imputed_data, pred), 0)
        # print("debug1")
                    # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]
        # print("debug2")
        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)
        
        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        # print('imputed', imputed_data, M)
        # print('imputed', M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data)
        # print("debug3")
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
