# TabularMAE_LabFoundation

**TabularMAE_LabFoundation** is a Masked Autoencoder model designed for the representation learning and imputation of medical lab values, considering their temporal dependencies. This model is particularly useful in healthcare analytics, where missing data is common, and temporal patterns are crucial for accurate predictions and analysis.

## Repository Structure

This repository is organized as follows:

- **MAE.py**: Contains the core implementation of the Masked Autoencoder (MAE) model.
- **MAEImputer.py**: Implements the training and inference processes for the MAE model, specifically tailored for data imputation and embedding extraction.
- **run_mae.py**: Script to train the MAE model on the provided dataset.
- **run_embedding.py**: Script to extract embeddings from the trained MAE model.
- **run_test_mae.py**: Script to test the MAE model's performance on a given dataset.
- **run_test_mae_race.py**: Script to test the MAE model's performance for the paper vs XGBoost models on a given test set and compare performance overall and per race.
- **run_test_mae_race_follow_up.py**: Script to test the MAE model's performance for the paper vs XGBoost models on a given test set and compare performance with a follow-up data and without follow-up data overall and per race.
- **Notebook Demos**:
  - **mae_demo.ipynb**: Demonstrates the basic usage of the MAE model, including training and testing.
  - **mae_imputer_inference_demo.ipynb**: Explores how to use the trained MAE model for data imputation.
  - **mae_imputer_training_demo.ipynb**: Provides an in-depth walkthrough of training the MAE model on a custom dataset.
- **imput_format.csv**: Sample csv with the format for training the MAE model or for inference.
- **requirements.txt**: Contains the required libraries for running the scripts.
- **results**: Directory with the results of the model on the test set for the paper vs XGBoost models. In the directory, there are the results for the overall performance, and the per race performance. You can find the results for the follow-up data and without follow-up data. In the notebook, you'll also see some demo results for the first 3 lab values for a cohor of 10k patients in the test set. Full results were calculated for 100 laboratoy tests on a cohort of 100k patients.

## Getting Started

### Prerequisites

Before running the scripts, ensure that you have the required libraries installed. You can install them using `pip`:

The code was created using Python 3.9.13. create a virtual environment and install the required libraries using the following commands:

with venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

or with conda:

```bash
conda create -n venv python=3.9.13
conda activate venv
```

Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the MAE model, run:

```bash
python run_mae.py
```

This script will train the model on the provided dataset, saving the model weights and other relevant training artifacts.

### Use the Lab-MAE Checkpoint

If you want to use the model pre-trained on over 1.4M data points from MIMIC, you need the checkpoint file and normalization parameters. You can download them from the following link:

[Lab-MAE Checkpoint](https://drive.google.com/drive/folders/1i_hdYGcB0mi0L8BJTYSI4VU-GKSaQIAo?usp=sharing)

Download the files and place them in your checkpoint directory. 

### Extracting Embeddings
Once the model is trained, you can extract embeddings using:

```bash
python run_embedding.py
```
This script will output the learned representations for the data, which can be used for various downstream tasks.

### Testing the Model
To evaluate the performance of the trained MAE model, run:

```bash
python run_test_mae.py
```

This script will provide metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for the imputation task.

If you want to compare the performance of the MAE model with the XGBoost model, you can run the following scripts:

- For the paper vs XGBoost models overall and per race:

```bash
python run_test_mae_race.py
```

- For the paper vs XGBoost models with follow-up data and without follow-up data:

```bash
python run_test_mae_race_follow_up.py
```

These scripts will output the MAE and RMSE for the MAE model and the XGBoost model, comparing the performance. To use the XGBoost model, you need to have the XGBoost model checkpoint files. You can download it from the following link:

[XGBoost Checkpoints](https://drive.google.com/drive/folders/1EGJlWUqcQpV46G2_R0LxxdZ-saDlzeNC?usp=sharing)

## Notebook Demos

Explore the following Jupyter notebooks for interactive demonstrations:

- **mae_demo.ipynb**: A general demonstration of how to use the MAE model.
- **mae_imputer_inference_demo.ipynb**: Learn how to perform data imputation using the trained MAE model and embedding extraction.
- **mae_imputer_training_demo.ipynb**: A comprehensive guide to training the MAE model on a custom dataset, and inference.

## Contributions

Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter any bugs or have suggestions for improvements.

### Contact

If you have any questions or need further assistance, please feel free to contact me at davidres@mit.edu

## License

This project is licensed under the MIT License.
