# TabularMAE_LabFoundation

**TabularMAE_LabFoundation** is a Masked Autoencoder model designed for the representation learning and imputation of medical lab values, considering their temporal dependencies. This model is particularly useful in healthcare analytics, where missing data is common, and temporal patterns are crucial for accurate predictions and analysis.

## Repository Structure

This repository is organized as follows:

- **MAE.py**: Contains the core implementation of the Masked Autoencoder (MAE) model.
- **MAEImputer.py**: Implements the training and inference processes for the MAE model, specifically tailored for data imputation and embedding extraction.
- **run_mae.py**: Script to train the MAE model on the provided dataset.
- **run_embedding.py**: Script to extract embeddings from the trained MAE model.
- **run_test_mae.py**: Script to test the MAE model's performance on a given dataset.
- **Notebook Demos**:
  - **mae_demo.ipynb**: Demonstrates the basic usage of the MAE model, including training and testing.
  - **mae_imputer_inference_demo.ipynb**: Explores how to use the trained MAE model for data imputation.
  - **mae_imputer_training_demo.ipynb**: Provides an in-depth walkthrough of training the MAE model on a custom dataset.
- **imput_format.csv**: Sample csv with the format for training the MAE model or for inference.

## Getting Started

### Prerequisites

Before running the scripts, ensure that you have the required libraries installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the MAE model, run:

```bash
python run_mae.py
```

This script will train the model on the provided dataset, saving the model weights and other relevant training artifacts.

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
