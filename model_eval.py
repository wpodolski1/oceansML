import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from helper_functions import open_nc_dataset
from dotenv import load_dotenv
from pathlib import Path
from time_series import PredictionModel, create_scaler_and_apply, load_ds_to_df
from torch.serialization import safe_globals
import joblib
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Load environment variables
load_dotenv(".env.dev")
# Setup device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_loaders(batch_size):
    """
    Loads training and testing datasets and returning DataLoaders.
    
    Args:
        batch_size (int): Batch size to use with data loader
        
    Returns:
        train loader & test loader
    """
    
    DATA_PATH = Path("data_loaders")
    
    with safe_globals([TensorDataset]):
        train_dataset = torch.load(DATA_PATH / "train_dataset.pth", weights_only=False)
        test_dataset = torch.load(DATA_PATH / "test_dataset.pth", weights_only=False)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def model_import(PATH, MODEL_NAME, device=device):
    """
    Load trained model from path
    
    Args:
        PATH (Path): dir where model is saved
        MODEL_NAME (str): Name of the model
        device (torch.device): Device to load the model (gpu recommended)
        
    Returns:
        pretrained model
    """
    
    MODEL_PATH = Path(PATH)
    MODEL_LOAD_PATH = MODEL_PATH / MODEL_NAME
    
    model = PredictionModel(
                            input_dim=5,
                            hidden_dim=128,
                            num_layers=2,
                            output_dim=1
                            ).to(device)
    
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=False))
    model.eval()
    
    return model
    
def make_predictions(test_loader, y_scaler):
    """
    Use to make predictions on dataset within torch.inference_mode:
    Args:
        test_loader (Dataloader): test dataloader
        y_scaler (sklearn's scaler): Saved scaler to inverser.transform data
        
    Returns:
        original test data and predictions
    """
    with torch.inference_mode():
        
        y_preds_list = []
        y_test_list = []
        
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            
            y_preds = model(X_test)
            
            y_preds_list.append(y_preds.cpu())
            y_test_list.append(y_test.cpu())
            
        y_test_preds = torch.cat(y_preds_list, dim=0).numpy()
        y_test= torch.cat(y_test_list, dim=0).numpy()
        
        y_test_preds = y_scaler.inverse_transform(y_test_preds)
        y_test = y_scaler.inverse_transform(y_test)
        
    return y_test_preds, y_test
        

def apply_predictions_to_main_dataset(df, y_test, y_test_preds, seq_length, train_size):
    """
    Aligns predictions with original df and returns df with actual + predicted values
    
    Args:
        df (pd.DataFrame): original df
        y_test (np.ndarray) : test data (actual)
        y_test_preds (np.ndarray): predicted test values
        seq_length (int): sequence length used for training
        train_size (float): size of train test split
        
    Returns:
        Combined pd.DataFrame with aligned predictions along original data
    """
    
    num_sequences = len(df) - seq_length + 1
    num_train = int(train_size * num_sequences)
    
    # Selects row in the orignal df that matches with prediction 
    # if window index = 0, sequence rows are 0-89 (seq_length - 1) so window 0 prediction is on 89th row window 1 on 90 etc.
    df_test_aligned = df.iloc[seq_length - 1 + num_train : seq_length - 1 + num_sequences].reset_index(drop=True)
    
    # Building results df SST original + predicted
    results_df = df_test_aligned.copy()
    results_df["actual_sst"] = y_test.flatten()[:len(df_test_aligned)]
    results_df["predicted_sst"] = y_test_preds.flatten()[:len(df_test_aligned)]
    
    return results_df
    
def plot_preds(df):
    """
    Visualize predictions vs test data
    
    Args:
        df (pd.DataFrame): df with aligned predictions and original data across dims
        
    Returns: None
    """
    # Group by month (1-12, across all years)
    actual_mean = df.groupby(df["time"].dt.to_period("M"))["actual_sst"].mean()
    pred_mean = df.groupby(df["time"].dt.to_period("M"))["predicted_sst"].mean()

    plt.figure(figsize=(12, 10))
    plt.plot(actual_mean.index.astype(str), actual_mean.values, color="green", label="Actual SST")
    plt.plot(pred_mean.index.astype(str), pred_mean.values, color="red", label="Predicted SST")
    plt.xlabel("Month")
    plt.ylabel("Mean SST")
    plt.title("Mean Actual vs Predicted SST By Month-Year")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
if __name__ == "__main__":
    
    # Load train_loader and test_loader
    train_loader, test_loader = data_loaders(batch_size=512)
    
    # Import trained model
    model = model_import(PATH="models",
                         MODEL_NAME="LSTM_REEF_POINTS_SST.pth",
                         device=device)
    
    # Import scalers used to prepare data for modeling
    X_scaler = joblib.load("scalers/X_scaler.pkl")
    y_scaler = joblib.load("scalers/y_scaler.pkl")
    

    y_test_preds, y_test = make_predictions(test_loader=test_loader,
                                            y_scaler=y_scaler)
    
    # Model eval metrics
    rmse = root_mean_squared_error(y_test, y_test_preds)
    mae = mean_absolute_error(y_test, y_test_preds)
    r2 = r2_score(y_test, y_test_preds)
    bias = (y_test_preds - y_test).mean()
    
    print(f"RMSE:{rmse:2f}°C\nMAE:{mae:.2f}°C\nR2:{r2:.2f}\nBias:{bias:.2f}°C")
    
    
    # Load SST data (merged yearly sst files)
    df = load_ds_to_df(filename="merged_reef_data.nc")
    
    results_df = apply_predictions_to_main_dataset(df=df,
                                                   y_test=y_test,
                                                   y_test_preds=y_test_preds,
                                                   seq_length=90,
                                                   train_size=0.8)
    
    
    # Plot predictions vs historical data
    plot_preds(df=results_df)