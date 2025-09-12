import numpy as np
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
import joblib

print(torch.__version__)

# Load environment variables
load_dotenv(".env.dev")
# Setup device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### DATA PREPARATION #### 

def load_ds_to_df(filename):
    """
    Load NetCDF dataset and convert to Pandas DataFrame.
    Filter the data
    
    Args:
        filename (str or Path): Filename of NetCDF dataset to convert
    Return:
        df (pd.DataFrame): converted dataframe
    """
    ds = open_nc_dataset(filename=filename)
    ds = ds.sel(time=slice("2016", None))
    df = ds.to_dataframe().reset_index()
    del ds
    
    return df

def feature_engineering(df):
    """
    Add day of year feature to capture sesonality
    
    Args:
        df (pd.DataFrame): daily dataframe with 'time' index
    Returns:
        pd.DataFrame: dataframe with 'day_sin' and 'day_cos' features
    """
    df['day_of_year'] = df['time'].dt.dayofyear
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def calculate_seq_length_of_features(df, seq_length):
    """
    Transform time series into overlapping sequences for LSTM model training.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'sst', 'lat', 'lon', 'day_sin', and 'day_cos'.
    Returns:
        np.ndarray: 3D numpy array of shape (num_sequences, seq_length, num_features),
            where num_features = 5 (sst, lat, lon, day_sin, day_cos).
    """
    
    sst_array = df['sst'].values
    lat_array = df['lat'].values
    lon_array = df['lon'].values
    
    # Create  sliging window of each feature
    sst_data = np.lib.stride_tricks.sliding_window_view(sst_array, seq_length)
    lat_data = np.lib.stride_tricks.sliding_window_view(lat_array, seq_length)
    lon_data = np.lib.stride_tricks.sliding_window_view(lon_array, seq_length)
    day_sin_data = np.lib.stride_tricks.sliding_window_view(df['day_sin'].values, seq_length)
    day_cos_data = np.lib.stride_tricks.sliding_window_view(df['day_cos'].values, seq_length)
    
    data = np.stack((sst_data, lat_data, lon_data, day_sin_data, day_cos_data), axis=-1)
    
    del df
    return data

def test_train_split(data, train_size: float):
    """
    Split dataset into test-train
    
    Args:
        train_size (float): length of test-train split (0-1)
    Returns:
        tuple: (X_train, X_test, y_train, y_test) of torch.FloatTensor type.
            
    """
    train_size_value = int(train_size*len(data))
    
    X_train = torch.from_numpy(data[:train_size_value, :-1]).float()
    X_test = torch.from_numpy(data[train_size_value:, :-1]).float()

    y_train = torch.from_numpy(data[:train_size_value, -1, 0]).float()  
    y_test  = torch.from_numpy(data[train_size_value:, -1, 0]).float()

    return X_train, X_test, y_train, y_test

def create_scaler_and_apply(X_train, X_test, y_train, y_test, return_scalers_only):
    """
    Normalize input features and targets using scikit's learn StandardScaler
    
    Args:
        X_train (torch.FloatTensor): Training features
        X_test (torch.FloatTensor): Testing features
        y_train (torch.FloatTensor): Training targets
        y_test (torch.FloatTensor): Testing targets
        return_scalers_only (Bool): if True return only fitted Scaler
    Returns:
        if return_scaler_only == True:
            (X_scaler, y_scaler): fitted StandardScallers for X and Y
        if return_scaler_only == False:
            (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled) where each is a numpy array
    """
    num_samples_train, seq_len, num_features = X_train.shape
    num_samples_test = X_test.shape[0]
    
    DATA_PATH = Path("scalers")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Scale X
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_samples_train, seq_len, num_features)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(num_samples_test, seq_len, num_features)
    joblib.dump(X_scaler, 'scalers/X_scaler.pkl')

    # Scale y
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    joblib.dump(y_scaler, 'scalers/y_scaler.pkl')
    
    if not return_scalers_only:
    # Return everything
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    else:
    # Return only scalers
        return X_scaler, y_scaler

#### MODEL ####

class PredictionModel(nn.Module):
    """
    LSTM-based regression model for Sea Surface Temperature (SST) prediction on reef points.

    This model consists of an LSTM layer followed by a fully connected layer.
    It is designed to take a sequence of feature vectors as input and produce a single regression output per sample.

    Args:
        input_dim (int): Number of input features per time step.
        hidden_dim (int): Number of features in the hidden state of the LSTM.
        num_layers (int): Number of recurrent layers in the LSTM.
        output_dim (int): Number of output features (typically 1 for regression).
        
    Methods:
        forward(x):
            Runs a forward pass through the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_dim),
                              corresponding to the regression prediction for each input sequence.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        # Pass 
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Take only last output
        output = self.fc(out[:, -1, :])
        
        return output

#### DATA LOADER ####

def data_loaders(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, batch_size: int):
    """
    Converts preprocessed NumPy arrays into PyTorch TensorDatasets and DataLoaders, and saves them for future use.

    Creates TensorDatasets for training and testing data from scaled feature and target arrays,
    saves them as .pth file

    Args:
        X_train_scaled (np.ndarray): Scaled training features 
        X_test_scaled (np.ndarray): Scaled testing features 
        y_train_scaled (np.ndarray): Scaled training labels 
        y_test_scaled (np.ndarray): Scaled testing labels 
        batch_size (int): Number of samples per batch to load in DataLoader.

    Returns:
        tuple:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.

    Side Effects:
        - Saves the train and test TensorDataset objects in the directory "data_loaders" as "train_dataset.pth" and "test_dataset.pth".
        - Creates the "data_loaders" directory if it does not exist.

    Notes:
        - Does not shuffle the data in the DataLoader to preserve sequence order, which may be important for time series tasks.
    """
    train_dataset = TensorDataset(
                                torch.from_numpy(X_train_scaled).float(),
                                torch.from_numpy(y_train_scaled).float()
                                )

    test_dataset = TensorDataset(
                                torch.from_numpy(X_test_scaled).float(),
                                torch.from_numpy(y_test_scaled).float()
                                )
    
    DATA_PATH = Path("data_loaders")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    data = [train_dataset, test_dataset]
    names = ["train_dataset.pth", "test_dataset.pth"]

    for i in range(len(data)):
        DATA_NAME = names[i]
        DATA_SAVE_PATH = DATA_PATH / DATA_NAME
        torch.save(obj=data[i], f=DATA_SAVE_PATH)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

#### TRAIN EVAL LOOP ####

def train_eval_loop(model,
                    train_loader,
                    test_loader,
                    epochs:int,
                    device,
                    lr=0.001,
                    weight_decay=1e-4,
                    print_every=100):
    """
    Training and evaluation loop for PyTorch regression models, with learning rate scheduling.

    This function trains the provided model using mean squared error loss, Adam optimizer, and
    ReduceLROnPlateau learning rate scheduler.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        epochs (int): Number of epochs to train for.
        device (torch.device): Device to use for computation
        lr (float, optional): Initial learning rate for the optimizer. Default is 0.001.
        weight_decay (float, optional): L2 weight decay (regularization). Default is 1e-4.
        print_every (int, optional): Print training status every N epochs. Default is 100.

    Returns:
        None
    """
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5)

    size = len(train_loader.dataset)
    
    for epoch in tqdm(range(epochs)):
        # Training

        train_loss = 0
        model.train()
        
        # Loop through training batches
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            #Forward Pass
            y_train_preds = model(X)
            
            #Calculate loss per batch
            loss = loss_fn(y_train_preds, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if batch % 10000 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")
        # Divide total training loss by length of train dataloader (avg loss per batch)  
        train_loss /= len(train_loader)
        
        # Testing
        test_loss = 0
        # Switch model to eval mode
        model.eval()

        with torch.inference_mode():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                # Forward pass
                y_test_preds = model(X_test)
                
                # Calculate test loss   
                loss_test = loss_fn(y_test_preds, y_test)
                test_loss += loss_test
                
            # Divide total test loss by length of test dataloader (Avg loss per batch)       
            test_loss /= len(test_loader)
            
            # Step scheduler to reduce lr if model doesnt learn for patience period
            scheduler.step(test_loss)
            

        if epoch % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    
def save_model(name, model):
    """
    Save model to .pth file for future use
    
    Args:
        name (str): Model name
        model (torch.nn.Module): Model to save
        
    Returns:
        None
    
    Notes:
        Creates "models" directory if doesn't exist and saves smodel to "models" directory
        
    """
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    torch.save(model, MODEL_SAVE_PATH)

if __name__ == "__main__":

    # Load dataset from NETCDF file
    df = load_ds_to_df(filename='merged_reef_data.nc')
    
    # Feature engineering (daily sesonality)
    df = feature_engineering(df)
    
    # Create sequence windows of desired length
    data = calculate_seq_length_of_features(df, seq_length=90)
    del df
    
    # Train-test split
    X_train, X_test, y_train, y_test = test_train_split(data, train_size=0.8)
    del data
    
    # Scale Features
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = create_scaler_and_apply(X_train,
                                                                                           X_test,
                                                                                           y_train,
                                                                                           y_test,
                                                                                           return_scalers_only=False)
    del X_train, X_test, y_train, y_test
    
    # Create model
    model = PredictionModel(
                            input_dim=5,
                            hidden_dim=128,
                            num_layers=2,
                            output_dim=1
                            ).to(device)
    
    # Create data loaders
    train_loader, test_loader = data_loaders(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, batch_size=512)
    
    
    # Train and eval model
    train_eval_loop(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=200,
                    device=device,
                    lr=0.001,
                    weight_decay=1e-4,
                    print_every=1) 

    # Save model to .pth file
    save_model(name="LSTM_REEF_POINTS_SST_TRAINED_YEAR_2016",
               model=model)


