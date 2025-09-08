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
    """
    ds = open_nc_dataset(filename=filename)
    ds = ds.sel(time=slice("2016", None))
    df = ds.to_dataframe().reset_index()
    del ds
    
    return df

def feature_engineering(df):
    """
    Add day of year feature to capture sesonality
    """
    df['day_of_year'] = df['time'].dt.dayofyear
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def calculate_seq_length_of_features(df, seq_length):
    """
    Transform time series into sequences for LSTM model training.
    seq_lengt: sliding window length
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
    train_size: length of split(0-1) as float
    """
    train_size_value = int(train_size*len(data))
    
    X_train = torch.from_numpy(data[:train_size_value, :-1]).float()
    X_test = torch.from_numpy(data[train_size_value:, :-1]).float()

    y_train = torch.from_numpy(data[:train_size_value, -1, 0]).float()  
    y_test  = torch.from_numpy(data[train_size_value:, -1, 0]).float()

    return X_train, X_test, y_train, y_test

def create_scaler_and_apply(X_train, X_test, y_train, y_test, return_scalers_only):
    """
    Normalize input and features using scikit's learn StandardScaler
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
    LSTM based regression model for SST prediction on reef-points
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

def data_loaders(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,batch_size: int):
    """
    Convert NumPy arrays to tensors and dataloaders, save preprocessed data for future evaluating.
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
    Training + evaluation loop with rate scheduling.
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





