import os, sys
import pandas as pd
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


PROJECT_PATH = "/Users/shawn/Documents/personal/rsi_divergence_detector"
sys.path.append(PROJECT_PATH)


from src.dataset.lstm_dataset import LSTMDivergenceDataset
from src.model.lstm_mixed import MixedLSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ----------------------------------------
# Training and Evaluation Functions
# ----------------------------------------

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu', 
               log_interval=10, save_path='best_model.pt'):
    """
    Train the model and validate after each epoch. Save the best model based on validation accuracy.
    
    Parameters:
    - model: The PyTorch model to train.
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - epochs: Number of training epochs.
    - lr: Learning rate.
    - device: 'cpu' or 'cuda'.
    - log_interval: Interval for logging training progress.
    - save_path: Path to save the best model.
    
    Returns:
    - train_losses, val_losses, train_accuracies, val_accuracies: Lists of metrics per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduces LR by 10x every 10 epochs

    model.to(device)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for batch_idx, (X_seq, X_nonseq, y) in enumerate(train_loader):
            X_seq = X_seq.to(device)
            X_nonseq = X_nonseq.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X_seq, X_nonseq)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_seq.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()

            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Step the scheduler
        scheduler.step()

        # Save the model if it has the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved with Val Acc: {best_val_acc:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate the model on a given dataset.
    
    Parameters:
    - model: The PyTorch model to evaluate.
    - data_loader: DataLoader for the dataset.
    - device: 'cpu' or 'cuda'.
    
    Returns:
    - avg_loss: Average loss over the dataset.
    - avg_acc: Average accuracy over the dataset.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for X_seq, X_nonseq, y in data_loader:
            X_seq = X_seq.to(device)
            X_nonseq = X_nonseq.to(device)
            y = y.to(device)

            outputs = model(X_seq, X_nonseq)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X_seq.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_correct / len(data_loader.dataset)

    return avg_loss, avg_acc


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, outdir='plots'):
    """
    Plot training and validation loss and accuracy.
    
    Parameters:
    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    - train_accuracies: List of training accuracies per epoch.
    - val_accuracies: List of validation accuracies per epoch.
    - outdir: Directory to save the plots.
    """
    os.makedirs(outdir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'loss_plot.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'accuracy_plot.png'))
    plt.close()



def model_naming(**kwargs):
    """
    Generate a model name based on hyperparameters.
    
    Parameters:
    - kwargs: Dictionary of hyperparameters.
    
    Returns:
    - name: String representing the model name.
    """
    name = ""
    for key, value in kwargs.items():
        abbrev = ''.join([word[0] for word in key.split('_')])  # e.g., 'seq_input_dim' -> 'sid'
        if isinstance(value, float):
            name += f"{abbrev}{value:.4f}_"
        else:
            name += f"{abbrev}{value}_"
    name = name.rstrip('_') + ".pt"
    return name


def main():
    df = pd.read_pickle(f'{PROJECT_PATH}/data/training_data.pickle')
    divergence_data = pd.read_pickle(f"{PROJECT_PATH}/data/divergence_data2.pickle")

    price_df = df.loc[df.timeframe == '5m']
    divergence_df = divergence_data['15m']
    # Assume price_df and divergence_df (for 5m divergences) and divergence_data dict are loaded
    # divergence_data might look like: {'5m': divergence_df_5m, '1h': divergence_df_1h, '4h': divergence_df_4h ...}
    # For simplicity, let's assume you have them:
    # price_df: full 5-min data
    # divergence_df: divergence events with columns including 'end_datetime', 'label', etc.
    # divergence_data: a dictionary with multiple timeframe divergence DFs

    # Split divergence_df into train/val/test
    total_events = len(divergence_df)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    # Make sure ratios sum up to 1

    train_count = int(total_events * train_ratio)
    val_count = int(total_events * val_ratio)
    test_count = total_events - train_count - val_count

    divergence_df_train = divergence_df.iloc[:train_count]
    divergence_df_val = divergence_df.iloc[train_count:train_count+val_count]
    divergence_df_test = divergence_df.iloc[train_count+val_count:]

    # Create dataset and scale sequential features on training set only
    train_temp = LSTMDivergenceDataset(divergence_df_train, price_df, divergence_data)
    scaler = train_temp.scaler  # fitted on training set
    train_dataset = LSTMDivergenceDataset(divergence_df_train, price_df, divergence_data, scaler=scaler)
    val_dataset = LSTMDivergenceDataset(divergence_df_val, price_df, divergence_data, scaler=scaler)
    test_dataset = LSTMDivergenceDataset(divergence_df_test, price_df, divergence_data, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_arguments = {
        "seq_input_dim": len(train_dataset.ts_cols),
        "seq_hidden_dim": 64,
        "seq_num_layers": 2,
        "nonseq_input_dim": len(train_dataset.nonseq_cols),
        "mlp_hidden_dim": 64,
        "num_classes": 2,
        "dropout": 0.2
    }

    model = MixedLSTMModel(**model_arguments)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=20, lr=1e-3, device=device, save_path=f'{PROJECT_PATH}/mode_data/mixed_lstm/{name_model(model_arguments)}.pt'
    )

    # Load best model
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    test_loss, test_acc = evaluate_model(model, test_loader, device=device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot results
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, outdir='.')


# ----------------------------------------
# Example Usage
# ----------------------------------------
if __name__ == "__main__":
    main()