import os
import sys
import pandas as pd
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Add project path to system path
PROJECT_PATH = "/Users/shawn/Documents/personal/rsi_divergence_detector"
sys.path.append(PROJECT_PATH)

from src.dataset.lstm_dataset import LSTMDivergenceDataset
from src.model.lstm_mixed import MixedLSTMModel
from src.training.train_lstm import train_model, evaluate_model, plot_results, model_naming


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to train, evaluate, and test the LSTM model for divergence classification.
    """
    # Load data
    logger.info("Loading data...")
    df = pd.read_pickle(os.path.join(PROJECT_PATH, 'data', 'training_data.pickle'))
    divergence_data = pd.read_pickle(os.path.join(PROJECT_PATH, 'data', 'divergence_data2.pickle'))

    # Filter 5-minute timeframe data
    price_df = df[df['timeframe'] == '5m'].copy()
    divergence_df = divergence_data['5m'].copy()  # Assuming '5m' key exists

    # Split divergence_df into train/val/test
    logger.info("Splitting data into train, validation, and test sets...")
    total_events = len(divergence_df)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_count = int(total_events * train_ratio)
    val_count = int(total_events * val_ratio)
    test_count = total_events - train_count - val_count

    divergence_df_train = divergence_df.iloc[:train_count]
    divergence_df_val = divergence_df.iloc[train_count:train_count+val_count]
    divergence_df_test = divergence_df.iloc[train_count+val_count:]

    logger.info(f"Train events: {len(divergence_df_train)}, "
                f"Validation events: {len(divergence_df_val)}, "
                f"Test events: {len(divergence_df_test)}")

    # Prepare divergence_data for multiple timeframes (if applicable)
    # Assuming divergence_data contains multiple timeframes
    # Example: divergence_data = {'5m': ddf_5m, '15m': ddf_15m, '1h': ddf_1h, ...}
    # For simplicity, using only '5m' here
    divergence_data_subset = {'5m': divergence_df_train}

    # Initialize Dataset
    logger.info("Initializing datasets...")
    train_dataset = LSTMDivergenceDataset(ddf=divergence_df_train, 
                                         df5=price_df, 
                                         divergence_data=divergence_data, 
                                         seq_length=288)  # 288 * 5min = 24 hours
    # Use the same scaler for validation and test
    scaler = train_dataset.scaler
    val_dataset = LSTMDivergenceDataset(ddf=divergence_df_val, 
                                       df5=price_df, 
                                       divergence_data=divergence_data, 
                                       seq_length=288, 
                                       scaler=scaler)
    test_dataset = LSTMDivergenceDataset(ddf=divergence_df_test, 
                                        df5=price_df, 
                                        divergence_data=divergence_data, 
                                        seq_length=288, 
                                        scaler=scaler)

    # Initialize DataLoaders
    logger.info("Creating DataLoaders...")
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize Model
    model_args = {
        "seq_input_dim": len(train_dataset.ts_cols),
        "seq_hidden_dim": 128,
        "seq_num_layers": 3,
        "nonseq_input_dim": len(train_dataset.nonseq_cols),
        "mlp_hidden_dim": 256,
        "num_classes": 2,
        "dropout": 0.3
    }
    model = MixedLSTMModel(**model_args)
    logger.info(f"Model initialized with args: {model_args}")

    # Generate model name
    model_name = model_naming(**model_args)
    model_save_path = os.path.join(PROJECT_PATH, 'model_data', 'mixed_lstm', model_name)

    # Train the model
    logger.info("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=1e-3,
        device=device,
        log_interval=10,
        save_path=model_save_path
    )

    # Load the best model
    logger.info("Loading the best model for testing...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot training and validation metrics
    logger.info("Plotting training results...")
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, outdir=os.path.join(PROJECT_PATH, 'plots'))

    # Optionally, you can save the metrics for reporting
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    metrics_path = os.path.join(PROJECT_PATH, 'model_data', 'mixed_lstm', 'metrics.pkl')
    pd.to_pickle(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()