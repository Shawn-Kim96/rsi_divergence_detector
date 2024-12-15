import os, sys
import pandas as pd
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_data')
sys.path.append(PROJECT_PATH)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ----------------------------------------
# Training and Evaluation Functions
# ----------------------------------------


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, lr=1e-3, device='cpu', 
               log_interval=10, save_path='best_model.pt', patience=50):
    """
    Enhanced train_model with early stopping and a learning rate scheduler that reduces LR on plateau.
    
    Parameters:
    - patience: Number of epochs to wait for improvement before early stopping.
    """

    model.to(device)

    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        t= time.time()
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

            # if (batch_idx + 1) % log_interval == 0:
            #     logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        matrix = evaluate_model(model, val_loader, device)
        val_loss, val_acc = matrix['loss'], matrix['accuracy']
        # val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"Time cost: {(time.time() - t):.2f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved with Val Acc: {best_val_acc:.4f}")
        elif epoch - best_epoch >= patience:
            logger.info(f"No improvement for {patience} epochs. Early stopping.")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate the model and compute additional metrics.
    
    Parameters:
    - model: The PyTorch model to evaluate.
    - data_loader: DataLoader for the dataset.
    - device: 'cpu' or 'cuda'.
    
    Returns:
    - metrics: Dictionary containing loss, accuracy, precision, recall, f1-score.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_seq, X_nonseq, y in data_loader:
            X_seq = X_seq.to(device)
            X_nonseq = X_nonseq.to(device)
            y = y.to(device)

            outputs = model(X_seq, X_nonseq)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X_seq.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    return metrics

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
