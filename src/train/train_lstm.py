import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        for X_seq, X_nonseq, y in train_loader:
            X_seq = X_seq.to(device)
            X_nonseq = X_nonseq.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X_seq, X_nonseq)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_seq.size(0)
            _, pred = torch.max(outputs, 1)
            total_correct += (pred == y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / len(train_loader.dataset)

        val_loss, val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")


def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for X_seq, X_nonseq, y in data_loader:
            X_seq = X_seq.to(device)
            X_nonseq = X_nonseq.to(device)
            y = y.to(device)

            outputs = model(X_seq, X_nonseq)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X_seq.size(0)
            _, pred = torch.max(outputs, 1)
            total_correct += (pred == y).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_correct / len(data_loader.dataset)
    return avg_loss, avg_acc
