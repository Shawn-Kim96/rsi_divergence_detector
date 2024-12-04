import os
import sys
from ..model.transformer import *


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

feature_size = sequences.shape[2]
num_classes = len(np.unique(labels))

model = TimeSeriesTransformer(feature_size=feature_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for sequences_batch, labels_batch in train_loader:
        sequences_batch = sequences_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    with torch.no_grad():
        for sequences_batch, labels_batch in val_loader:
            sequences_batch = sequences_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch)
            total_val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels_batch).sum().item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / len(val_dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
