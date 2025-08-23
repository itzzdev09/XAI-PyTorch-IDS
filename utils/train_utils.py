import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_model(model, train_loader, val_loader, epochs, lr, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint after each epoch
        os.makedirs("output/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"output/checkpoints/{model_name}_epoch{epoch+1}.pth")

    return history
