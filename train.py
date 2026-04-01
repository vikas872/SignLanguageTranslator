import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import SignLanguageDataset
from model import ASLNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms (Data Augmentation for training, just tensor for testing)
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = SignLanguageDataset(split='train', transform=train_transform)
    test_dataset = SignLanguageDataset(split='test', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Needs a custom collate loop if HF dataset returns images out of dict... wait let's fix dataset transform
    
    # Initialize model, loss function, and optimizer
    model = ASLNet(num_classes=25).to(device) # J and Z are excluded, so 24 classes usually, but max label is 24 (total 25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in progress_bar:
            # HuggingFace datasets often need some tweaking inside the loader
            # Our dataset __getitem__ extracts image and label correctly
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar_val = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in progress_bar_val:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}% | Val Loss: {val_loss/len(test_loader):.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Saving best model with accuracy {best_acc:.2f}%")
            torch.save(model.state_dict(), "asl_cnn_model.pth")
            
    print("Training finished!")

if __name__ == "__main__":
    train()
