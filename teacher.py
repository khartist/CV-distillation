import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

import timm
from utils import save_checkpoint, load_checkpoint

def create_teacher(model_name, num_classes=100, pretrained=True):
    
    assert model_name in ['vit_tiny_patch16_224', 'vit_small_patch16_224',
                          'vit_base_patch16_224', 'vit_large_patch16_224'], "Chọn model ViT hợp lệ!"
    
    teacher_model = timm.create_model(model_name, pretrained=pretrained)
    in_features = teacher_model.head.in_features
    teacher_model.head = nn.Linear(in_features, num_classes)
    return teacher_model

def train_teacher(model, model_name, train_loader, val_loader, device, epochs=2, lr=1e-4):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        best_acc = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"[Teacher] Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        if epoch > 10 or epoch % 5 == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch, best_acc, f"teacher_{model_name}.pth")

def evaluate(model, test_loader, device):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    print(f"[Evaluate] Test Loss={avg_loss:.4f}, Test Acc={avg_acc:.2f}%")
    return avg_loss, avg_acc

def evaluate_on_cpu(model, test_loader):
    device_cpu = torch.device("cpu") 
    model.to(device_cpu)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time() 

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device_cpu), labels.to(device_cpu)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    end_time = time.time() 

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    inference_time = end_time - start_time

    print(f"[Evaluate Teacher on CPU] Test Loss={avg_loss:.4f}, Test Acc={avg_acc:.2f}%")
    print(f"Teacher Inference Time on CPU: {inference_time:.4f} seconds")
    return avg_loss, avg_acc, inference_time