import torch
import torch.nn as nn
import torch.nn.functional as F

def distill_train_epoch(teacher_model, student_model, train_loader,
                        optimizer, device, alpha=0.5, temperature=4.0):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            teacher_outputs = teacher_model(images)
            
        student_outputs = student_model(images)
        

        loss_ce = criterion_ce(student_outputs, labels)
        
        kd_loss = criterion_kd(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1)
        ) * (temperature**2)
        
        loss = alpha * loss_ce + (1 - alpha) * kd_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = student_outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
