import torch
import torch.nn as nn
import torchvision.models as models
import time

def create_student(num_classes=100, model_name="resnet18"):
    assert model_name in ["resnet18", "mobilenet_v2", "efficientnet_b0", "squeezenet1_0", "shufflenet_v2_x1_0"], "Chọn model hợp lệ!"

    if model_name == "mobilenet_v2":
        student_model = models.mobilenet_v2(pretrained=False)
        student_model.classifier[1] = nn.Linear(student_model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        student_model = models.efficientnet_b0(pretrained=False)
        student_model.classifier[1] = nn.Linear(student_model.classifier[1].in_features, num_classes)

    elif model_name == "squeezenet1_0":
        student_model = models.squeezenet1_0(pretrained=False)
        student_model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)

    elif model_name == "shufflenet_v2_x1_0":
        student_model = models.shufflenet_v2_x1_0(pretrained=False)
        student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)

    return student_model

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

    print(f"[Evaluate on CPU] Test Loss={avg_loss:.4f}, Test Acc={avg_acc:.2f}%")
    print(f"Inference Time on CPU: {inference_time:.4f} seconds")
    return avg_loss, avg_acc, inference_time

