import argparse
import torch
from data import *
from utils import *
from teacher import create_teacher, train_teacher, evaluate
from student import create_student, evaluate_on_cpu
from distillation import distill_train_epoch
import torch.optim as optim
import torchvision
import time

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")  
    
    train_loader, test_loader = get_cifar100_loaders(batch_size=args.batch_size)
    
    teacher_model = create_teacher(model_name=args.teacher_model, num_classes=args.num_classes, pretrained=True)
    teacher_model.to(device)
    print("=== Train Teacher (ViT) ===")
    train_teacher(teacher_model,args.teacher_model, train_loader, test_loader, device, epochs=args.num_epochs_teacher, lr=args.lr_teacher)
    print("=== Evaluate Teacher ===")
    load_checkpoint(teacher_model, optim.Adam(teacher_model.parameters(), lr=args.lr_teacher), f"teacher_{args.teacher_model}.pth")
    evaluate(teacher_model, test_loader, device)
    
    print("=== Evaluate Teacher on CPU ===")
    teacher_cpu_loss, teacher_cpu_acc, teacher_cpu_time = evaluate_on_cpu(teacher_model, test_loader)

    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    for model_name in ["mobilenet_v2", "efficientnet_b0", "squeezenet1_0", "shufflenet_v2_x1_0"]:
        student_model = create_student(num_classes=args.num_classes, model_name=model_name)
        student_model.to(device)
        optimizer_student = optim.Adam(student_model.parameters(), lr=args.lr_student)
        best_acc = 0.0
        print(f"\n=== Distill: Teacher {args.teacher_model} -> Student {model_name} ===")
        for epoch in range(args.num_epochs_distill):
            train_loss, train_acc = distill_train_epoch(
                teacher_model, student_model.to(device), train_loader,
                optimizer_student, device, alpha=args.alpha, temperature=args.temperature
            )
            print(f"[Distill] Epoch {epoch+1}/{args.num_epochs_distill}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if epoch % 5 == 0 or epoch > 20:
                print("=== Evaluate Student ===")
                val_loss, val_acc = evaluate(student_model, test_loader, device)
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(student_model, optimizer_student, epoch, best_acc, f"student_{model_name}.pth")
        print(f"Best Acc: {best_acc:.2f}%")
        print("=== Evaluate Student on CPU ===")
        load_checkpoint(student_model, optimizer_student, f"student_{model_name}.pth")
        evaluate_on_cpu(student_model, test_loader)
        _, _, student_cpu_time = evaluate_on_cpu(student_model, test_loader)
        
        print(f"Speed comparison: Teacher CPU time: {teacher_cpu_time:.4f}s, Student CPU time: {student_cpu_time:.4f}s")
        print(f"Student is {teacher_cpu_time/student_cpu_time:.2f}x faster than Teacher on CPU")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation từ ViT sang CNN")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size cho quá trình huấn luyện")
    parser.add_argument("--num_classes", type=int, default=100, help="Số lớp (mặc định: 1000 cho ImageNet)")
    parser.add_argument("--num_epochs_teacher", type=int, default=2, help="Số epoch huấn luyện teacher")
    parser.add_argument("--num_epochs_distill", type=int, default=5, help="Số epoch huấn luyện distillation cho student")
    parser.add_argument("--lr_teacher", type=float, default=1e-4, help="Learning rate cho teacher")
    parser.add_argument("--lr_student", type=float, default=1e-3, help="Learning rate cho student")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trọng số kết hợp giữa CE loss và KD loss")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature cho distillation")
    parser.add_argument("--teacher_model", type=str, default="vit_base_patch16_224",
                        choices=["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224"],
                        help="Chọn mô hình ViT cho teacher")
    # parser.add_argument("--student_model", type=str, default="resnet18",
    #                     choices=["resnet18", "mobilenet_v2", "efficientnet_b0", "squeezenet1_0", "shufflenet_v2_x1_0"],
    #                     help="Chọn mô hình CNN cho student")
    args = parser.parse_args()
    
    main(args)
