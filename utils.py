import torch
import os

def save_checkpoint(model, optimizer, epoch, best_acc, filename="checkpoint.pth"):

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):

    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Checkpoint loaded: {filename}, Epoch: {start_epoch}, Best Acc: {best_acc:.2f}%")
        return start_epoch, best_acc
    else:
        print(f"No checkpoint found at {filename}, starting fresh.")
        return 0, 0.0
