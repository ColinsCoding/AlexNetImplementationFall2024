import torch

def save(model, path='alexnet.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')
