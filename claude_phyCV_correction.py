import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from phycv import VEVID, PST
import numpy as np
import tempfile
import cv2

class PhyCVTransform:
    def __init__(self, transform_type='vevid'):
        self.transform_type = transform_type
        if transform_type == 'vevid':
            self.transform = VEVID()
        elif transform_type == 'pst':
            self.transform = PST()

    def __call__(self, img):
        img_np = np.array(img)

        with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
            # Save the numpy array as an image
            cv2.imwrite(tmpfile.name, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            # Use the temporary file path in the VEVID/PST transformation
            if self.transform_type == 'vevid':
                processed = self.transform.run(img_file=tmpfile.name, S=0.4, T=0.001, b=0.25, G=0.8)
            elif self.transform_type == 'pst':
                processed = self.transform.run(img_file=tmpfile.name, S=0.4, W=20, sigma_LPF=0.1, thresh_min=0.0, thresh_max=0.8, morph_flag=1)

        # Ensure the processed image has 3 channels
        if len(processed.shape) == 2:  # Grayscale image
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        return torchvision.transforms.ToPILImage()(processed)
        
# Define AlexNet for CIFAR-10
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# class PhyCVTransform:
#     def __init__(self, transform_type='vevid'):
#         self.transform_type = transform_type
#         if transform_type == 'vevid':
#             self.transform = VEVID()
#         elif transform_type == 'pst':
#             self.transform = PST()

#     def __call__(self, img):
#         # Convert PIL Image to numpy array
#         img_np = np.array(img)
        
#         if self.transform_type == 'vevid':
#             # Apply VEVID
#             processed = self.transform.run(img_np, S=0.4, T=0.001, b=0.25, G=0.8)
#         elif self.transform_type == 'pst':
#             # Apply PST
#             processed = self.transform.run(img_np, S=0.4, W=20, sigma_LPF=0.1, thresh_min=0.0, thresh_max=0.8, morph_flag=1)
        
#         # Convert back to PIL Image
#         return torchvision.transforms.ToPILImage()(processed)

# class PhyCVTransform:
#     def __init__(self, transform_type='vevid'):
#         self.transform_type = transform_type
#         if transform_type == 'vevid':
#             self.transform = VEVID()
#         elif transform_type == 'pst':
#             self.transform = PST()

#     def __call__(self, img):
#         img_np = np.array(img)

#         if self.transform_type == 'vevid':
#             # Apply VEVID with required parameters
#             processed = self.transform.run(img_np, S=0.4, T=0.001, b=0.25, G=0.8)
#         elif self.transform_type == 'pst':
#             # Apply PST with required parameters
#             processed = self.transform.run(img_np, S=0.4, W=20, sigma_LPF=0.1, thresh_min=0.0, thresh_max=0.8, morph_flag=1)

#         return torchvision.transforms.ToPILImage()(processed)

# class PhyCVTransform:
#     def __init__(self, transform_type='vevid'):
#         self.transform_type = transform_type
#         if transform_type == 'vevid':
#             self.transform = VEVID()
#         elif transform_type == 'pst':
#             self.transform = PST()

#     def __call__(self, img):
#         img_np = np.array(img)

#         if self.transform_type == 'vevid':
#             processed = self.transform.run(img_np)
#         elif self.transform_type == 'pst':
#             processed = self.transform.run(img_np)

#         return torchvision.transforms.ToPILImage()(processed)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Wrap the main block to avoid multiprocessing issues
if __name__ == "__main__":
    # PhyCV transform pipeline
    transform_phycv = transforms.Compose([
        # PhyCVTransform(transform_type='vevid'),  # or 'pst'
        PhyCVTransform(transform_type='pst'),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Standard transform pipeline (without PhyCV)
    transform_standard = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset with PhyCV preprocessing
    testset_phycv = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_phycv)
    testloader_phycv = DataLoader(testset_phycv, batch_size=64, shuffle=False, num_workers=0)  # Set num_workers=0

    # Load CIFAR-10 dataset without PhyCV preprocessing
    testset_standard = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_standard)
    testloader_standard = DataLoader(testset_standard, batch_size=64, shuffle=False, num_workers=0)  # Set num_workers=0

    # Load your pre-trained model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlexNet().to(device)
    model.load_state_dict(torch.load('/Users/colincasey/AlexNetImplementationFall2024/alexnet_cifar10.pth'))

    # Evaluate the model with standard preprocessing
    accuracy_standard = evaluate(model, testloader_standard, device)
    print(f'Accuracy on the test images with standard preprocessing: {accuracy_standard:.2f}%')

    # Evaluate the model with PhyCV preprocessing
    accuracy_phycv = evaluate(model, testloader_phycv, device)
    print(f'Accuracy on the test images with PhyCV preprocessing: {accuracy_phycv:.2f}%')

    # Compare results
    print(f'Difference in accuracy: {accuracy_phycv - accuracy_standard:.2f}%')