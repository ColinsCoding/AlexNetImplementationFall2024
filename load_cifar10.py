import torch
import torchvision
import torchvision.transforms as transforms

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize(256),            # Resize the images to 256x256 pixels
    transforms.CenterCrop(224),        # Crop the center 224x224 pixels (required for AlexNet)
    transforms.ToTensor(),             # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load the CIFAR-10 training and test datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torchvision.models as models

# Load the pre-trained AlexNet model
model = models.alexnet(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Verify that the model is loaded correctly
# print(model)

import torch.nn.functional as F

# Get a batch of test images
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Move the images to the same device as the model (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = images.to(device)
model = model.to(device)

# Perform a forward pass through the model
outputs = model(images)

# Get the predicted class with the highest score
_, predicted = torch.max(outputs, 1)

# Print the results
print("Predicted: ", predicted)
print("Ground Truth: ", labels)


import matplotlib.pyplot as plt
import numpy as np

# Function to display an image along with its label
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Display some test images
imshow(torchvision.utils.make_grid(images.cpu()))
if len(predicted) >= 4 and len(labels) >= 4:
    print('Predicted: ', ' '.join('%5s' % train_dataset.classes[predicted[j]] for j in range(4)))
    print('Ground Truth: ', ' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(4)))
else:
    print("Not enoguh predictions or labels to display")
