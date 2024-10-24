import torchvision.models as models

def load_model():
    model = models.alexnet(pretrained=True)
    return model
