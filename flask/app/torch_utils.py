import io
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

IMG_SIZE = 32
NUM_CLASSES = 10
CLASSES = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"]

# load model
model = torchvision.models.resnet18(
    pretrained=True,
)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(500, NUM_CLASSES)
)

PATH = "../models/image_classification_resnet18.pth"
model.load_state_dict(torch.load(
    PATH, map_location=torch.device('cpu')), strict=False)
model.eval()

# image -> tensor


def transform_image(image_bytes):
    training_norm = {
        'mean': np.array([0.47410759, 0.4726623, 0.47331911]),
        'std': np.array([0.2520572, 0.25201249, 0.25063239])
    }

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=training_norm['mean'], std=training_norm['std'])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    print(transform(image).shape)

    return transform(image).unsqueeze(0)

# predict


def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 3, IMG_SIZE, IMG_SIZE)

    outputs = model.forward(images)

    # max returns (value ,index)
    predicted = torch.argmax(outputs, 1)

    return CLASSES[predicted[0]]
