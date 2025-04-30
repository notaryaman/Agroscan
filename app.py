import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json


with open("class_names.json", "r") as f:
    class_names = json.load(f)


class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2 * 2 * 512, 14)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cpu")
model = DeeperCNN()
model.load_state_dict(torch.load("plant_disease_classifier2.pth", map_location=device))

model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image to identify the disease. There are samples avalaible in this google drive link: https://shorturl.at/jZx5T.")
st.write("The model can recognize 14 different classes, including both healthy and infected leaves from tomato, potato, and bell pepper plants.")
st.write("Github Link: https://github.com/notaryaman/Agroscan")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    nice_label = prediction.replace('_', ' ').title()

    if "healthy" in prediction.lower():
        st.success(f"The plant looks **healthy** — ({nice_label})")
    else:
        st.warning(f"The plant may have **{nice_label}**.\nPlease consult an expert if symptoms are visible.")

