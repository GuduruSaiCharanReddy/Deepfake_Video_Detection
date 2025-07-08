import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from transformers import SwinForImageClassification

# Define Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
class Temporal3DCNN(torch.nn.Module):
    def __init__(self):
        super(Temporal3DCNN, self).__init__()
        self.conv3d_1 = torch.nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3d_2 = torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3d_3 = torch.nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = torch.nn.Linear(256, 512)

    def forward(self, x):
        x = F.relu(self.conv3d_1(x))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FusionModel(torch.nn.Module):
    def __init__(self, swin, temporal_cnn):
        super(FusionModel, self).__init__()
        self.swin = swin
        self.temporal_cnn = temporal_cnn
        self.fc_fusion = torch.nn.Linear(514, 2)

    def forward(self, x):
        batch_size, frames, channels, height, width = x.size()

        x_swin = x.view(batch_size * frames, channels, height, width)
        swin_features = self.swin(x_swin).logits
        swin_features = swin_features.view(batch_size, frames, -1)
        swin_features = swin_features.mean(dim=1)

        x_3d = x.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_cnn(x_3d)

        fused_features = torch.cat((swin_features, temporal_features), dim=1)
        output = self.fc_fusion(fused_features)

        return output

# Load model
swin = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=2, ignore_mismatched_sizes=True)
temporal_cnn = Temporal3DCNN()
fusion_model = FusionModel(swin, temporal_cnn)

fusion_model.load_state_dict(torch.load("C:/Users/saich/Downloads/fusion_model (1).pth", map_location=torch.device('cpu')))
fusion_model.eval()

# Video frame extraction function
def extract_frames(video_path, frame_count=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // frame_count)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    frames = np.array(frames) / 255.0  # Normalize to [0, 1]
    frames = np.transpose(frames, (0, 3, 1, 2))  # Change to (frames, channels, height, width)
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# API endpoint for video upload
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"Error": "No video uploaded"}), 400

    video = request.files["video"]
    video_path = "uploaded_video.mp4"
    video.save(video_path)

    frames = extract_frames(video_path)

    with torch.no_grad():
        output = fusion_model(frames)
        print(output)
        prediction = torch.argmax(output, dim=1).item()

    result = "Fake" if prediction == 1 else "Real"
    return jsonify({"Prediction": result})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)