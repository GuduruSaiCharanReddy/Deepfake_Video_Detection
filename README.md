This project presents a deepfake detection system that leverages Swin Transformer for extracting spatial features and 3D CNN for capturing temporal dynamics from videos. By combining both spatial and temporal information, the model achieves high accuracy in detecting manipulated video content.

**Features**

- Spatial Feature Extraction using Swin Transformer

- Temporal Feature Learning using 3D Convolutional Neural Network

- Feature Fusion for combining spatial and temporal representations

- Achieves 84% accuracy on FaceForensics++ dataset

- Integrated with React.js frontend and Flask backend for real-time prediction

- Evaluated on Celeb-DF(v2) for cross-dataset generalization

**Datasets Used**

- FaceForensics++ – Used for training and testing the model

- Celeb-DF(v2) – Used for cross-dataset evaluation

**Tech Stack**
  
**Model & Backend**

- Python 3.13.2

- PyTorch

- HuggingFace Transformers

- OpenCV

- Flask (for model serving)

**Frontend**

- React.js

- Axios
  
