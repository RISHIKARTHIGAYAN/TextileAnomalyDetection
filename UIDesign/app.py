from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import torch  # assuming you are using PyTorch for the model
import torchvision.transforms as transforms

app = Flask(__name__)

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Make sure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Anomaly detection model (assuming a pretrained model is loaded)
def load_model():
    # Assuming you have a trained model saved
    model = torch.load('your_model.pth')
    model.eval()
    return model

# Image processing and anomaly detection function
def detect_anomaly(image_path, model):
    # Preprocess the image for your model (resize, normalization, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust for your model
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        output = model(image)
        anomaly_score = output.item()  # Adjust depending on how your model outputs results

    # Simple threshold for anomaly detection (you may change the threshold)
    if anomaly_score > 0.5:  # Example threshold
        return "Anomaly detected!"
    else:
        return "No anomaly detected."

# Endpoint to handle image upload and anomaly detection
@app.route('/detect-anomaly', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"result": "No image uploaded."}), 400

    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        model = load_model()
        result = detect_anomaly(filepath, model)

        return jsonify({"result": result})
    else:
        return jsonify({"result": "Invalid file format. Please upload an image."}), 400

if __name__ == '__main__':
    app.run(debug=True)
