from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import requests
import json
from PIL import Image
import io
import torch # Import torch
import torch.nn as nn # Import nn for model architecture
from torchvision import models, transforms # Import models and transforms from torchvision

# Load environment variables from .env file (for local development)
# This line is primarily for local development. On Render, environment variables
# are injected directly and load_dotenv() won't do anything, but it's harmless.
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- MongoDB Configuration ---
# Get MONGO_URI from environment variables, default to localhost for local dev
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/micro_problem_db")

db = None
try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping') # Test connection
    db = client['micro_problem_db']
    print("MongoDB connected successfully!")
    db_status_message = "connected"
except Exception as e:
    print(f"MongoDB connection error: {e}")
    db_status_message = f"error: {e}"
    db = None # Ensure db is None if connection fails

# --- Gemini API Configuration ---
# Get GEMINI_API_KEY from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- ML Model Loading ---
# Device configuration (use GPU if available, otherwise CPU)
ml_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"ML Model will use device: {ml_device}")

# Paths to the saved model and class names.
# These paths are relative to where app.py is executed.
# If app.py is at the root and models/ is at the root, these are correct.
MODEL_PATH = './models/plant_disease_resnet18.pth'
CLASS_NAMES_PATH = './models/class_names.json'

# Initialize model and load weights
model = None
class_names = []

try:
    # Load a pre-trained ResNet-18 model (ImageNet weights)
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # First, load class names to get num_classes for the final layer
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    
    # Adjust the final fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the trained state dictionary (your specific model weights)
    # map_location ensures it loads correctly regardless of CUDA availability
    model.load_state_dict(torch.load(MODEL_PATH, map_location=ml_device))
    model.eval() # Set model to evaluation mode (important for inference)
    model.to(ml_device) # Move model to the configured device (CPU or GPU)
    print("ML Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH} or class names at {CLASS_NAMES_PATH}.")
    print("Please ensure the model files are downloaded to the 'models/' directory.")
except Exception as e:
    print(f"Error loading ML model: {e}")
    model = None # Set model to None if loading fails, so endpoints can check

# Define the same transformations used for validation during training
preprocess_transform = transforms.Compose([
    transforms.Resize(256),      # Resize image to 256x256
    transforms.CenterCrop(224),  # Crop the center 224x224 (ResNet input size)
    transforms.ToTensor(),       # Convert PIL Image to PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize with ImageNet stats
])

@app.route('/')
def home():
    """
    Root endpoint for the API. Returns a status message and backend health.
    """
    return jsonify({
        "status": "success",
        "message": "AI Problem Solver Backend API is running!",
        "database_status": db_status_message,
        "ml_model_status": "loaded" if model else "failed to load"
    })

# --- ML Model Inference Function ---
def identify_problem_with_ml_model(image_data):
    """
    Uses the loaded PyTorch ML model to identify the problem in the image.
    Returns the identified class name and a confidence score.
    """
    if model is None or not class_names:
        print("ML Model or class names not loaded, skipping inference.")
        return "ML Model Not Loaded", 0.0 # Return dummy if model failed to load

    try:
        # Open image using Pillow from bytes data
        image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB format for 3 channels

        # Apply defined transformations
        input_tensor = preprocess_transform(image)
        input_batch = input_tensor.unsqueeze(0) # Add a batch dimension (BCHW format)

        # Move input tensor to the same device as the model
        input_batch = input_batch.to(ml_device)

        with torch.no_grad(): # Disable gradient calculation for inference (saves memory and speeds up)
            output = model(input_batch)

        # Get probabilities by applying softmax to the model's output logits
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # Get the highest probability and its corresponding index
        confidence, predicted_idx = torch.max(probabilities, 0)

        # Map the predicted index to the actual class name
        identified_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100 # Convert confidence tensor to Python float and then to percentage

        print(f"ML identified: {identified_class} with confidence: {confidence_score:.2f}%")
        return identified_class, confidence_score

    except Exception as e:
        print(f"Error during ML inference: {e}")
        return "ML Inference Error", 0.0

# --- Gemini API Advice Generation ---
def get_gemini_advice(identified_problem, user_context_text="", confidence=0.0):
    """
    Calls the Gemini API to generate advice based on the identified problem
    and optional user-provided context.
    """
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set. Cannot call Gemini API.")
        return "AI advice is unavailable: GEMINI_API_KEY not configured."

    # Craft a detailed prompt for Gemini
    # Incorporate confidence for better context
    prompt = f"""
    A user has submitted an image for problem identification.
    The image analysis model identified the problem as: "{identified_problem}" with a confidence of {confidence:.2f}%.
    The user also provided the following context/description: "{user_context_text}".

    Please act as an expert problem-solver for this specific issue. Provide a concise, actionable solution or advice for this identified problem, taking into account the user's context and the model's confidence.
    Focus on practical steps, potential causes, and preventative measures. If the confidence is below 70%, you can mention that the identification might need further verification. Keep the response under 250 words.

    Identified Problem: {identified_problem}
    User Context: {user_context_text if user_context_text else 'None provided.'}

    Solution/Advice:
    """

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 400 # Increased tokens for potentially more detailed advice
        }
    }

    headers = {
        'Content-Type': 'application/json'
    }
    
    # Construct the full Gemini API URL with the API key
    GEMINI_API_URL_WITH_KEY = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    try:
        print(f"Calling Gemini API for advice on: '{identified_problem}' with confidence {confidence:.2f}%...")
        response = requests.post(GEMINI_API_URL_WITH_KEY, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        gemini_response = response.json()
        
        # Parse the response from Gemini
        if gemini_response and gemini_response.get('candidates'):
            advice_text = gemini_response['candidates'][0]['content']['parts'][0]['text'].strip()
            print("Gemini advice generated successfully.")
            return advice_text
        else:
            print("Gemini API returned no candidates or content.")
            return "AI advice could not be generated. Please try again."

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error calling Gemini API: {e.response.status_code} - {e.response.text}")
        return f"AI advice failed (HTTP Error: {e.response.status_code}). Please check API key/quota."
    except requests.exceptions.RequestException as e:
        print(f"Network error calling Gemini API: {e}")
        return f"AI advice failed (Network Error). Please check your internet connection."
    except Exception as e:
        print(f"Unexpected error processing Gemini response: {e}")
        return f"AI advice failed (Unexpected Error: {e})."

# --- Main Endpoint for Image Upload and Problem Identification ---
@app.route('/identify_issue', methods=['POST'])
def identify_issue():
    """
    Handles image uploads, performs ML inference, and generates AI advice.
    """
    if db is None:
        return jsonify({"error": "Database connection not available"}), 500

    if model is None: # Check if ML model loaded successfully
        return jsonify({"error": "ML model not loaded. Please check backend logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    user_context_text = request.form.get('context', '')

    image_data = image_file.read()

    # 1. Use ML model to identify the problem
    identified_problem, confidence_score = identify_problem_with_ml_model(image_data)

    # 2. Use Gemini to get advice based on identified problem and user context
    advice = get_gemini_advice(identified_problem, user_context_text, confidence_score)

    return jsonify({
        "identified_problem": identified_problem,
        "advice": advice,
        "confidence": f"{confidence_score:.2f}%"
    }), 200


if __name__ == '__main__':
    # For local development, run on port 5000
    # For deployment (e.g., Render), the host should be '0.0.0.0' and port from environment
    port = int(os.environ.get('PORT', 5000)) # Get port from env var or default to 5000
    app.run(host='0.0.0.0', debug=True, port=port) # <--- CHANGED THIS LINE
