import os
import numpy as np
from PIL import Image
import pickle
import sys

def extract_features(img_path, size=(64, 64)):
    """Extract features from an image by converting to grayscale and flattening"""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(size)  # Resize to 64x64
        return np.array(img).flatten()  # Flatten pixels into 1D array
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}", file=sys.stderr)
        return None

def load_model(model_path='cat_dog_classifier.pkl'):
    """Load the trained model and parameters"""
    try:
        # Try to load from the absolute path first
        if os.path.isabs(model_path):
            model_file = model_path
        else:
            # If not absolute, look in the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_file = os.path.join(script_dir, model_path)
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

def predict_image(image_path, model_data):
    """Predict whether an image is a cat or dog"""
    # Extract features using the same parameters as training
    features = extract_features(image_path, size=model_data['image_size'])
    
    if features is None:
        return f"Error: Could not process image {image_path}"
    
    # Reshape features for prediction
    features = features.reshape(1, -1)
    
    # Make prediction
    prediction = model_data['model'].predict(features)[0]
    probability = model_data['model'].predict_proba(features)[0]
    confidence = max(probability)
    
    return f"{image_path} -> {prediction.upper()} (confidence: {confidence:.4f})"

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)
    
    # Load the model
    print("Loading model...")
    model_data = load_model()
    
    # Get image path from command line
    image_path = sys.argv[1]
    
    # Make prediction
    result = predict_image(image_path, model_data)
    print("\nPrediction result:")
    print(result)

if __name__ == "__main__":
    main()