import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import urllib.request
import zipfile
import shutil
import pickle
import warnings
warnings.filterwarnings('ignore')

def download_dataset():
    # Download and extract the Microsoft Cats and Dogs dataset
    print("Downloading dataset...")
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    urllib.request.urlretrieve(url, "cats_and_dogs.zip")
    
    print("Extracting dataset...")
    with zipfile.ZipFile("cats_and_dogs.zip", "r") as zip_ref:
        zip_ref.extractall("dataset_raw")
    
    # Create directories for processed images
    os.makedirs("dataset/test/cat", exist_ok=True)
    os.makedirs("dataset/test/dog", exist_ok=True)

def extract_features(img_path, size=(64, 64)):
    #Extract features from an image by converting to grayscale and flattening
    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(size)  # Resize to 64x64
        return np.array(img).flatten()  # Flatten pixels into 1D array
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def process_images(source_dir, limit=1000):
    # Process images from source directory and return features and labels
    data = []
    labels = []
    count = 0
    
    # Process cat images
    cat_dir = os.path.join(source_dir, "PetImages/Cat")
    for file in os.listdir(cat_dir):
        if count >= limit:
            break
        if file.lower().endswith('.jpg'):
            path = os.path.join(cat_dir, file)
            features = extract_features(path)
            if features is not None:
                data.append(features)
                labels.append('cat')
                count += 1
    
    # Reset count for dogs
    count = 0
    
    # Process dog images
    dog_dir = os.path.join(source_dir, "PetImages/Dog")
    for file in os.listdir(dog_dir):
        if count >= limit:
            break
        if file.lower().endswith('.jpg'):
            path = os.path.join(dog_dir, file)
            features = extract_features(path)
            if features is not None:
                data.append(features)
                labels.append('dog')
                count += 1
    
    return np.array(data), np.array(labels)

def main():
    # Download and extract dataset
    download_dataset()
    
    print("Processing images...")
    
    # Process a balanced subset of images (1000 each of cats and dogs)
    data, labels = process_images("dataset_raw", limit=1000)
    
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    print("\nDataset shape:", df.shape)
    print("\nSample of the dataset:")
    print(df.head())
    
    # Split into train/test sets
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest classifier...")
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Test the model
    y_pred = clf.predict(X_test)
    
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save the model and feature extraction parameters
    model_data = {
        'model': clf,
        'image_size': (64, 64)
    }
    
    print("\nSaving model...")
    with open('cat_dog_classifier.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'cat_dog_classifier.pkl'")
    
    # Clean up downloaded files
    print("\nCleaning up downloaded files...")
    if os.path.exists("cats_and_dogs.zip"):
        os.remove("cats_and_dogs.zip")
    if os.path.exists("dataset_raw"):
        shutil.rmtree("dataset_raw")
    
    print("\nTraining completed! You can now use predict_image.py to classify new images.")

if __name__ == "__main__":
    main()