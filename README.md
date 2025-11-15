Emily Amos
# Cat vs Dog Image Classifier

A machine learning project that uses Random Forest classification to distinguish between images of cats and dogs. The project includes automatic dataset download, model training, and both command-line and web-based prediction interfaces.

## Features
- Automatic download of Microsoft's Cats and Dogs dataset
- Image preprocessing (grayscale conversion and resizing)
- Random Forest classifier training
- Model persistence
- Command-line prediction interface
- Web-based interface with image upload
- SQLite database for prediction history
- Confidence scores for predictions
- Recent predictions display

## Requirements
- Python 3.x
- Node.js and npm
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - Pillow (PIL)
- Required Node.js packages:
  - express
  - multer
  - sqlite3
  - ejs

## Installation
1. Clone this repository or download the source code
2. Install the required Python packages:
```bash
pip install numpy pandas scikit-learn Pillow
```
3. Install the required Node.js packages:
```bash
cd web
npm install
```

## Usage

### Training the Model
To train the model, run:

```bash
python train_classifier.py
```

This will:
1. Download the Microsoft Cats and Dogs dataset
2. Process the images (convert to grayscale and resize)
3. Train a Random Forest classifier
4. Save the trained model as `cat_dog_classifier.pkl`
5. Clean up downloaded files automatically

### Making Predictions
To classify a new image, use:

```bash
python predict_image_rf.py <path_to_image>
```

Example:
```bash
python predict_image_rf.py path/to/your/cat_or_dog_image.jpg
```

The script will output the prediction (CAT or DOG) along with a confidence score.

### Using the Web Interface

1. Start the web server:
```bash
cd web
npm start
```

2. Open your web browser and navigate to `http://localhost:3000`

3. Use the web interface to:
   - Upload images for classification
   - See prediction results with confidence scores
   - View recent predictions history
   - Preview images before classification

## Model Details
- Image preprocessing: 64x64 grayscale
- Algorithm: Random Forest Classifier
- Features: Flattened pixel values
- Training data: 2000 images (1000 each of cats and dogs)
- Test split: 20%

## Project Structure
```
├── train_classifier.py     # Training script
├── predict_image_rf.py     # Prediction script
├── cat_dog_classifier.pkl  # Saved model (after training)
├── dataset/               # Dataset directory (created during training)
│   └── test/
│       ├── cat/
│       └── dog/
└── web/                   # Web interface directory
    ├── server.js          # Express server
    ├── package.json       # Node.js dependencies
    ├── predictions.db     # SQLite database
    ├── public/           
    │   ├── css/          # Stylesheets
    │   └── js/           # Client-side JavaScript
    └── views/            
        └── index.html    # Web interface template
```

## Performance
The model typically achieves around 60-65% accuracy on the test set. Predictions include confidence scores to help assess the reliability of classifications.

## Notes
- The dataset is automatically downloaded and cleaned up during training
- Images are processed in grayscale to reduce complexity
- The model is saved with pickle for easy reuse
- Confidence scores are provided with each prediction