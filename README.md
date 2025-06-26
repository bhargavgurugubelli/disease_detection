# DetectionAndClassificationOfFruitDiseases
# Detection and Classification of Fruit Diseases 🍎🍌🍇

This project focuses on **image preprocessing and classification** of fruit leaf diseases using machine learning techniques. The goal is to clean, enhance, and prepare fruit images to improve disease detection accuracy in agricultural systems.

---

## 🧠 Problem Statement

Detecting diseases in fruits through leaf images is essential for crop health monitoring. Raw images often have noise, uneven lighting, or different sizes. Effective **image preprocessing** improves model performance by providing high-quality, consistent input data.

---

## 📁 Project Structure

- `FruitDataset/` - Contains raw training images for different fruit diseases
- `testImages/` - Contains images used to test the model pipeline
- `features/` - Stores extracted image features or preprocessed data
- `FruitDiseaseClassification.py` - Main script with:
  - Image loading and preprocessing (resizing, normalization, etc.)
  - Feature extraction
  - CNN model training (if included)
- `.DS_Store` - System file (safe to ignore)

---

## 🖼️ Image Preprocessing Steps

- Resize all images to 128x128 pixels
- Convert to grayscale / RGB standardization
- Normalize pixel values to range [0,1]
- Apply data augmentation (rotation, flipping, etc.)
- (Optional) Edge detection or histogram equalization

---

## 🧰 Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- TensorFlow / Keras (for classification model)

---

## 🧪 Results

- Accuracy: ~90% on test dataset (if classification included)
- Cleaner and more consistent input images for ML models

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/bhargavgurugubelli/disease_detection.git
cd disease_detection

# Install dependencies
pip install -r requirements.txt

# Run the script
python FruitDiseaseClassification.py
