# 🍎 Fruit Disease Detection using Image Preprocessing and SVM

This project detects diseases in various fruits using **image preprocessing**, **KMeans segmentation**, **feature extraction**, and classification using an **SVM (Support Vector Machine)** model. It includes a **Graphical User Interface (GUI)** built with **Tkinter**, allowing users to perform all steps interactively.

---

## 🧠 Features

- Upload fruit leaf dataset
- Apply image preprocessing (resize, normalize, clean)
- Perform **KMeans segmentation** to highlight affected areas
- Extract features (color histogram, texture, etc.)
- Train an **SVM classifier** with the extracted features
- Upload and classify test images
- View results in a user-friendly desktop GUI

---

## 🖥️ GUI Overview

The GUI includes the following buttons:

- 🔹 **Upload Fruits Dataset**  
- 🔹 **Image Preprocessing & KMEANS Segmentation**  
- 🔹 **Features Extraction**  
- 🔹 **Train SVM Classifier**  
- 🔹 **Upload Test Image & Classification**

> All actions are handled via the interface — no need to touch the code while testing!

---

## 📂 Folder Structure

- `FruitDataset/` – Training images grouped by disease type  
- `testImages/` – Images used to test the classifier  
- `features/` – Preprocessed and feature-extracted data  
- `FruitDiseaseClassification.py` – Main script (includes GUI and ML pipeline)  

---

## 🧰 Technologies Used

- Python
- OpenCV
- NumPy
- scikit-learn
- Tkinter

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
python FruitDiseaseClassification.py
