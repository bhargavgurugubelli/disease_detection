from tkinter import messagebox
from tkinter import *
#from tkFruitDiseaseClassification.pyFruitDiseaseClassification.pyinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


main = tkinter.Tk()
main.title("DETECTION AND CLASSIFICATION OF FRUIT DISEASES")  # designing main screen
main.geometry("1300x1200")

fruits_disease = ['Black spot', 'Canker', 'Greening', 'healthy', 'scab']

global filename
global X_train, X_test, y_train, y_test


def uploadFruitDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")


def Preprocessing():
    global X, Y
    X, Y = [],[]

    for i in range(len(fruits_disease)):
        for root, dirs, directory in os.walk('FruitDataset/'+fruits_disease[i]):
            for j in range(len(directory)):
                img = cv2.imread('FruitDataset/'+fruits_disease[i]+"/"+directory[j])
                img = cv2.resize(img,(128,128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pixel_vals = img.reshape((-1,3))
                pixel_vals = np.float32(pixel_vals)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
                retval, labels, centers = cv2.kmeans(pixel_vals, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                X.append(segmented_data.ravel())
                Y.append(i)
                print('FruitDataset/'+fruits_disease[i]+"/"+directory[j]+" "+str(X[j].shape))
    np.save("features/features.txt.npy", X)
    np.save("features/labels.txt.npy", Y)

    X = np.load("features/features.txt.npy")
    Y = np.load("features/labels.txt.npy")
    text.insert(END, "Total preprocess images are : " + str(X.shape[0]) + "\n\n")

def featuresExtraction():
    global X_train, X_test, y_train, y_test
    global X, Y
    ind = random.randint(0, (len(X) - 1))
    img = X[ind].reshape(128, 128, 3)
    cv2.imshow('Image after KMEANS & Feature Extraction', cv2.resize(img, (450, 450)))
    cv2.waitKey(0)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END, "Features Extraction Process Completed\n")
    text.insert(END, "Total images used to train SVM is : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total images used to test  SVM is : " + str(X_test.shape[0]) + "\n\n")

def svmClassifier():
    global classifier
    cls = svm.SVC(C=12, gamma='scale', kernel='rbf', random_state=0)
    cls.fit(X, Y)
    prediction = cls.predict(X_test)
    svm_acc = accuracy_score(y_test, prediction) * 100
    text.insert(END, "SVM Accuracy : " + str(svm_acc) + "\n")
    cm = confusion_matrix(y_test, prediction)
    total = sum(sum(cm))
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    text.insert(END, 'SVM Algorithm Specificity : ' + str(specificity * 100) + "\n")
    classifier = cls

def Classification():
    name = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(name)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    temp = []
    temp.append(segmented_data.ravel())
    temp = np.array(temp)
    predict = classifier.predict(temp)[0]
    img = cv2.imread(name)
    img = cv2.resize(img, (500, 500))
    disease = 'Disease'
    if fruits_disease[predict] == 'healthy':
        disease = ''
    cv2.putText(img, 'Fruit classify as ' + fruits_disease[predict] + " " + disease, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    cv2.imshow("Classification Result : " + 'Fruit classified as ' + fruits_disease[predict] + " " + disease, img)
    cv2.waitKey(0)

def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(main, text='DISEASE DETECTION OF VARIOUS FRUITS USING IMAGE PREPROCESSING')
title.config(bg='white', fg='Black')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Fruits Dataset", command=uploadFruitDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

processButton = Button(main, text="Image Preprocessing & KMEANS Segmentation", command=Preprocessing)
processButton.place(x=250, y=550)
processButton.config(font=font1)

featuresButton = Button(main, text="Features Extraction", command=featuresExtraction)
featuresButton.place(x=650, y=550)
featuresButton.config(font=font1)


svmButton = Button(main, text="Train SVM Classifier", command=svmClassifier)
svmButton.place(x=50, y=600)
svmButton.config(font=font1)


classifyButton = Button(main, text="Upload Test Image & Classification", command=Classification)
classifyButton.place(x=250, y=600)
classifyButton.config(font=font1)

main.config(bg='white')
main.mainloop()
