# MACHINE-LEARNING-MODEL-IMPLEMENTATION
CodTech Task 4

*COMPANY*: CODTECH IT SOLUTIONS,

*NAME*: DHARSHANA DEVI S,

*INTERN ID*: CTIS6407,

*DOMAIN*: PYTHON PROGRAMMING,

*DURATION*: 4 WEEKS,

*MENTOR*: NEELA SANTHOSH KUMAR

#DESCRIPTION OF THE TASK 4

# 📧 Spam Email Detection using Machine Learning

## 📌 Project Overview

This project implements a **Machine Learning model** to classify emails or messages as **Spam** or **Ham (Not Spam)**.
The model uses **Natural Language Processing (NLP)** techniques and **Scikit-learn** algorithms to analyze message content and predict whether it is spam.

This project was developed as part of the **Machine Learning Model Implementation Task** for the internship program at **CodTech Solutions**.

## 🎯 Objective

The objective of this project is to:

* Build a **predictive machine learning model**
* Perform **text preprocessing and feature extraction**
* Train a **classification model**
* Evaluate model performance using standard ML metrics
* Predict whether a message is **Spam or Not Spam**

## 🛠 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Matplotlib**
* **Seaborn**

## 📂 Project Structure

Codtech_Task4
│
├── spam_detection.py      # Main machine learning code
├── spam.csv               # Dataset
├── README.md              # Project documentation

## 📊 Dataset

The dataset used in this project is the **SMS Spam Collection Dataset**, which contains labeled SMS messages categorized as:

* **Ham** – Legitimate message
* **Spam** – Unwanted promotional message

Each record contains:

* Label (spam/ham)
* Message text

## ⚙️ Implementation Steps

### 1️⃣ Data Loading

The dataset is loaded using **Pandas**.

### 2️⃣ Data Preprocessing

Text messages are cleaned using:

* Lowercasing
* Removing punctuation
* Removing stopwords

### 3️⃣ Feature Extraction

Text data is converted into numerical form using **TF-IDF Vectorization**.

### 4️⃣ Data Splitting

The dataset is split into:

* **Training data (80%)**
* **Testing data (20%)**

### 5️⃣ Model Training

A **Multinomial Naive Bayes classifier** is used to train the spam detection model.

### 6️⃣ Model Evaluation

The model performance is evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix

### 7️⃣ Prediction

The trained model can predict whether a **new message** is spam or not.

## 📈 Results

The model achieved an accuracy of approximately:

**~97% Accuracy**

This demonstrates that the machine learning model is effective at identifying spam messages.

## ▶️ How to Run the Project

### Step 1 – Install Dependencies

pip install pandas numpy scikit-learn matplotlib seaborn nltk


### Step 2 – Download Dataset

Download the dataset and place **spam.csv** in the project folder.

### Step 3 – Run the Program

python spam_detection.py

## 🧪 Example Prediction

Input Message:

Congratulations! You have won a free prize

Prediction:

Spam

OUTPUT:

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/bd70484b-7fce-4c62-945e-257848036d62" />

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/0b8ec0db-7b7c-4528-8eb9-21ba6cb2190a" />
