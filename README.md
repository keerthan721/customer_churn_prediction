Customer Churn Prediction using Deep Learning 📉🧠

This project predicts customer churn for a telecom company using an Artificial Neural Network (ANN).
It is a classic binary classification problem where the goal is to identify whether a customer will leave (churn) or stay based on their profile and behavior patterns.

🌍 Geographies Covered: Hyderabad, Mumbai, Bengaluru

🔍 Problem Statement

Telecom companies often face challenges in retaining customers.
Churn prediction enables businesses to identify high-risk customers and take proactive actions to improve retention.
In this project, we use a deep learning model to predict churn based on various customer attributes.

✅ Features Used

Geography (Hyderabad, Mumbai, Bengaluru)

Gender

Age

Tenure

Balance

Number of Products

Credit Score

Active Membership

Estimated Salary

Has Credit Card

🧠 Model Overview

Frameworks: TensorFlow, Keras

Model Type: Artificial Neural Network (ANN)

Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation Metric: Accuracy

🧪 Steps Performed

Data Cleaning & Preprocessing

Label Encoding for Categorical Variables

Feature Scaling using StandardScaler

Splitting the Dataset into Training and Testing

Building and Compiling the ANN

Training the Model

Evaluating the Performance

📁 Folder Structure
Customer-Churn-Prediction/
│
├── Churn_Modelling.csv            # Dataset
├── experiment.ipynb               # Jupyter Notebook with model code
├── README.md                      # Project Overview
├── .gitignore                     # Files and folders ignored by Git
└── venv/                          # Virtual environment (should be ignored)

🚀 How to Run Locally
# 1. Clone the repository
git clone https://github.com/saiengineerss/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

🌐 Live Demo

You can try the deployed app here 👉 Customer Churn Prediction App

📸 Screenshots / Demo

Here’s how the deployed app looks:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/310e73ea-8d62-4e7c-b724-a2e3374dca9c" />


📊 Model Performance

Accuracy Score: ~85%

Evaluation done using:

Confusion Matrix

Classification Report

🙌 Acknowledgements

Dataset inspired by publicly available Kaggle datasets

Built using:

Python

Pandas & NumPy

Scikit-learn

TensorFlow & Keras

Streamlit

🛑 Note

Ensure the venv/ folder is excluded from the repository using .gitignore.

This project is for educational purposes.
