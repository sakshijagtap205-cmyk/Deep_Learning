#----------------------------------------------------------------------------------
#   Deep Learning Pipeline for FNN Placement
#----------------------------------------------------------------------------------
# 1. Read the Data From CSv
# 2. Data Analysis (EDA)
# 3. PreProcessing
# 4. Train Test Split
# 5. Feature Scaling
# 6. FNN Model Training
# 7. Model Evaluation
# 8. Graphical Representation of the Model
# 9. Model Preserve
# 10. Model Loading and preserve
# 11. Test Unseen Data
#----------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.neural_network import MLPClassifier

#----------------------------------------------------------------------------------
# 1. Read the Data From CSV
#----------------------------------------------------------------------------------
print("Reading the Data From CSV")

data = pd.read_csv("placement_data.csv")

print("Data Read Successfully")

#----------------------------------------------------------------------------------
# 2. Data Analysis (EDA)
#----------------------------------------------------------------------------------
print("Performing Data Analysis (EDA)")

print("First 5 rows :")
print(data.head())

print("Columns Names:")
print(data.columns)

print("shape of Dataset")
print(data.shape)

print("statistical Summary:")
print(data.describe())

#----------------------------------------------------------------------------------
# 3. PreProcessing
#----------------------------------------------------------------------------------
print("Performing PreProcessing")

X = data[['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship']]

Y = data['Placed']

print("PreProcessing Completed")

print("Input features:")
print(X.head())

print("Target variable:")
print(Y.head())

#----------------------------------------------------------------------------------
# 4. Train Test Split
#----------------------------------------------------------------------------------

print("Performing Train Test Split")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

print("Training input shape:", X_train.shape)
print("Test input shape:", X_test.shape)

print("Training target shape:", Y_train.shape)
print("Test target shape:", Y_test.shape)

#----------------------------------------------------------------------------------
# 5. Feature Scaling
#---------------------------------------------------------------------------------
print("Performing Feature Scaling")

Scaler = StandardScaler()

X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

print("Feature Scaling Completed")
print(X_train_scaled[:5])

#----------------------------------------------------------------------------------
# 6. FNN Model Training
#---------------------------------------------------------------------------------

print("FNN Model Training")

model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation='relu',
    solver='adam',
    max_iter=1000, 
    random_state=42
)

print(model)

print("Train the model")
model.fit(X_train_scaled, Y_train)

print("Model Training Completed")

#----------------------------------------------------------------------------------
# 7. Model Evaluation
#---------------------------------------------------------------------------------
print("Model Evaluation")

Y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix:",cm)

print("Predict the probablity:")

Y_prob = model.predict_proba(X_test_scaled)
print(Y_prob[:5])


#----------------------------------------------------------------------------------
# 9. Model Preserve
#---------------------------------------------------------------------------------
print("Preserving the Model")

joblib.dump(model, 'fnn_placement_model.pkl')
joblib.dump(Scaler, 'scaler.pkl')

print("Model and scaler preserved successfully")

#---------------------------------------------------------------------------------
# 10. Model Loading and preserve
#---------------------------------------------------------------------------------

print("Loading the Model")

Loaded_model = joblib.load('fnn_placement_model.pkl')
Loaded_scaler = joblib.load('scaler.pkl')

print("Model gets loaded successfully")

#---------------------------------------------------------------------------------
# 11. Test unseen Data
# Aptitude       :  70
# Coding         :  75
# Communication  :  80
# Academics      :  85
# Internship     :  1
#---------------------------------------------------------------------------------

new_Student = pd.DataFrame([[70, 75, 80, 85, 1]], columns=['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship'])

new_Student_scaled = Loaded_scaler.transform(new_Student)

new_prediction = Loaded_model.predict(new_Student_scaled)

new_probablity = Loaded_model.predict_proba(new_Student_scaled)

print("New Students Data:")
print(new_Student)

print("Prediction probablity : ", new_probablity)

if new_prediction[0]==1:
    print("Prediction : placed")
else:
    print("Prediction: Not Placed")