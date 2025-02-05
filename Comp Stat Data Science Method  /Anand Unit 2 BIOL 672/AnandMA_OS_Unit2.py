## AnandMA_unit2_BIOL672.py
# Operating system : macOS Ventura Version 13.6.9
# install necessary packages
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

data = pd.read_csv("/Users/maya.anand/Desktop/Anand Comp Stat BIOL 672/Anand Unit 2 Coding BIOL 672/data.csv")

print("DatasetPreview:")
print(data.head()) #display first few rows of the dataset

data = data.drop(["id", "Unnamed: 32"], axis=1) #drop the 'id' column as it is not necessary and irrelevant
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1}) #convert 'diagnosis' column to numerical values: 0 for Benign, 1 for Malignant
print("\nDataset Information:")
print(data.info())

print("\nMissing Values Per Column:") # Check for missing values
print(data.isnull().sum())
data = data.fillna(data.mean())# Fill missing values with the mean of each column (if any)

X = data.drop(['diagnosis'], axis = 1) #features
y = data['diagnosis'] #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split the dataset

print("\nSample of X_train with NaNs:")
print(X_train.isnull().sum())  # Summarize NaNs in each column of X_train

print("\nSample of X_test with NaNs:")
print(X_test.isnull().sum())  # Summarize NaNs in each column of X_test

# ------ PROMPT 1: Simpler Methods -------
knn = KNeighborsClassifier(n_neighbors=5) # apply ML model K-Nearest Neighbors (KNN)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("Confusion Matrix (KNN):")
print(confusion_matrix(y_test, y_pred_knn))

nb = GaussianNB() # Naive Bayes
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix (Naive Bayes):")
print(confusion_matrix(y_test, y_pred_nb))


lda = LinearDiscriminantAnalysis() # Linear Discriminany Analysis (LDA)
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
print("\nLDA Classification Report:")
print(classification_report(y_test, y_pred_lda))
print("Confusion Matrix (LDA):")
print(confusion_matrix(y_test, y_pred_lda))

qda =QuadraticDiscriminantAnalysis() # Quadratic Discriminant Analysis (QDA)
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
print("\nQDA Classification Report:")
print(classification_report(y_test, y_pred_qda))
print("Confusion Matrix (QDA):")
print(confusion_matrix(y_test, y_pred_qda))

print("Columns in X_test:") #check column names for accuracy
print(X_test.columns)

plt.figure(figsize=(8, 6)) ## Visualize KNN results using a scatter plot (e.g., 'mean radius' vs. 'mean texture')
plt.scatter(X_test['radius_mean'], X_test['texture_mean'], c=y_pred_knn,cmap='viridis', edgecolor='k', s=100)
plt.title("KNN Predictions (Mean radius vs. Mean Texture")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.colorbar(label="Predicted Diagnosis")
plt.show()

scores = cross_val_score(knn, X, y, cv=5)
print(f"\n5-Fold Cross-Validation Scores for KNN: {scores}")
print(f"Mean CV Score: {scores.mean()}")

# --- Optional: Unsupervised Clustering (EM Clustering) ---
em = GaussianMixture(n_components=2, random_state=42)
em.fit(X)
cluster_labels = em.predict(X)

plt.figure(figsize=(8, 6)) # Scatter plot for EM clustering ('radius_mean' vs. 'texture_mean')
plt.scatter(X['radius_mean'], X['texture_mean'], c=cluster_labels, cmap='viridis', edgecolor='k', s=100)
plt.title("EM Clustering (Radius Mean vs. Texture Mean)")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.colorbar(label="Cluster Label")
plt.show()

# ------ PROMPT 2: Support Vector Machines ------
kernels = ['linear', 'poly', 'rbf'] # SVM with Different Kernals (Linear, Polynomial, Radial Basis Functions)
results = {}

for kernel in kernels:
    print(f"\nRunning SVM with {kernel} kernel...")

    svm_model = SVC(kernel=kernel, probability=True, random_state=42) # initialize and train the SVM model
    svm_model.fit(X_train, y_train)

    y_pred =svm_model.predict(X_test) # make predictions
    accuracy = accuracy_score(y_test, y_pred)
    results[kernel] = accuracy

    print(f"\nClassification Report for {kernel} kernel:") # classification report
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix for {kernel} kernel:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy for {kernel} kernel: {accuracy:.2%}")

    plt.figure(figsize=(8,6)) # scatterplot visualization
    plt.scatter(X_test['radius_mean'], X_test['texture_mean'], c=y_pred, cmap='viridis', edgecolor='k', s=100)
    plt.title(f"SVM Predictions with{kernel} Kernel (Radius Mean vs. Texture Mean)")
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.colorbar(label="Predicted Diagnosis")
    plt.show()

print("\nSVM Kernel Comparison:") #compare results
for kernel, accuracy in results.items():
    print(f"{kernel.capitalize()} Kernel Accuracy: {accuracy:.2%}")

print("\nPerforming 5-Fold Cross-Validation for RBF Kernel...") # perform cross-validation for SVM with RBF kernel
svm_rbf = SVC(kernel= 'rbf', random_state=42)
cross_val_scores = cross_val_score(svm_rbf, X, y, cv=5)
print(f"Cross-Validation Scores for RBF Kernel: {cross_val_scores}")
print(f"Mean Cross-Validation Score for RBF Kernel: {cross_val_scores.mean():.2%}")

knn_accuracy = accuracy_score(y_test, y_pred_knn)  # comparison with simpler methods from Prompt 1
print("\nComparison with Simpler Methods (Prompt 1):")
print(f"KNN Accuracy: {knn_accuracy:.2%}")
print(f"SVM Linear Kernel Accuracy: {results['linear']:.2%}")
print(f"SVM RBF Kernel Accuracy: {results['rbf']:.2%}")

# ------PROMPT 3: Artificial Neural Networks------
ann = MLPClassifier(hidden_layer_sizes = (100,), max_iter=500, random_state =42) #Simple ANN
ann.fit(X_train, y_train)
y_pred_ann = ann.predict(X_test) # make predictions with the ANN

ann_accuracy = accuracy_score(y_test, y_pred_ann) #evaluate the ANN
print("\nClassification Report for ANN:")
print(classification_report(y_test, y_pred_ann))
print("Confusion Matrix for ANN:")
print(confusion_matrix(y_test, y_pred_ann))
print(f"Accuracy for ANN: {ann_accuracy:.2%}")

print("\nComparison of SVM and ANN:") # compare with SVM results
print(f"SVM RBF Kernel Accuracy: {results['rbf']:.2%}")
print(f"ANN Accuracy: {ann_accuracy:.2%}")

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data() #Deep learning with Keras
X_train_mnist = X_train_mnist / 255.0
X_test_mnist = X_test_mnist / 255.0
y_train_mnist = to_categorical(y_train_mnist)
y_test_mnist = to_categorical(y_test_mnist)

def build_and_train_model(num_layers):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))

    for _ in range(num_layers):
        model.add(Dense(128, activation='relu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(X_train_mnist, y_train_mnist, epochs=5, batch_size=128, verbose=0)
    elapsed_time = time.time() - start_time
    
    test_loss, test_accuracy = model.evaluate(X_test_mnist, y_test_mnist, verbose=0)
    return test_accuracy, elapsed_time 

layers = [1,2,3,4,5]
layer_performance = []
layer_speed = []

for num_layers in layers:
    accuracy, elapsed_time =  build_and_train_model(num_layers)
    layer_performance.append(accuracy)
    layer_speed.append(elapsed_time)
    print(f"Layers: {num_layers}, Accuracy: {accuracy:.2%}, Time: {elapsed_time:.2f}s")

plt.figure(figsize=(10, 6)) # Plot Performance vs. Layers
plt.plot(layers, layer_performance, marker='o')
plt.title("Model Performance vs. Number of Layers")
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6)) # Plot Speed vs. Layers
plt.plot(layers, layer_speed, marker='o', color='r')
plt.title("Model Training Time vs. Number of Layers")
plt.xlabel("Number of Layers")
plt.ylabel("Training Time (seconds)")
plt.grid()
plt.show()

# ------ PROMPT 4: Random Forest and AdaBoost Classifiers ------
# 1. Random Forest Classifier
print("\nRunning Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))
print(f"Accuracy for Random Forest: {rf_accuracy:.2%}")

# 2. AdaBoost Classifier
print("\nRunning AdaBoost Classifier...")
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

ada_accuracy = accuracy_score(y_test, y_pred_ada)
print("\nClassification Report for AdaBoost:")
print(classification_report(y_test, y_pred_ada))
print("Confusion Matrix for AdaBoost:")
print(confusion_matrix(y_test, y_pred_ada))
print(f"Accuracy for AdaBoost: {ada_accuracy:.2%}")

# 3. Comparison of Models
print("\nComparison of Random Forest and AdaBoost:")
print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
print(f"AdaBoost Accuracy: {ada_accuracy:.2%}")

# 4. Visualize the Results
plt.figure(figsize=(10, 6))  # Compare Model Accuracy
models = ['Random Forest', 'AdaBoost']
accuracies = [rf_accuracy, ada_accuracy]
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.grid(axis='y')
plt.show()

# 5. Perform Cross-Validation for Robust Evaluation
from sklearn.model_selection import cross_val_score

rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
ada_cv_scores = cross_val_score(ada_model, X, y, cv=5)

print("\nCross-Validation Scores:")
print(f"Random Forest: {rf_cv_scores}")
print(f"Random Forest Mean CV Accuracy: {rf_cv_scores.mean():.2%}")
print(f"AdaBoost: {ada_cv_scores}")
print(f"AdaBoost Mean CV Accuracy: {ada_cv_scores.mean():.2%}")