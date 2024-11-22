# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.
2. Load the Iris dataset.
3. Convert the dataset into a pandas DataFrame and add the target column.
4. Split the data into features (x) and target (y).
5. Divide the dataset into training and testing sets using train_test_split().
6. Create an SGDClassifier instance with default parameters.
7. Train the classifier on the training data using .fit().
8. Make predictions on the test data using .predict().
9. Calculate and print the accuracy using accuracy_score().
10. Compute and display the confusion matrix using confusion_matrix().
11. Display the classification report

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VARSHA S
RegisterNumber:  212222220055
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target'] = iris.target

print(df.head())

x = df.iloc[:, :-1]
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)

y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :",accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n",cm)

report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:\n", report)
```

## Output:
![image](https://github.com/user-attachments/assets/13b36358-3bbb-42b3-9ec0-0351082ff662)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
