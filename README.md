# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start

2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

6. Stop

## Program:
#### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:SARWESHVARAN A

RegisterNumber:212223230198

```python
import pandas as pd
data=pd.read_csv('Placement_data.csv')
data.head(5)
```
![image](https://github.com/user-attachments/assets/34c473cd-32c9-4e74-8ff9-b7b60a620a0f)

```python
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
```

![image](https://github.com/user-attachments/assets/791e3cb1-7531-4fe9-a62e-d16733c6f461)
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
![image](https://github.com/user-attachments/assets/4459614a-c7b5-40c7-bdc1-3b19e268ff1c)

```python
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/afaf58fa-2de7-485a-8d71-77bdf26ee47b)

```python
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/69698e06-74e5-45eb-9e29-f0b3aa66e3af)

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/12d8cddf-3d3a-43a8-af6e-b9acae252166)

```python
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy=",accuracy)
```
![image](https://github.com/user-attachments/assets/df6b919e-d4fe-40f2-81ed-738e156dbaf0)

```python
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix
```
![image](https://github.com/user-attachments/assets/ac2544fe-e47a-48ad-89e9-c0729b8df303)

```python
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/4ea01799-6255-4f31-8717-71c479b8b2a3)

```python
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/9189be3a-46aa-42ac-bdb9-da58fcff0d81)











## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
