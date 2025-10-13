# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

. # Algorithm
1. **Data Loading & Preprocessing**  
   - Read the dataset using `pandas.read_csv()`.  
   - Inspect the data with `.head()`, `.info()`, and `.isnull().sum()` to understand structure and check for missing values.  
   - Encode the categorical column `"salary"` to numeric using `LabelEncoder`.

2. **Feature and Target Selection**  
   - Define the feature matrix **X** with columns:  
     `["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]`.  
   - Define the target vector **y** as the `"left"` column.

3. **Model Training & Prediction**  
   - Split the dataset into training and testing sets using `train_test_split`.  
   - Create a `DecisionTreeClassifier` with `criterion="entropy"` and fit it on the training data.  
   - Predict the target labels for the test set using the trained model.

4. **Model Evaluation & Single Prediction**  
   - Compute and display metrics such as `accuracy_score`, `confusion_matrix`, and `classification_report`.  
   - Make a sample prediction for a new employee record using `dt.predict([[0.5,0.8,9,260,6,8,1,2]])`.

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: CHARUKESH S
RegisterNumber:  212224230044
*/

import pandas as pd
df=pd.read_csv("Employee.csv")

df.head()

df.info()

df.isnull().sum()1

df.notnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()

df["left"].value_counts()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("NAME:CHARUKESH S")
print("REG NO:212224230044")
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("NAME:CHARUKESH S")
print("REG NO:212224230044")
accuracy

from sklearn.metrics import confusion_matrix
confuse = confusion_matrix(y_test, y_pred)
print("NAME:CHARUKESH S")
print("REG NO:212224230044")
print(confuse)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print("NAME:CHARUKESH S")
print("REG NO:212224230044")
print(report)

dt.predict([[0.5,0.8,9,260,6,8,1,2]])
```

## Output:
<img width="1262" height="232" alt="image" src="https://github.com/user-attachments/assets/28b360eb-37fe-438d-b3f5-7ef6084d2269" />

<img width="667" height="387" alt="image" src="https://github.com/user-attachments/assets/ba38e351-59cc-422e-9641-1aa41a599122" />

<img width="617" height="259" alt="image" src="https://github.com/user-attachments/assets/d998c660-9120-470d-b228-5e26b192d3ed" />

<img width="562" height="256" alt="image" src="https://github.com/user-attachments/assets/a36bfe14-e3b0-44c0-b485-e6caa6fe1637" />

<img width="1268" height="239" alt="image" src="https://github.com/user-attachments/assets/45100f52-2e30-41ce-8860-4e0c896a6755" />

<img width="904" height="90" alt="image" src="https://github.com/user-attachments/assets/0e9dc057-e463-4d18-b265-6e8d4154d447" />

<img width="1188" height="216" alt="image" src="https://github.com/user-attachments/assets/f3b91fdb-3639-4239-b086-c7af74f449a1" />

<img width="728" height="152" alt="image" src="https://github.com/user-attachments/assets/d255d1ee-372e-4e43-a782-ee83d403b4a3" />

<img width="772" height="110" alt="image" src="https://github.com/user-attachments/assets/318aba9c-b34c-436c-8009-943921e476dc" />

<img width="793" height="104" alt="image" src="https://github.com/user-attachments/assets/9b0fc3aa-26a5-4965-9f1b-90fe306274fb" />

<img width="627" height="115" alt="image" src="https://github.com/user-attachments/assets/7d03928c-3d0b-4d65-b06b-8089ab9cceed" />

<img width="793" height="256" alt="image" src="https://github.com/user-attachments/assets/87f980b6-3de1-4ddd-9fca-0d74744a94c7" />

<img width="1260" height="134" alt="image" src="https://github.com/user-attachments/assets/5fa54385-415a-455a-b5d9-12e9e95dc5d7" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
