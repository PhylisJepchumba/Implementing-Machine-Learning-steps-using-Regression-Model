# Implementing-Machine-Learning-steps-using-Regression-Model
* We start by importing necessary modules as shown:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
```

* Then import the data.

```python
data=pd.read_csv('insurance.csv')
data
```

* Clean the data by removing duplicate values and transform the columns into numerical values to make the easier to work with.
```python
label=LabelEncoder()
label.fit(data.sex.drop_duplicates())
data.sex=label.transform(data.sex)

label.fit(data.smoker.drop_duplicates())
data.smoker=label.transform(data.smoker)

label.fit(data.region.drop_duplicates())
data.region=label.transform(data.region)
data
```

* Using the cleaned dataset, now split it into training and test sets.
```python
X=data.drop(['charges'], axis=1)
y=data[['charges']]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

```
* After splitting the model choose the suitable algorithm. In this case we will use Linear Regression since we need to predict a numerical value based on some parameters.

```python
model=LinearRegression())
model.fit(X_train,y_train)
```
* Now predict the testing dataset and find how accurate your predictions are.


![Screenshot (39)](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/x51y94yaklvpkhf40qhx.png)


* Accuracy score is predicted as follows:



![Screenshot (40)](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/70dlkdegc3efkj0m4r8x.png)

* parameter tuning
* 
Lets find the hyperparameters which affect various variables in the dataset.

![Screenshot (41)](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/6u4c37podaohcp8l1ckb.png)
 










