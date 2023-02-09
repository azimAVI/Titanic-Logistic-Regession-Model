# Machine Learning LogisticRegression
This simple project is my first step on Machine Learning where I have learned how to implement Logistic Regression in Python for classification.

# Dataset

I have worked with the famous Titanic dataset from Kaggle which contains two different files, train.csv that contains the details of a subset of the passengers on board and importantly, will reveal whether they survived or not, and test.csv that contains similar information but does not disclose Survived information for each passenger. And this dataset is considered as the student’s first step in machine learning.



This code works on titanic.csv file, where we apply logistic regression and find out the accuracy of the data with X=("Age","Fare") and Y=("Survived").


# import dataset

```df=pd.read_csv("titanic.csv")```




# import librarires

```import plotly.express as px```

```import pandas as pd```

```import matplotlib.pyplot as plt```

# apply Logistic Regression

```from sklearn.model_selection import train_test_split```

```X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)```

```from sklearn.linear_model import LogisticRegression```

```model= LogisticRegression()```

```model.fit(X_train,Y_train)```

# Finding Accuracy

```from sklearn.metrics import accuracy_score```

```Y_predict= model.predict(X_test)```

```print(accuracy_score(Y_predict,Y_test))```

# 3D plot

```fig=px.scatter_3d(df, x='Age',y='Fare',z='Survived')```

```fig.show()```


# Logistic Regression

Logistic regression is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes (binary classification problem). The real life example of classification example would be, to categorize the mail as spam or not spam, to categorize the tumor as malignant or benign.

