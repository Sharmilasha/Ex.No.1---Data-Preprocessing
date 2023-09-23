# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
```
1.Importing the libraries

2.Importing the dataset

3.Taking care of missing data

4.Encoding categorical data

5.Normalizing the data

6.Splitting the data into test and train
```
## PROGRAM:
```
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
df.head()
le=LabelEncoder()
df["CustomerId"]=le.fit_transform(df["CustomerId"])
df["Surname"]=le.fit_transform(df["Surname"])
df["CreditScore"]=le.fit_transform(df["CreditScore"])
df["Geography"]=le.fit_transform(df["Geography"])
df["Gender"]=le.fit_transform(df["Gender"])
df["Balance"]=le.fit_transform(df["Balance"])
df["EstimatedSalary"]=le.fit_transform(df["EstimatedSalary"])
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)
df.duplicated()
print(df['Exited'].describe())
scaler= MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
x_train,x_test,y_train,x_test=train_test_split(X,Y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:

Printing first five rows of the dataset:
![Screenshot 2023-09-23 164855](https://github.com/Sharmilasha/Ex.No.1---Data-Preprocessing/assets/94506182/4cf5e079-cb89-4de4-9df0-e52dae58495f)

Separating x and y values:
![b](https://github.com/Sharmilasha/Ex.No.1---Data-Preprocessing/assets/94506182/2c675d50-6d84-4aad-9afb-12b81eb9be05)

Checking NULL value for the dataset:
![Screenshot 2023-09-23 165102](https://github.com/Sharmilasha/Ex.No.1---Data-Preprocessing/assets/94506182/0300a1de-e886-4073-a820-f152ffac6a17)

Column y and its description:
![Screenshot 2023-09-23 165306](https://github.com/Sharmilasha/Ex.No.1---Data-Preprocessing/assets/94506182/4193b3d2-8213-47a7-a23b-8fe24d2b4b0f)

Training Set:
![Screenshot 2023-09-23 165421](https://github.com/Sharmilasha/Ex.No.1---Data-Preprocessing/assets/94506182/3ef6bb5d-c329-4d82-b66d-903051b17418)

Testing Set and its length:
![Screenshot 2023-09-23 165524](https://github.com/Sharmilasha/Ex.No.1---Data-Preprocessing/assets/94506182/bf3b9c41-8385-46d7-ac9a-095208f02c21)




## RESULT
Hence the data preprocessing is done using the above code and data has been splitted into trainning and testing data for getting a better model.


