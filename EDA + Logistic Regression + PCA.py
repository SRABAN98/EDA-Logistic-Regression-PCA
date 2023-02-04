#import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#ignore the warnings
import warnings
warnings.filterwarnings("ignore")


#import the dataset
df = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\28th,29th\adult.csv")


#Exploratory Data Analysis
df.shape
df.info()
df.head()
df.tail()
df[df == "?"] = np.nan
df.info()
df.isnull().sum()
for col in ["workclass", "occupation", "native.country"]:
    df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum()


#splitting the dataset into I.V and D.V as "x" and "y"
x = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values


#impute categorical values of independent variables(variable transformation)
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()


labelencoder_x.fit_transform(x[:,1])
x[:,1] = labelencoder_x.fit_transform(x[:,1])


labelencoder_x.fit_transform(x[:,3])
x[:,3] = labelencoder_x.fit_transform(x[:,3])


labelencoder_x.fit_transform(x[:,5])
x[:,5] = labelencoder_x.fit_transform(x[:,5])


labelencoder_x.fit_transform(x[:,6])
x[:,6] = labelencoder_x.fit_transform(x[:,6])


labelencoder_x.fit_transform(x[:,7])
x[:,7] = labelencoder_x.fit_transform(x[:,7])


labelencoder_x.fit_transform(x[:,8])
x[:,8] = labelencoder_x.fit_transform(x[:,8])


labelencoder_x.fit_transform(x[:,9])
x[:,9] = labelencoder_x.fit_transform(x[:,9])


labelencoder_x.fit_transform(x[:,13])
x[:,13] = labelencoder_x.fit_transform(x[:,13])


#impute categorical values of dependent variables(variable transformation)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labelencoder_y.fit_transform(y)
y = labelencoder_y.fit_transform(y)


#scale the data(feature scalling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


#split the data into training and testing phase(train and test data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state = 0)


#training the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


#predicting the test set results
y_pred = classifier.predict(x_test)


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)


#calculating the bias of the model
bias = classifier.score(x_train,y_train)
bias


#calculating the variance of the model
variance = classifier.score(x_test,y_test)
variance


#Explained Variance Ratio
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum >= 0.90) + 1
print("the number of dimensions required to preserve 90% of variance is",dim)


#find out the variance ratio explained
x_train = pca.fit_transform(x_train)
pca.explained_variance_ratio_


#As for getting the highest accuracy from the given dataset we need only 12 features, so
#we can remove "native.country" and "hours.per.week" features from the dataset, because these features
#are having less variance explained as compared to the other features.
