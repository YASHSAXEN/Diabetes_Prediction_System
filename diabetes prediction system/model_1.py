# important libraries
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib # it convert the data in the python program into serialise form and send the data one by one 

patients_data = pd.read_csv("diabetes.csv")
print(patients_data.head())

print(patients_data.info())

print(patients_data.describe())

print(patients_data.isnull().sum())

x = patients_data.iloc[:,:8].values # independent variable
y = patients_data["Outcome"].values # dependent variables(target variable)

# feature scaling 
scaled_x = StandardScaler()
scaled_x.fit(x)
x = scaled_x.transform(x)

# splitting the independent and dependent variables in 2 parts i.e training part and testing part
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

# building the model
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
model = SVC()
model.fit(x_train,y_train)

# checking the training accuracy of the model
y_train_predicted = model.predict(x_train)
training_accuracy = accuracy_score(y_train,y_train_predicted)
print(training_accuracy)

# checking the testing accuracy of the model
y_test_predicited = model.predict(x_test)
testing_accuracy = accuracy_score(y_test,y_test_predicited)
print(testing_accuracy)

# checking the error in the result
mse = mean_squared_error(y_train,y_train_predicted)
rmse = np.sqrt(mse)
print(rmse)

# cross validate the model and checking mean error as well as the standard deviation of error in the model
score = cross_val_score(model,x_train,y_train,scoring="neg_mean_squared_error",cv=10)
rmse_score = np.sqrt(-score)
print("rmse_score:\n",rmse_score)
print("rmse_mean:",rmse_score.mean())
print("rmse_std:",rmse_score.std())

# model testing
input_data = [8,183,64,0,0,23.3,0.672,32]
input_data_as_array = np.array(input_data)
input_data_reshape = input_data_as_array.reshape(1,-1)
std_data = scaled_x.transform(input_data_reshape)
output = model.predict(std_data)
print(output)

joblib.dump(model,"model_4.pkl")

model = joblib.load("model_4.pkl")
model.predict([np.array([4,110,92,0,0,37.6,0.191,30])])