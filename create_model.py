import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# read data
X = pd.read_csv("./biomech_data.csv")

# Getting only fastball data
X = X.loc[X["pitch_type"] == "FF"]

# drop the session, session_pitch, p_throws, and pitch_type, as these
# features are here to describe meta features of the throw, and 
# will not affect the pitch speed
X.drop(["session", "session_pitch", "p_throws", "pitch_type"], axis=1, inplace=True)

# seperate our features and target, target will be pitch speed
y = X.pop("pitch_speed_mph")

# define the model
model = XGBRegressor(random_state=0)

# split data into training and test subsets,
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2,random_state=0)

# fit the model
model.fit(X_train, y_train)

# get predictions
predictions = model.predict(X_test)

# check performance with mean absolute error as a metric
mae = mean_absolute_error(predictions, y_test)

print("Mean Absolute Error:" , mae)