import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset = pd.read_csv("winequality-red.csv")

X = wine_dataset.drop("quality", axis=1)

print(X)

Y = wine_dataset["quality"].apply(lambda y_value: 1 if y_value >= 7 else 0)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(Y, X, test_size=0.2, random_state=3)

print(X.shape, X_train.shape, X_test.shape)

model = RandomForestClassifier()

model.fit(Y_train, X_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy : ", test_data_accuracy)

input_data = (7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")
