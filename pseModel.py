import tensorflow
import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("stock_exchange.csv")
data = data[["c", "h", "l", "o", "wd", "last"]]

predict = "c"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

pickle_rick = open("pseModel.pickle", "rb")
linear = pickle.load(pickle_rick)

print("co: ", linear.coef_)
print("intercept: ", linear.intercept_)
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("The predicted value: ", predictions[x], "\nThe value: ", y_test[x], "\nThe indicators: ", x_test[x],"\n")

#uncomment if u want some graphs boiiii
'''
p = 'o'
style.use("ggplot")
pyplot.scatter(data[p],data["c"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
'''
