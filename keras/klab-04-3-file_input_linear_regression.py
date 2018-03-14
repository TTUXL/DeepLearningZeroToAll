from keras.models import Sequential
from keras.layers import Dense
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',')

x_data = xy[:, 0:-1]   #从0列 至 倒数1列，x=[?,?,?]
y_data = xy[:, [-1]]   #只是最后一列     y=?

print("x_data", x_data)
print("y_data", y_data)

model = Sequential()
model.add(Dense(input_dim=3, units=1))

model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_data, y_data, epochs=2000)

print("0, 2, 1", model.predict(np.array([[0, 2, 1]])))
print("0, 9, -1", model.predict(np.array([[0, 9, -1]])))
