import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(35)
# seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，
# 如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。

x_train = [1, 2, 3, 4]
y_train = [1, 2, 3, 4]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.summary()
# prints summary of the model to the terminal

sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)
model.fit(x_train, y_train, epochs=200)

y_predict = model.predict(np.array([5]))
# 可以使用predict(预测)关键字对模型进行调用。
print(y_predict)
