import numpy as np
from stradimod.utils import sigmoid, relu, tanh
from stradimod.netwok import Network
from stradimod.layers import Dense
from load_dataset import load_data

np.random.seed(1)

x_train, y_train, x_test, y_test, classes = load_data()

# === TEST 1 =====

model = Network()
model.add(Dense(5, relu))
model.add(Dense(5, relu))
model.add(Dense(1, sigmoid))
model.build(12288)
model.train(x_train, y_train, epochs=3000, learning_rate=0.03)
accuracy = model.predict(x_test, y_test)
print(f"Accuracy: {accuracy*100}%")
print("")

# === TEST 2 ====

model = Network()
model.add(Dense(5, tanh))
model.add(Dense(5, tanh))
model.add(Dense(1, sigmoid))
model.build(12288)
model.train(x_train, y_train, epochs=3000, learning_rate=0.03)
accuracy = model.predict(x_test, y_test)
print(f"Accuracy: {accuracy*100}%")
print("")

# === TEST 3 ====

model = Network()
model.add(Dense(20, relu))
model.add(Dense(7, relu))
model.add(Dense(5, relu))
model.add(Dense(1, sigmoid))
model.build(12288)
model.train(x_train, y_train, epochs=500, learning_rate=0.05)
model.train(x_train, y_train, epochs=500, learning_rate=0.009)
model.train(x_train, y_train, epochs=1000, learning_rate=0.005)
model.train(x_train, y_train, epochs=2000, learning_rate=0.006)
accuracy = model.predict(x_test, y_test)
print(f"Accuracy: {accuracy*100}%")
