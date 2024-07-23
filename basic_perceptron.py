import numpy as np

# Define the activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Define the perceptron model
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size + 1)  # Including bias
        self.lr = lr

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert bias
        activation = np.dot(self.weights, x)
        return step_function(activation)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Insert bias
                y_hat = self.predict(X[i])
                self.weights += self.lr * (y[i] - y_hat) * x_i

# Define the dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 0, 0, 0])

# Initialize and train the perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)

# Test the perceptron
for x in X:
    print(f"Input: {x} - Predicted: {perceptron.predict(x)}")