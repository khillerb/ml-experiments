import numpy as np
import matplotlib.pyplot as plt
import random


def mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted probabilities
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_stable(x):
    # Compute the sigmoid function using log tricks for numerical stability
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


class LinearSigmoidRegression:
    def __init__(self, num_linear_funcs):
        self.num_linear_funcs = num_linear_funcs
        self.weights = [random.randrange(-1, 1)] * num_linear_funcs
        self.biases = [random.randrange(-1, 1)] * num_linear_funcs

    def model(self, x):
        y = self.biases[0] + self.weights[0] * x
        for i in range(1, self.num_linear_funcs):
            prev_func = self.biases[i - 1] + self.weights[i - 1] * x
            curr_func = self.biases[i] + self.weights[i] * x
            y += (curr_func - prev_func) * sigmoid_stable(x)
        return y

    def fit(self, X, y, learning_rate=0.005, num_iterations=1000):
        num_samples = X.shape[0]
        X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()
        self.weights = np.random.randn(self.num_linear_funcs)
        self.intercepts = np.random.randn(self.num_linear_funcs)

        for _ in range(num_iterations):
            y_pred = self.model(X)
            errors = y_pred - y

            # Update weights
            for i in range(self.num_linear_funcs):
                if i == 0:
                    gradient = (2 / num_samples) * np.sum(errors * X)
                else:
                    prev_func = self.intercepts[i - 1] + self.weights[i - 1] * X
                    curr_func = self.intercepts[i] + self.weights[i] * X
                    sigmoid_input = X  # No need to compute sigmoid(X) directly
                    sigmoid_term = sigmoid_stable(sigmoid_input)
                    gradient = (2 / num_samples) * np.sum(
                        errors * sigmoid_term * (curr_func - prev_func) * X
                    )
                self.weights[i] -= learning_rate * gradient

            # Update intercepts
            for i in range(self.num_linear_funcs):
                if i == 0:
                    gradient = (2 / num_samples) * np.sum(errors)
                else:
                    prev_func = self.intercepts[i - 1] + self.weights[i - 1] * X
                    curr_func = self.intercepts[i] + self.weights[i] * X
                    sigmoid_input = X  # No need to compute sigmoid(X) directly
                    sigmoid_term = sigmoid_stable(sigmoid_input)
                    gradient = (2 / num_samples) * np.sum(
                        errors * sigmoid_term * (curr_func - prev_func)
                    )
                self.intercepts[i] -= learning_rate * gradient

    def predict(self, X):
        return self.model(X)

    def plot(self, X, y):
        y_pred = self.predict(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label="Data", color="b", marker="o", s=30)
        plt.plot(X, y_pred, label="Model", color="r")
        plt.title("Linear Sigmoid Regression")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()


# Generate some sample data
X = np.linspace(-10, 10, 100)
y = np.sin(X) + 0.5 * X**2 + np.random.randn(100) * 0.1

# Create an instance of the LinearSigmoidRegression class with 3 linear functions
model = LinearSigmoidRegression(num_linear_funcs=6)

# Train the model
model.fit(X, y, learning_rate=0.01, num_iterations=2000)

# Make predictions
y_pred = model.predict(X)

print("Model predictions:", y_pred)
print("True values:", y)
model.plot(X, y)
