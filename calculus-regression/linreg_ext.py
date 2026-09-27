import numpy as np
import random
import matplotlib.pyplot as plt


def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted probabilities
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class QuadraticRegression:
    def __init__(self, learning_rate=0.0001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X, y, binary_class=False, threshold=0.00001):

        X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()

        # Initialize weights with random values
        self.weights = np.random.uniform(
            -0.5, 0.5, size=(1, 3)
        )  # bias, weight1, weight2
        print(self.weights.shape)
        self.error = float("inf")

        for _ in range(self.n_iterations):
            print("Iteration:", _, "Error:", self.error)
            # Perform prediction
            y_pred = self.predict(X, binary_class=binary_class)
            # Compute error
            error = (
                binary_cross_entropy_loss(y, y_pred)
                if binary_class
                else mean_squared_error(y, y_pred)
            )

            if abs(error - self.error) < threshold:
                print(f"Converged at iteration {_} with RMSE: {error}")
                break

            self.error = error

            # calc ssr/w_i | the partial derivatives
            for i in range(self.weights.shape[1]):
                gradient = -2 * np.mean((y - y_pred) * X[:, 0])
                self.weights[0, i] -= self.learning_rate * gradient

    def predict(self, X, binary_class=False):
        # self.weights (bias, weight)
        prediction = (
            (self.weights[0, 2] * X**2) + (self.weights[0, 1] * X) + self.weights[0, 0]
        )
        if binary_class:
            prediction = 1 / (1 + np.exp(-prediction))
            prediction = np.where(prediction >= 0.5, 1, 0)
        return prediction.ravel()  # Return as 1D array

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


# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[i] for i in range(1, 101)])
    y = np.array([[i * 3] for i in range(1, 101)])
    print(X.shape, y.shape)

    # Initialize and train the model
    model = QuadraticRegression()
    model.fit(X, y, binary_class=False)  # For regression

    # Make predictions
    y_pred = model.predict(X, binary_class=False)

    # Print the predicted values
    print("Predicted values:")
    print(y_pred)
    print("Used weights:")
    print(model.weights)

    y = np.array([[1] if i % 2 == 0 else [0] for i in range(1, 101)])
    # use model for classification now with new y to predict evens or odds
    class_model = QuadraticRegression()
    class_model.fit(X, y, binary_class=True)

    # Make predictions
    y_pred = class_model.predict(X, binary_class=True)

    # Print the predicted values
    print("Predicted values:")
    print(y_pred)
    print("Used weights:")
    print(class_model.weights)
