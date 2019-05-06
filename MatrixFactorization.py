import numpy as np

class MF():

    def __init__(self, X, k, alpha, beta, iterations):
        """
        Perform matrix factorization to predict np.nan entries in a matrix.

        Arguments
        - X (ndarray)   : sample-feature matrix
        - k (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.X = X
        self.num_samples, self.num_features = X.shape
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        # True if not nan
        self.not_nan_index = (np.isnan(self.X) == False)

    def train(self):
        # Initialize factorization matrix U and V
        self.U = np.random.normal(scale=1./self.k, size=(self.num_samples, self.k))
        self.V = np.random.normal(scale=1./self.k, size=(self.num_features, self.k))

        # Initialize the biases
        self.b_u = np.zeros(self.num_samples)
        self.b_i = np.zeros(self.num_features)
        self.b = np.mean(self.X[np.where(self.not_nan_index)])
        # Create a list of training samples
        self.samples = [
            (i, j, self.X[i, j])
            for i in range(self.num_samples)
            for j in range(self.num_features)
            if not np.isnan(self.X[i, j])
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        predicted = self.full_matrix()
        error = 0
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index[i, j]:
                    error += pow(self.X[i, j] - predicted[i, j], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, x in self.samples:
            # Computer prediction and error
            prediction = self.get_x(i, j)
            e = (x - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (2 * e - self.beta * self.b_i[j])

            # Update factorization matrix U and V
            """
            If RuntimeWarning: overflow encountered in multiply,
            then turn down the learning rate alpha.
            """
            self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i,:])
            self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j,:])
    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + self.U.dot(self.V.T)

    def replace_nan(self, X_hat):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X

if __name__ == '__main__':
    X = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ], dtype=np.float)
    # replace 0 with np.nan
    X[X == 0] = np.nan
    print(X)
    # np.random.seed(1)
    mf = MF(X, k=2, alpha=0.1, beta=0.1, iterations=100)
    mf.train()
    X_hat = mf.full_matrix()
    X_full = mf.replace_nan(X_hat)

    print(X_hat)
    print(X_full)
    print(X)
