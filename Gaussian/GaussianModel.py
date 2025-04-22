import json

import numpy as np
import matplotlib.pyplot as plt

with open('review_data.json', 'r') as file:
    data = json.load(file)
data_array = np.array([list(row.values()) for row in data.values()])
print(data_array.shape)

# Helps determine the responsibilities for each datapoint
def multivariate_gaussian(x, mean, cov):
    d = x.shape[1]
    cov_inv = np.linalg.inv(cov)
    diff = x - mean
    exponent = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
    return np.exp(-0.5 * exponent) / np.sqrt((2 * np.pi)**d * np.linalg.det(cov))

# Sets random base parameters to start the GMM
def initialize_parameters(X, n_components):
    n_samples, d = X.shape
    np.random.seed(0)
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.cov(X.T) for _ in range(n_components)])
    weights = np.full(n_components, 1 / n_components)
    return means, covariances, weights

# Runs the expectation step which assigns each cluster a responsibility probability for each datapoint
def expectation(X, means, covariances, weights):
    n_samples = X.shape[0]
    responsibilities = np.zeros((n_samples, len(means)))
    for k in range(len(means)):
        responsibilities[:, k] = weights[k] * multivariate_gaussian(X, means[k], covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

# Runs the maximization step which reassigns the cluster's weights
def maximization(X, responsibilities):
    n_samples, d = X.shape
    n_components = responsibilities.shape[1]
    Nk = responsibilities.sum(axis=0)
    means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
    covariances = np.zeros((n_components, d, d))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / Nk[k]
    weights = Nk / n_samples
    return means, covariances, weights

# Determines if the cluster's centroids moved or stayed the same
def compute_log_likelihood(X, means, covariances, weights):
    n_samples = X.shape[0]
    log_likelihood = 0
    for i in range(n_samples):
        likelihood = 0
        for k in range(len(means)):
            likelihood += weights[k] * multivariate_gaussian(X[i:i+1], means[k], covariances[k])
        log_likelihood += np.log(likelihood + 1e-10)  # add small value to avoid log(0)
    return log_likelihood

# RUns the GMM by running the expectation step, the maximization step and then checks if the centroids moved to determine if it should iterate again
def fit_gmm(X, n_components, n_iter=100, tol=1e-4):
    means, covariances, weights = initialize_parameters(X, n_components)
    log_likelihood_prev = None
    for i in range(n_iter):
        responsibilities = expectation(X, means, covariances, weights)
        means, covariances, weights = maximization(X, responsibilities)
        log_likelihood = compute_log_likelihood(X, means, covariances, weights)
        if log_likelihood_prev is not None and abs(log_likelihood - log_likelihood_prev) < tol:
            break
        log_likelihood_prev = log_likelihood
    return means, covariances, weights, responsibilities

# Fit the manual GMM
means_est, covs_est, weights_est, resp_est = fit_gmm(data_array, 6)
y_pred = np.argmax(resp_est, axis=1)

# Stores the cluster assignments
results = [
    {"id": key, "label": int(pred)}
    for key, pred in zip(data.keys(), y_pred)
]
with open("gmm_predictions.json", "w") as file:
  json.dump(results, file, indent=4)

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=y_pred, cmap='viridis', s=10)
ax.scatter(means_est[:, 0], means_est[:, 1], means_est[:, 2], c='red', s=100, marker='x', label='GMM Centers')
ax.set_title("3D GMM with Manual EM Implementation")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.legend()
plt.tight_layout()
plt.show()
