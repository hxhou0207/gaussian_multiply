import numpy as np
import matplotlib.pyplot as plt

mu1, sigma1 = 0, 1
mu2, sigma2 = 2, 1
def gaussian_multiply(mu1, sigma1, mu2, sigma2):
    # Calculate the mean and variance of the resulting Gaussian
    sigma_result = 1 / (1/sigma1**2 + 1/sigma2**2)
    mu_result = sigma_result * (mu1/sigma1**2 + mu2/sigma2**2)
    sigma_result = np.sqrt(sigma_result)
    return mu_result, sigma_result

def plot_gaussians(mu1, sigma1, mu2, sigma2, mu_result, sigma_result):
    x = np.linspace(-10, 10, 500)
    gaussian1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gaussian2 = (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    gaussian_result = (1 / (sigma_result * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_result) / sigma_result) ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, gaussian1, label=f'N({mu1}, {sigma1**2})', color='blue')
    plt.plot(x, gaussian2, label=f'N({mu2}, {sigma2**2})', color='green')
    plt.plot(x, gaussian_result, label=f'Result N({mu_result:.2f}, {sigma_result**2:.2f})', color='red')
    plt.legend()
    plt.title('Gaussian Multiplication')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()


mu_result, sigma_result = gaussian_multiply(mu1, sigma1, mu2, sigma2)
plot_gaussians(mu1, sigma1, mu2, sigma2, mu_result, sigma_result)
