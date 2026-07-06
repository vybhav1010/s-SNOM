from scipy.optimize import least_squares
import numpy as np


def forward_function(input, z_samples):
    x, y = input
    return np.exp(x) - np.pow(y, 2) + np.cosh(z_samples)




def residual(theta, z_samples, expected_output):
    preds = forward_function(theta, z_samples)

    return preds - expected_output

z_samples = np.linspace(-10, 10, 100)

x_real = 0.3
y_real = 0.5
expected_values = forward_function([x_real, y_real], z_samples)

input_initial = [0.2, 0.6]

result = least_squares(
    residual,
    input_initial,
    args=(z_samples, expected_values),
    method='lm'

)

print(result)