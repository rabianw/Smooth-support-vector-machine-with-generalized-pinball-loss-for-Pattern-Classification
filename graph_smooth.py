import numpy as np
import matplotlib.pyplot as plt

def GPloss(margin, mu):
    u = 1 - margin
    p = np.piecewise(u, [
        u >= (0.01/1) + 1*mu,
        ((0.01/1) <= u) & (u <= (0.01/1) + 1*mu),
        (-(0.1/0.5) <= u) & (u <= (0.01/1)),
        (-(0.1/0.5) - 0.5*mu <= u) & (u <= -(0.1/0.5))
    ], [
        lambda u: 1*(u - (0.01/1))- (((1)**2)*mu)/2 ,
        lambda u: (u - (0.01/1))**2/(2*mu),
        0,
        lambda u: ((u + (0.1/0.5))**2)/(2 * mu),
        lambda u: -0.5*(u + (0.1/0.5))-(((0.5)**2)*mu)/2
    ])
    return p

# Function to compute numerical derivative
def numerical_derivative(x, y):
    dx = x[1] - x[0]
    dy_dx = np.gradient(y, dx)
    return dy_dx

# Example usage
mu_value = 1
margin_values = np.linspace(-2, 2, 100)
loss_values = GPloss(margin_values, mu_value)

# Compute numerical derivative
derivative_values = numerical_derivative(margin_values, loss_values)

# Plot the derivative
plt.plot(1-margin_values, derivative_values, label='Derivative of Custom Loss')
plt.scatter([(0.01/1) + 1*mu_value, (0.01/1), -(0.1/0.5), -(0.1/0.5) - 0.5*mu_value], 
            [0, 0, 0, 0], color='red', marker='o', label='Transition Points')
plt.title('Derivative of Custom Loss Function')
plt.xlabel('Margin')
plt.ylabel('Derivative')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def approximate_generalized_pinbal(margin, mu):
    u = 1 - margin
    p = np.piecewise(u, [
        u >= (0.01/1) + 1*mu,
        ((0.01/1) <= u) & (u <= (0.01/1) + 1*mu),
        (-(0.1/0.5) <= u) & (u <= (0.01/1)),
        (-(0.1/0.5) - 0.5*mu <= u) & (u <= -(0.1/0.5))
    ], [
        lambda u: 1*(u - (0.01/1))- (((1)**2)*mu)/2 ,
        lambda u: (u - (0.01/1))**2/(2*mu),
        0,
        lambda u: ((u + (0.1/0.5))**2)/(2 * mu),
        lambda u: -0.5*(u + (0.1/0.5))-(((0.5)**2)*mu)/2
    ])
    return p

# Example usage
mu_values = [0,0.01, 0.1, 0.5,0.9]  # You can add more values as needed
margin_values = np.linspace(0, 2, 100)

# Plot the custom loss function for different mu values
plt.figure(figsize=(12, 8))
for mu in mu_values:
    loss_values = GPloss(margin_values, mu)
    plt.plot(margin_values, loss_values, label=f'Mu={mu}')

plt.title('Custom Loss Function for Different Mu Values')
plt.xlabel('Margin')
plt.ylabel('Custom Loss')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()
