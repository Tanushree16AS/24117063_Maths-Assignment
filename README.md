import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import ttest_1samp

# Solving with Linear Algebra
# Creating coefficient matrix A
print('='*75)
print('-'*75)
print('TASK 1 : Solving equations by Gaussian Elimination')
print('-'*75)
A = np.array([[4, -1, -1, 0],
             [-1, 4, 0, -1],
             [-1, 0, 4, -1],
             [0, -1, -1, 4]], dtype =float)
print('Matrix (A) \n',A)
print('\n')

# Creating vector b
b = np.array([100, 100, 0, 0], dtype = float)
print('Vector (b) \n' ,b)
print('\n')

#Solving linear equation
x = np.linalg.solve(A,b)
print('Solution Vector (x) \n',x)
print('-'*75)
print('Heat flow approximation at four grid points across the steel plate:')
for i, val in enumerate(x,1):
  print(f'Temperature at x{i} = {val:.2f} °C')
print('='*75)

# Solving with Calculus
print('-'*75)
print('TASK 2 : Calculating fastest rate of Temperature change along one edge')
print('-'*75)

# Define temperature functions
x_vals = np.linspace(0, 10, 200)
T_vals = 100 * np.sin(np.pi * x_vals / 10)
dT_vals = 100 * (np.pi / 10) * np.cos(np.pi* x_vals / 10)

# Maximun magnitude of rate change
max_rc = 10*np.pi
print(f'Maximum rate change is {max_rc :.2f}°C per unit length.')

# Maximum Temperature change
max_ftc = [0,10]
print("Fastest changes occurs at x =", max_ftc)

# Plotting
plt.figure(figsize=(8,5))
plt.plot(x_vals, T_vals, label="Temperature T(x)")
plt.plot(x_vals, dT_vals, label="Rate of Change dT/dx")
plt.legend()
plt.xlabel("x (cm)")
plt.ylabel("Temperature / Rate")
plt.title("Temperature Distribution and Rate of Change along Plate Edge")
plt.grid(True)
plt.show()
print('='*75)

# Finding Probability
print('-'*75)
print('TASK 3 : Finding probability of random error')
print('-'*75)

# Parameters of Normal distribution
mu = 0
sigma = 2
x_value = 3 # The value to check error > 3°C
probability = 1 - norm.cdf(3, loc = mu, scale =sigma)

print(f"P(random error >3°C) = {probability:.4f}")
print(f"{probability * 100:.2f}% chance of error exceeding 3°C")
print('='*75)

# Hypothesis Testing - Statistics
print('-'*75)
print('TASK 4 : Hypothesis Testing - True mean = 100°C')
print('-'*75)

# Hypothesis Testing - True mean = 100°C
data = np.array([99, 100, 98, 101, 97, 99, 100, 98])
sample_mean = np.mean(data)
std_dev = np.std(data, ddof = 1)

# T-Test
t_statistic, p_value = stats.ttest_1samp(data, 100)

print(f"Sample mean = {sample_mean:.2f} °C")
print(f"Sample standard deviation = {std_dev:.2f}")
print(f"t-statistic = {t_statistic:.2f}")
print(f"p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis. The true mean is not equal to 100°C.")
else:
    print("Fail to reject the null hypothesis. The true mean is equal to 100°C.")
print('='*75)
