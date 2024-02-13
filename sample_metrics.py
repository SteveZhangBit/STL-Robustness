import numpy as np
import matplotlib.pyplot as plt

def laplace_filter(signal):
    # Laplace filter: computes second derivative
    laplace_filtered_signal = np.diff(np.diff(signal))
    return laplace_filtered_signal

# Generate a sample signal
x = np.linspace(0, 10, 100)
signal = np.sin(x) + 0.1 * np.random.randn(100)  # Sample signal (with noise)

# Apply Laplace filter
laplace_filtered_signal = laplace_filter(signal)

# Plot the original signal and Laplace filtered signal
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(x, signal, label='Original Signal')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x[2:], laplace_filtered_signal, label='Laplace Filtered Signal')
plt.legend()

plt.show()

def find_zero_crossings(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return zero_crossings

# Find zero crossings in the Laplace filtered signal
zero_crossings = find_zero_crossings(laplace_filtered_signal)

# Plot the zero crossings on top of the Laplace filtered signal
plt.figure(figsize=(10, 5))
plt.plot(x[2:], laplace_filtered_signal, label='Laplace Filtered Signal')
plt.scatter(x[2:][zero_crossings], laplace_filtered_signal[zero_crossings], color='red', label='Zero Crossings')
plt.legend()
plt.show()

def compute_trend(signal, degree=1):
    # Fit a polynomial to the signal to compute the trend
    coeffs = np.polyfit(x, signal, degree)
    trend = np.polyval(coeffs, x)
    return trend

# Compute a trend (e.g., using a linear fit)
trend = compute_trend(signal, degree=1)

# Plot the original signal and the computed trend
plt.figure(figsize=(10, 5))
plt.plot(x, signal, label='Original Signal')
plt.plot(x, trend, label='Trend (Linear Fit)')
plt.legend()
plt.show()