import numpy as np

# Create a complex input signal
x = np.array([0 +7j, 1 + 6j, 2 + 5j, 3 + 4j, 4 + 3j, 5 + 2j, 6 + 1j, 7 + 0j])

# Compute the FFT
X = np.fft.fft(x)

print(X)