import matplotlib.pyplot as plt
import numpy as np
import math
import time


def generate_coordinates(N):

    X = np.linspace(0, 2*np.pi, N)
    Y = np.cos(X) + np.sin(X)

    return X,Y

def fft(a, direction):

    N = len(a)

    if N == 1: return a

    even = fft(a[0::2], direction)
    odd = fft(a[1::2], direction)

    Wn = complex(math.cos(2 * math.pi / N), direction * math.sin(2 * math.pi / N))
    W = complex(1) 

    y = [0] * N
    for i in range (N // 2):
        y[i] = even[i] + W * odd[i]
        y[i + N // 2] = even[i] - W * odd[i]
        W*= Wn

    return y
  

N = 8
x, y = generate_coordinates (N) 
xx, yy = generate_coordinates(1000)

start_fft = time.time()
fft_y = fft(y,-1)
end_fft = time.time()-start_fft

result_fft = [x / N for x in fft_y]

amplitude = np.abs(result_fft)
phase = np.angle(result_fft)

start_ifft = time.time()
ifft_y = fft (result_fft, 1)
end_ifft = time.time()-start_ifft

print("FFT:", end_fft, "s")
print("IFFT:", end_ifft, "s")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y, marker='o')
plt.plot(xx, yy)
plt.title('Initial Signal y=cos(x)+sin(x)')
plt.xlabel('t')
plt.ylabel('f(t)')

plt.subplot(2, 2, 2)
plt.stem(amplitude)
plt.title('Fast Fourier Tranform - Amplitude spectrum')
plt.xlabel('k')
plt.ylabel('|Ck|')

plt.subplot(2, 2, 3)
plt.stem(phase)
plt.title('Fast Fourier Tranform - Phase Spectrum')
plt.xlabel('k')
plt.ylabel('<Ck')

plt.subplot(2, 2, 4)
plt.plot(x, np.real(ifft_y), label='Reconstructed Signal')
plt.plot(xx, yy, label='Original Signal')
plt.title('Inversed Fast Fourier Transform')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()

plt.tight_layout()
plt.show()

