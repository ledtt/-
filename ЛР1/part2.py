import numpy as np
import matplotlib.pyplot as plt
import timeit

def generate_coordinates(N):
    X = np.linspace(0, 2*np.pi, N)
    Y = np.cos(X) + np.sin(X)
    return X,Y

N = 8
x, y = generate_coordinates (N) 
xx, yy = generate_coordinates(1000)

# FFT 
fft_result = np.fft.fft(y)
devided_fft_result = [x / N for x in fft_result] #так как данная функция не производит деление и возвращает N*Сk

# Расшифровка результатов FFT
amplitude = np.abs(devided_fft_result) 
phase = np.angle(devided_fft_result)    

# Обратное преобразование Фурье
ifft_result = np.fft.ifft(fft_result) #передаём фактический результат прямого преобразования, 
                                        #так как в данной функции есть коэффициент 1/N

# Отрисовка графиков
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y, marker='o')
plt.plot(xx, yy)
plt.title('Initial Signal')
plt.xlabel('t')
plt.ylabel('f(t)')

plt.subplot(2, 2, 2)
plt.stem(amplitude)
plt.title('Amplitude Spectrum')
plt.xlabel('k')
plt.ylabel('|Ck|')

plt.subplot(2, 2, 3)
plt.stem(phase)
plt.title('Phase Spectrum')
plt.xlabel('k')
plt.ylabel('<Ck')

plt.subplot(2, 2, 4)
plt.plot(x, ifft_result.real, label='Reconstructed Signal')
plt.plot(xx, yy, label='Original Signal')
plt.title('Reconstructed Signal')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()

plt.tight_layout()
plt.show()
