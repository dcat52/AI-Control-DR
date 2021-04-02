from src.util import OUActionNoise
import numpy as np
import matplotlib.pyplot as plt

std = 0.1
theta = 0.05

ou = OUActionNoise(mean=np.zeros(1), std_deviation=float(std) * np.ones(1), theta=float(theta) * np.ones(1))


ou_noise_sample = []
for i in range(10000):
    ou_noise_sample.append(ou())

plt.figure(dpi=200, figsize=(12, 2))
plt.grid(True)
plt.plot(ou_noise_sample, linewidth=0.5)
plt.show()
