import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm

import scipy
from scipy.ndimage import gaussian_filter

# matplotlib.use("Agg")
np.set_printoptions(precision=2, linewidth=180, suppress=True)


x = np.load('x.npy')
y = np.load('y.npy')
z = np.load('z.npy')

z = scipy.ndimage.gaussian_filter(z, sigma=4)

print("---------")
print("---------")
print(x[0:3,0:3])
print("---------")
print(y[0:3,0:3])
print("---------")
print(z[0:3,0:3])

plt.ioff()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, rcount=100, ccount=100, edgecolor='none', cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
ax.set_xlabel('Tau')
ax.set_ylabel('Update Freq.')
ax.set_zlabel('Avg. Reward')

fig.tight_layout()

plt.show()
plt.savefig("out.png", dpi=400)
