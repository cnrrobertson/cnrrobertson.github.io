import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(.5, 4, 1000)
f = (x - 3)**3 * np.sin(x) + 3.5

plt.plot(x, f)
plt.scatter([x[0], x[-1]], [f[0], f[-1]], color="red", zorder=100)
plt.scatter(1.45, 0, color="orange", zorder=100)
plt.text(x[0]-.3, f[0]+.3, "f(a)", fontsize=14)
plt.text(x[-1]+.2, f[-1]+.2, "f(b)", fontsize=14)
plt.text(1.7, .2, "root", fontsize=14)

plt.ylim(-5, 5)
plt.xlim(-1, 5)
plt.hlines(0, -1, 5, color="black")
plt.vlines(0,-5, 5, color="black")
plt.savefig("img1.png")
plt.show()
