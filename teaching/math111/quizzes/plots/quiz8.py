import numpy as np
import matplotlib.pyplot as plt

# %%
# Problem 1d
x = np.linspace(-1,2,100)
crit_points = np.array([0,1])
inf_points = np.array([0,2/3])
y = lambda x: 3*x**4 - 4*x**3
xlims = np.min(x),np.max(x)
ylims = np.min(y(x)),np.max(y(x))
plt.hlines(0,xlims[0],xlims[1],color="black")
plt.vlines(0,ylims[0],ylims[1],color="black")
plt.plot(x,y(x),label="$y=3x^4-4x^3$")
plt.scatter(crit_points,y(crit_points),label="Critical points",zorder=100)
plt.scatter(inf_points,y(inf_points),label="Inflection points",zorder=100)
plt.xlabel("$x$")
plt.legend()
plt.savefig("quiz8_1d.png")
plt.show()

# %%
# Problem 2d
x = np.linspace(-2,4,100)
crit_points = np.array([0,2])
inf_points = np.array([1])
y = lambda x: 2 + 3*x**2 - x**3
xlims = np.min(x),np.max(x)
ylims = np.min(y(x)),np.max(y(x))
plt.hlines(0,xlims[0],xlims[1],color="black")
plt.vlines(0,ylims[0],ylims[1],color="black")
plt.plot(x,y(x),label="$y=2+3x^2 - x^3$")
plt.scatter(crit_points,y(crit_points),label="Critical points",zorder=100)
plt.scatter(inf_points,y(inf_points),label="Inflection points",zorder=100)
plt.xlabel("$x$")
plt.legend()
plt.savefig("quiz8_2d.png")
plt.show()
