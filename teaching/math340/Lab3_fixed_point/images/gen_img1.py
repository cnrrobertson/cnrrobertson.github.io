import numpy as np
import matplotlib.pyplot as plt

dom = np.linspace(.2, 1.2, 100)
dom2 = np.linspace(-5, 5, 1000)
f = np.cos(dom)

xs = np.array([.5, .8775, .6390, .8026])
ys = np.array([.8775, .6390, .8026, .6947])
plt.plot(dom, f)
dashed = plt.plot(dom2, dom2, color="black", ls="--", lw=1.2)
dashed[0].set_dashes([4, 4])
plt.scatter(xs, ys, color="red", zorder=100)

plt.scatter([.7390], [.7390], color="black", zorder=100)

for i in range(len(xs)):
    if i != len(xs)-1:
        arrow1 = plt.arrow(xs[i], ys[i], ys[i]-xs[i], 0, head_width=.02, lw=.2, color="black", length_includes_head=True)
        arrow2 = plt.arrow(ys[i], ys[i], 0, ys[i+1] - ys[i], head_width=.02, lw=.2, color="black", length_includes_head=True)

plt.text(.4, .3, "h(x) = x", fontsize=14)
plt.text(1.1, .5, "g(x) = cos(x)", fontsize=14)

plt.text(xs[0]-.12, ys[0]-.04, "$g(x_1)$", fontsize=10)
plt.text(xs[1]-.12, ys[1]-.04, "$g(x_2)$", fontsize=10)
plt.text(xs[2]-.12, ys[2]-.04, "$g(x_3)$", fontsize=10)
plt.text(xs[3]-.12, ys[3]-.04, "$g(x_4)$", fontsize=10)
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.ylim(.2, 1.2)
plt.xlim(0, 1.6)
plt.hlines(0, -1, 5, color="black")
plt.vlines(0,-5, 5, color="black")
plt.tight_layout()
plt.savefig("img1.png")
plt.show()
