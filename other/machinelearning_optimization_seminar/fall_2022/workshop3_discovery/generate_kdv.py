import numpy as np
import pde # conda install py-pde
import matplotlib.pyplot as plt

# Domain
xmax = 1.0
nx = 1000
tmax = 1.0
dt = 1e-8
save_dt = 0.01
init_cond = ".1*exp(-(1/.01)*(x-0.3)**2)"

# Initialize objects
grid = pde.CartesianGrid([(0.0,xmax)],nx,periodic=True)
h = pde.ScalarField.from_expression(grid,init_cond,label="h(x,t)")
eq = pde.PDE({"h": "-d_dx(h) + 6*d_dx(h**2)"})
storage = pde.MemoryStorage()

# Run
result = eq.solve(h,t_range=tmax,dt=dt,tracker=storage.tracker(save_dt))

# Visualize
pde.plot_kymograph(storage)
movie = pde.visualization.movie(storage,"videos/kdv.mp4")