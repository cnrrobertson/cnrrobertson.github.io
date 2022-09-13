import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from autograd import grad
import autograd.numpy as anp

# %%
# Check model function and gradients
# x,p1,p2,p3,p4 = sp.symbols("x p1 p2 p3 p4")
x,p1,p2 = sp.symbols("x p1 p2")
f = sp.exp(-p1*x)*(sp.sin((p1*x)**3) + sp.sin((p1*x)**2) - p1*x) + (p2*(x-1))**16

params = {p1:1,p2:1/4}
# params = {p1:-1,p2:1,p3:1,p4:1/4}
fstar = f.subs(params)

# Plot function
sp.plot(fstar,xlim=(-1,5),ylim=(-1,2))

# Plot marginalized functions
params1 = {x:2,p2:1/4}
sp.plot(f.subs(params1),xlim=(-5,5),ylim=(-1,1))
params2 = {p1:1,x:2}
sp.plot(f.subs(params2),xlim=(-5,5),ylim=(-1,1))

# fgrads = [sp.diff(f,p1),sp.diff(f,p2),sp.diff(f,p3),sp.diff(f,p4)]

# %%
# Sample data
ffstar = sp.lambdify(x,fstar)
xdom = np.linspace(-1/2,5,1000)
ydom = ffstar(xdom)
xsamples = np.random.uniform(-1/2,5,100)
ysamples = ffstar(xsamples)

plt.plot(xdom,ydom)
plt.scatter(xsamples,ysamples)
plt.show()

# %%
# Need to put into loss function form then find energy surface that will be minimized for each point
#    - Maybe sample points so that it can be evaluated there?

def gradient_descent(fp,x0,lr=.2,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    for s in range(steps):
        xnew = x - lr*fp(x)
        if np.linalg.norm(xnew - x) < tol:
            print("Converged in {} steps to tolerance {}.".format(s,tol))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge to tolerance {} after {} steps.".format(tol,steps))
    return x,xs

# With given gradient
t1 = lambda x: (x-1.2)**2
tp1 = lambda x: 2*(x-1.2)
gradient_descent(tp1,0)[0]

# With autograd
tg1 = grad(t1)
gradient_descent(tg1,0.0)[0]

# %%
# Plotting descent steps

def animate_steps(xs,func,xmin=-1,xmax=3,interval=50):
    def anim_func(i):
        ax.clear()

        ax.plot(fxs,fys)
        ax.scatter(xs[i],func(xs[i]))
    fig = plt.figure()
    ax = plt.axes()
    fxs = np.linspace(xmin,xmax,1000)
    fys = func(fxs)
    tanim = anim.FuncAnimation(fig,anim_func,interval=50,frames=len(xs))
    plt.show()

xs = gradient_descent(tp1,-2,.1)[1]
animate_steps(xs,t1)
# %%
# Implement forward mode automatic diff
class Dual:
    def __init__(self, real, diff):
        self.real = real
        self.diff = diff
    def _add(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.diff + other.diff)
        else:
            return Dual(self.real + other, self.diff)
    def _sub(self, other):
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.diff - other.diff)
        else:
            return Dual(self.real - other, self.diff)
    def __add__(self, other):
        return self._add(other)
    def __radd__(self, other):
        return self._add(other)
    def __sub__(self, other):
        return self._sub(other)
    def __rsub__(self, other):
        return self._sub(other)
    def __str__(self):
        return "Dual({},{})".format(self.real,self.diff)
    def __repr__(self):
        return "Dual({},{})".format(self.real,self.diff)
# %%
# Set up problem
f_known = lambda x: anp.exp(-x)*(anp.sin((x)**3) + anp.sin((x)**2) - x) + 1/(1 + anp.exp(-1*(x-1)))
xsamples = np.random.uniform(-1/2,5,100)
ysamples = f_known(xsamples)
plt.scatter(xsamples,ysamples)
plt.show()

f_model = lambda p: anp.exp(-p[0]*xsamples)*(anp.sin((p[0]*xsamples)**3) + anp.sin((p[0]*xsamples)**2) - p[0]*xsamples) + 1/(1 + anp.exp(-p[1]*(xsamples-1)))
loss = lambda p: anp.sum((f_model(p) - ysamples)**2)
grad_loss = grad(loss)

ps = np.linspace(-.1,2.5,1000)
plt.plot(ps,[loss(np.array([p_i,1])) for p_i in ps])
plt.plot(ps,[loss(np.array([1,p_i])) for p_i in ps])
plt.show()

p0 = np.array([.1,.95])
xs = gradient_descent(grad_loss, p0,.005)[1]

# %%
def animate_steps_2d(xs,func,xmin=-.1,xmax=2.5,ymin=-1,ymax=3,interval=50,path=True):
    def anim_func(i):
        ax.clear()
        ax1.clear()
        ax2.clear()
        # Add surface plot
        ax.plot_surface(X,Y,Z,cmap="gist_earth")
        x_loss = loss(xs[i])
        ax.scatter(xs[i][0],xs[i][1],x_loss,zorder=100,color="red",s=100)
        ax.set_xlabel("$p_1$")
        ax.set_ylabel("$p_2$")
        ax.set_zlabel("loss")
        if path:
            temp_x1 = [xs[j][0] for j in range(i)]
            temp_x2 = [xs[j][1] for j in range(i)]
            temp_losses = [loss(xs[j]) for j in range(i)]
            ax.plot(temp_x1,temp_x2,temp_losses,color="red")
        loss_fx = [func([fxs[j],xs[i][1]]) for j in range(len(fxs))]
        loss_fy = [func([xs[i][0],fys[j]]) for j in range(len(fys))]
        # Add flat plots for perspective
        ax1.plot(fxs,loss_fx)
        ax1.scatter(xs[i][0],x_loss,color="red",s=100,zorder=100)
        ax1.set_xlabel("$p_1$")
        ax1.set_ylabel("loss")
        ax1.set_xlim(np.min(X),np.max(X))
        ax1.set_ylim(np.min(Z),np.max(Z))
        ax2.plot(fys,loss_fy)
        ax2.scatter(xs[i][1],x_loss,color="red",s=100,zorder=100)
        ax2.set_xlabel("$p_2$")
        ax2.set_ylabel("loss")
        ax2.set_xlim(np.min(Y),np.max(Y))
        ax2.set_ylim(np.min(Z),np.max(Z))

    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(20,30)
    ax = fig.add_subplot(gs[1:19,0:18],projection="3d",computed_zorder=False)
    ax1 = fig.add_subplot(gs[0:8,20:30])
    ax2 = fig.add_subplot(gs[12:20,20:30])
    ax.view_init(47,47)

    fxs = np.linspace(xmin,xmax,100)
    fys = np.linspace(ymin,ymax,100)
    X,Y = np.meshgrid(fxs,fys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = func([X[i,j],Y[i,j]])

    tanim = anim.FuncAnimation(fig,anim_func,interval=50,frames=len(xs))
    plt.show()

xs = gradient_descent(grad_loss, p0,.007)[1]
animate_steps_2d(xs,loss)
# %%
# gradient descent with momentum
def gradient_descent_momentum(fp,x0,d,lr=.5,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # -------- CHANGE ----------
    sum_grad = 0
    for s in range(1,steps):
        sum_grad = fp(x) + d*sum_grad
        step = -lr*sum_grad
    # --------------------------
        xnew = x + step
        if np.linalg.norm(xnew - x) < tol:
            print("Converged in {} steps to tolerance {}.".format(s,tol))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge to tolerance {} after {} steps.".format(tol,steps))
    return x,xs

xs = gradient_descent_momentum(tp1,-2,.5)[1]
animate_steps(xs,t1,interval=200)

# %%
# Adaptive gradient (Adagrad)
def gradient_descent_adagrad(fp,x0,lr=.5,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # -------- CHANGE ----------
    sum_sq_grad = 0
    for s in range(1,steps):
        sum_sq_grad = fp(x)**2 + sum_sq_grad
        step = -lr*fp(x)/np.sqrt(sum_sq_grad)
    # --------------------------
        xnew = x + step
        if np.linalg.norm(xnew - x) < tol:
            print("Converged in {} steps to tolerance {}.".format(s,tol))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge to tolerance {} after {} steps.".format(tol,steps))
    return x,xs

xs = gradient_descent_adagrad(tp1,-2,.5)[1]
animate_steps(xs,t1,interval=200)

# %%
# Using root mean square propogation (RMSProp)

def gradient_descent_rmsprop(fp,x0,d,lr=.5,tol=1e-12,steps=1000):
    x = x0
    xs = [x]

    # -------- CHANGE ----------
    sum_sq_grad = 0
    for s in range(1,steps):
        sum_sq_grad = (1-d)*(fp(x)**2) + d*sum_sq_grad
        step = -lr*fp(x)/np.sqrt(sum_sq_grad)
    # --------------------------
        xnew = x + step
        if np.linalg.norm(xnew - x) < tol:
            print("Converged in {} steps to tolerance {}.".format(s,tol))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge to tolerance {} after {} steps.".format(tol,steps))
    return x,xs

xs = gradient_descent_rmsprop(tp1,-2,.5,.4)[1]
animate_steps(xs,t1,interval=200)

# %%
# Using adam

def gradient_descent_adam(fp,x0,beta1,beta2,lr=.5,tol=1e-12,steps=1000):
    x = x0
    xs = [x]

    # -------- CHANGE ----------
    sum_grad = 0
    sum_sq_grad = 0
    for s in range(1,steps):
        sum_grad = beta1*sum_grad + (1-beta1)*fp(x)
        sum_sq_grad = beta2*sum_sq_grad + (1-beta2)*(fp(x)**2)
        step = -lr*sum_grad/np.sqrt(sum_sq_grad)
    # --------------------------
        xnew = x + step
        if np.linalg.norm(xnew - x) < tol:
            print("Converged in {} steps to tolerance {}.".format(s,tol))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge to tolerance {} after {} steps.".format(tol,steps))
    return x,xs

xs = gradient_descent_adam(tp1,-2,.9,.99)[1]
animate_steps(xs,t1,interval=200)
