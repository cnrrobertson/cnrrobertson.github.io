import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from autograd import grad
import autograd.numpy as anp

# %%
def gradient_descent(f_p,x0,alpha=.2,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    for s in range(steps):
        v_i = -alpha*f_p(x)
        xnew = x + v_i
        if np.linalg.norm(f_p(xnew)) < tol:
            print("Converged to objective loss gradient below {} in {} steps.".format(tol,s))
            return x,xs
        elif np.linalg.norm(xnew - x) < tol:
            print("Converged to steady state of tolerance {} in {} steps.".format(tol,s))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge after {} steps (tolerance {}).".format(steps,tol))
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

# %%
def animate_steps_2d(xs,loss,savefile=None,xmin=-.1,xmax=2.5,ymin=-1,ymax=3,interval=50):
    fig = plt.figure(figsize=(8,5),constrained_layout=True)
    gs = gridspec.GridSpec(ncols=6,nrows=2,figure=fig)
    ax = fig.add_subplot(gs[:,0:4],projection="3d")#,computed_zorder=False)
    ax1 = fig.add_subplot(gs[0,4:])
    ax2 = fig.add_subplot(gs[1,4:])
    ax.view_init(47,47)
    ax.set_xlabel("$p_0$"); ax.set_ylabel("$p_2$"); ax.set_zlabel("loss",rotation=90)
    ax1.set_xlabel("$p_0$"); ax1.set_ylabel("loss")
    ax2.set_xlabel("$p_1$"); ax2.set_ylabel("loss")

    xs_arr = np.array(xs)
    fxs = np.linspace(xmin,xmax,100)
    fys = np.linspace(ymin,ymax,100)
    loss_fx = [loss([fxs[j],xs[0][1]]) for j in range(len(fxs))]
    loss_fy = [loss([xs[0][0],fys[j]]) for j in range(len(fys))]
    X,Y = np.meshgrid(fxs,fys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = loss([X[i,j],Y[i,j]])
    # Add surface plot
    surf = ax.plot_surface(X,Y,Z,cmap="gist_earth")
    ax1.set_xlim(np.min(X),np.max(X)); ax1.set_ylim(np.min(Z),np.max(Z))
    ax2.set_xlim(np.min(Y),np.max(Y)); ax2.set_ylim(np.min(Z),np.max(Z))
    plot1 = ax.plot(xs[0][0],xs[0][1],loss(xs[0]),zorder=100,color="red",linestyle="",marker="o")[0]
    plot2 = ax.plot([],[],[],color="orange")[0]
    # Add flat plots for perspective
    plot3 = ax1.plot(fxs,loss_fx)[0]
    plot4 = ax1.scatter(xs[0][0],loss(xs[0]),color="red",s=100,zorder=100)
    plot5 = ax2.plot(fys,loss_fy)[0]
    plot6 = ax2.scatter(xs[0][1],loss(xs[0]),color="red",s=100,zorder=100)
    def anim_func(i):
        x_loss = loss(xs[i])
        plot1.set_data_3d(xs[i][0],xs[i][1],x_loss)
        temp_x1 = [xs[j][0] for j in range(i)]
        temp_x2 = [xs[j][1] for j in range(i)]
        temp_losses = [loss(xs[j]) for j in range(i)]
        plot2.set_data_3d(temp_x1,temp_x2,temp_losses)
        loss_fx = [loss([fxs[j],xs[i][1]]) for j in range(len(fxs))]
        loss_fy = [loss([xs[i][0],fys[j]]) for j in range(len(fys))]
        plot3.set_data(fxs,loss_fx)
        plot4.set_offsets([xs[i][0],x_loss])
        plot5.set_data(fys,loss_fy)
        plot6.set_offsets([xs[i][1],x_loss])
        plots = [plot1,plot2,plot3,plot4,plot5,plot6]
        return plots

    tanim = anim.FuncAnimation(fig,anim_func,interval=50,frames=len(xs),blit=True)
    if savefile == None:
        plt.show()
    else:
        tanim.save(savefile)
# %%
p0 = np.array([2.1,.2])
xs = gradient_descent(grad_loss, p0,.008,tol=1e-8,steps=100)[1]
# animate_steps_2d(xs,loss,savefile="videos/gradient_descent1.mp4")
# animate_steps_2d(xs,loss,savefile="videos/gradient_descent1.gif")
animate_steps_2d(xs,loss)
# %%
# gradient descent with momentum
def gradient_descent_momentum(f_p,x0,gamma,alpha=0.01,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # --------- NEW -----------
    sum_grad = 0
    for s in range(steps):
        sum_grad = f_p(x) + gamma*sum_grad
        v_i = -alpha*sum_grad
    # -------------------------
        xnew = x + v_i
        if np.linalg.norm(f_p(xnew)) < tol:
            print("Converged to objective loss gradient below {} in {} steps.".format(tol,s))
            return x,xs
        elif np.linalg.norm(xnew - x) < tol:
            print("Converged to steady state of tolerance {} in {} steps.".format(tol,s))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge after {} steps (tolerance {}).".format(steps,tol))
    return x,xs

p0 = np.array([2.1,.2])
xs = gradient_descent_momentum(grad_loss, p0,.9,.005,tol=1e-8,steps=100)[1]
animate_steps_2d(xs,loss,savefile="videos/momentum.mp4")
animate_steps_2d(xs,loss,savefile="videos/momentum.gif")
animate_steps_2d(xs,loss)

# %%
# gradient descent with nesterov
def gradient_descent_nesterov(f_p,x0,gamma,alpha=0.01,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # --------- NEW -----------
    sum_grad = 0
    v_i = 0
    for s in range(steps):
        sum_grad = f_p(x-gamma*v_i) + gamma*sum_grad
        v_i = -alpha*sum_grad
    # -------------------------
        xnew = x + v_i
        if np.linalg.norm(f_p(xnew)) < tol:
            print("Converged to objective loss gradient below {} in {} steps.".format(tol,s))
            return x,xs
        elif np.linalg.norm(xnew - x) < tol:
            print("Converged to steady state of tolerance {} in {} steps.".format(tol,s))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge after {} steps (tolerance {}).".format(steps,tol))
    return x,xs

p0 = np.array([2.1,.2])
xs = gradient_descent_nesterov(grad_loss, p0,.8,.002,tol=1e-8,steps=100)[1]
animate_steps_2d(xs,loss,savefile="videos/nesterov.mp4")
animate_steps_2d(xs,loss,savefile="videos/nesterov.gif")
animate_steps_2d(xs,loss)

# %%
# Adaptive gradient (Adagrad)
def adagrad(f_p,x0,alpha=.2,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # --------- NEW -----------
    sum_sq_grad = 0
    for s in range(steps):
        sum_sq_grad = f_p(x)**2 + sum_sq_grad
        v_i = -alpha*f_p(x)/np.sqrt(sum_sq_grad)
    # -------------------------
        xnew = x + v_i
        if np.linalg.norm(f_p(xnew)) < tol:
            print("Converged to objective loss gradient below {} in {} steps.".format(tol,s))
            return x,xs
        elif np.linalg.norm(xnew - x) < tol:
            print("Converged to steady state of tolerance {} in {} steps.".format(tol,s))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge after {} steps (tolerance {}).".format(steps,tol))
    return x,xs

p0 = np.array([2.1,.2])
xs = adagrad(grad_loss, p0,.2,tol=1e-8,steps=100)[1]
animate_steps_2d(xs,loss,savefile="videos/adagrad.mp4")
animate_steps_2d(xs,loss,savefile="videos/adagrad.gif")
animate_steps_2d(xs,loss)

# %%
# Using root mean square propogation (RMSProp)
def rmsprop(f_p,x0,gamma=0.9,alpha=0.001,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # --------- NEW -----------
    sum_sq_grad = 0
    for s in range(steps):
        sum_sq_grad = (1-gamma)*(f_p(x)**2) + gamma*sum_sq_grad
        v_i = -alpha*f_p(x)/np.sqrt(sum_sq_grad)
    # -------------------------
        xnew = x + v_i
        if np.linalg.norm(f_p(xnew)) < tol:
            print("Converged to objective loss gradient below {} in {} steps.".format(tol,s))
            return x,xs
        elif np.linalg.norm(xnew - x) < tol:
            print("Converged to steady state of tolerance {} in {} steps.".format(tol,s))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge after {} steps (tolerance {}).".format(steps,tol))
    return x,xs

p0 = np.array([2.1,.2])
xs = rmsprop(grad_loss, p0,0.2,.05,tol=1e-8,steps=100)[1]
animate_steps_2d(xs,loss,savefile="videos/rmsprop.mp4")
animate_steps_2d(xs,loss,savefile="videos/rmsprop.gif")
animate_steps_2d(xs,loss)
# %%
# Using adam
def adam(f_p,x0,beta1,beta2,alpha=0.01,tol=1e-12,steps=1000):
    x = x0
    xs = [x]
    # --------- NEW -----------
    sum_grad = 0
    sum_sq_grad = 0
    for s in range(1,steps):
        sum_grad = beta1*sum_grad + (1-beta1)*f_p(x)
        sum_sq_grad = beta2*sum_sq_grad + (1-beta2)*(f_p(x)**2)
        v_i = -alpha*sum_grad/np.sqrt(sum_sq_grad)
    # -------------------------
        xnew = x + v_i
        if np.linalg.norm(f_p(xnew)) < tol:
            print("Converged to objective loss gradient below {} in {} steps.".format(tol,s))
            return x,xs
        elif np.linalg.norm(xnew - x) < tol:
            print("Converged to steady state of tolerance {} in {} steps.".format(tol,s))
            return x,xs
        x = xnew
        xs.append(x)
    print("Did not converge after {} steps (tolerance {}).".format(steps,tol))
    return x,xs

p0 = np.array([2.1,.2])
xs = adam(grad_loss, p0,0.9,0.99,.05,tol=1e-8,steps=100)[1]
animate_steps_2d(xs,loss,savefile="videos/adam.mp4")
animate_steps_2d(xs,loss,savefile="videos/adam.gif")
animate_steps_2d(xs,loss)
