import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import sympy as sp
from itertools import combinations_with_replacement
import sklearn.linear_model as lm

# %%
# Load data
raw_data = np.load("data/simple_wave.npz")
h = raw_data["h"].astype(jnp.float32)
x = raw_data["x"].astype(jnp.float32)
t = raw_data["t"].astype(jnp.float32)
# Mean center, std center data
h = (h - jnp.mean(h)) / jnp.std(h)
x = (x - jnp.mean(x)) / jnp.std(x)
t = (t - jnp.mean(t)) / jnp.std(t)
data = []
batch_size = 10000
index_list = list(np.ndindex(h.shape))
for ind in range(0,len(index_list),batch_size):
    inds = index_list[ind:ind+batch_size]
    xts = np.array([[x[j],t[i]] for i,j in inds])
    hs = np.array([[h[i,j]] for i,j in inds])
    if len(xts) == batch_size:
        data.append((xts,hs))

# %%
# Flax style
class MyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(12)(x)
        x = nn.tanh(x)
        x = nn.Dense(12)(x)
        x = nn.tanh(x)
        x = nn.Dense(12)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

rng1,rng2 = jax.random.split(jax.random.PRNGKey(42))
model = MyNet()
init_data = jax.random.normal(rng1,(2,))
params = model.init(rng2,init_data)
print(jax.tree_util.tree_map(lambda x: x.shape, params))

# Training
@jax.jit
def mse(params,input,targets):
    def squared_error(x,y):
        pred = model.apply(params,x)
        return jnp.mean((y - pred)**2)
    return jnp.mean(jax.vmap(squared_error)(input,targets),axis=0)

tx = optax.adam(1e-1)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

all_x = jnp.array([data[i][0] for i in range(len(data))])
all_y = jnp.array([data[i][1] for i in range(len(data))])
for i in range(1000):
    x_batch = data[i%len(data)][0]
    y_batch = data[i%len(data)][1]
    loss_val, grads = loss_grad_fn(params, x_batch, y_batch)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 100 == 0:
        valid_loss = mse(params,all_x,all_y)
        print("Validation loss step {}: {}".format(i,valid_loss))

# %%
# Plot fit

X,T = jnp.meshgrid(x,t)
xt_points = jnp.vstack([X.flatten(),T.flatten()]).T
hhat = model.apply(params,xt_points).reshape(X.shape)
# hhat_t = h_t.reshape(X.shape)
hhat_t = np.array(term_matrix[diff_terms[1]]).reshape(X.shape)

fig = plt.figure()
p1 = plt.plot(x,h[0,:])[0]
p2 = plt.plot(x,hhat[0,:])[0]
p3 = plt.plot(x,hhat_t[0,:])[0]

def anim_func(j):
    p1.set_ydata(h[j,:])
    p2.set_ydata(hhat[j,:])
    p3.set_ydata(hhat_t[j,:])

approx_anim = anim.FuncAnimation(fig, anim_func, range(len(t)))
plt.show()

# %%
# Test grad
t = 0.3
def model_for_diff(x,t):
    new_x = jnp.array([x,t])
    return model.apply(params, new_x)[0]

def model_in_x(x):
    new_x = jnp.array([x,t])
    return model.apply(params, new_x)[0]

def model_in_t(t):
    new_t = jnp.array([x,t])
    return model.apply(params, new_t)[0]

test_x = 0.3
print(model_in_x(test_x))
jax.grad(model_in_x)(test_x)
jax.grad(model_for_diff,0)(test_x,t)
jax.grad(jax.grad(model_in_x))(test_x)
jax.grad(jax.grad(jax.grad(model_in_x)))(test_x)

# Apply to all x
jax.lax.map(jax.grad(model_in_x), X.flatten())

# %%
# symoblic library
x_sym,t_sym = sp.symbols("x t")
h_sym = sp.Function("h")

# Make library
max_poly_order = 4
max_diff_order = 4

# diff_terms = [sp.diff(h_sym(x_sym,t_sym), x_sym, i) for i in range(max_diff_order+1)]
diff_terms = [h_sym(x_sym,t_sym)] + [sp.Function(str(h_sym)+"_"+(i*str(x_sym)))(x_sym,t_sym) for i in range(1,max_diff_order+1)]

# Differentiate model and store results with autodiff
diff_term_values = {}
for i in range(max_diff_order+1):
    diff_func = model_for_diff
    for _ in range(i):
        diff_func = jax.grad(diff_func, 0)
    def unpack_diff_func(x):
        new_x,new_t = x
        return diff_func(new_x,new_t)
    diff_term_values[diff_terms[i]] = np.array(jax.lax.map(unpack_diff_func, xt_points))

# Construct terms
term_values = {}
for po in range(max_poly_order+1):
    if po == 0:
        term = sp.core.numbers.One()
        term_values[term] = np.ones(xt_points.shape[0])
    else:
        combos = combinations_with_replacement(diff_terms,po)
        for combo in combos:
            term = 1
            temp_term_value = 1
            for combo_term in combo:
                term *= combo_term
                temp_term_value *= diff_term_values[combo_term]
            term_values[term] = temp_term_value

# Time derivative
def unpack_diff_func(x):
    new_x,new_t = x
    return jax.grad(model_for_diff,1)(new_x,new_t)

# term_values[sp.Function(str(h_sym)+"_"+str(t_sym))] = jax.lax.map(unpack_diff_func, xt_points)
h_t = -np.array(jax.lax.map(unpack_diff_func, xt_points))

# %%
term_matrix = pd.DataFrame(term_values)
ols = lm.LinearRegression()
ols.fit(term_matrix,h_t)
print(ols.coef_ > 1e-5)
lasso = lm.Lasso(2)
lasso.fit(term_matrix,h_t)

