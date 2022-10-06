import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import sympy as sp
from itertools import combinations_with_replacement

# %%
# Load data
raw_data = np.load("../workshop2_neuralnets/data/simple_wave.npz")
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
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
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

tx = optax.adam(1e-3)
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
h1 = h[0,:]
hhat = [float(model.apply(params,jnp.array([x[i],t[0]]))[0]) for i in range(len(x))]
plt.plot(x,h1)
plt.plot(x,hhat)
plt.savefig("testgraph.png")
plt.show()

# %%
# Test grad
t = 0.3
def model_sub(x):
    new_x = jnp.array([x,t])
    return model.apply(params, new_x)[0]
# test_pt = jnp.array([0.3,0.3])
# print(model_sub(test_pt))
test_x = 0.3
print(model_sub(test_x))
jax.grad(model_sub)(test_x)
jax.grad(jax.grad(model_sub))(test_x)
jax.grad(jax.grad(jax.grad(model_sub)))(test_x)

# %%
# symoblic library
x_sym,t_sym = sp.symbols("x t")
h_sym = sp.Function("h")
dh = sp.diff(h_sym(x_sym,t_sym), t_sym)
expr = dh*dh

def substitute_arr(expr, sym, arr):
    result = np.zeros_like(arr)
    for index in np.ndindex(arr.shape):
        result[index] = expr.subs(sym, arr[index])
    return result

print(substitute_arr(expr, dh, np.array([0,1,2])))

# Make library
poly_orders = range(4)
diff_orders = range(4)

terms = [sp.diff(h_sym(x_sym,t_sym), x_sym, i) for i in diff_orders]
permutations = []
