import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

# %%
# Load data
raw_data = np.load("../workshop2_neuralnets/data/simple_wave.npz")
h = raw_data["h"].astype(jnp.float32)
x = raw_data["x"].astype(jnp.float32)
t = raw_data["t"].astype(jnp.float32)
data = []
batch_size = 1
index_list = list(np.ndindex(h.shape))
for ind in range(0,len(index_list),batch_size):
    xts = np.array(index_list[ind:ind+batch_size])
    hs = np.array([[h[i,j]] for i,j in xts])
    if len(xts) == batch_size:
        data.append((xts,hs))
# %%
# Jax style
def model(params,inputs):
    for W,b in params:
        outputs = jnp.dot(inputs,W) + b
        inputs = jnp.tanh(outputs)
    return outputs

def loss(params, inputs, targets):
    preds = model(params, inputs)
    return jnp.sum((preds - targets)**2)

# Initialize parameters
rng = jax.random.PRNGKey(42)
params = [
    (jax.random.uniform(rng,(2,10)), jax.random.uniform(rng,(10,))),
    (jax.random.uniform(rng,(10,5)), jax.random.uniform(rng,(5,))),
    (jax.random.uniform(rng,(5,1)), jax.random.uniform(rng,(1,)))
]

tx = optax.adam(1e-3)
opt_state = tx.init(params)
loss_grad_fn = jax.jit(jax.value_and_grad(loss))

for i in range(10000):
    loss_val, grads = loss_grad_fn(params, data[i%len(data)][0], data[i%len(data)][1])
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 100 == 0:
        print("Loss step {}: {}".format(i,loss_val))

# %%
# Plot fit
h1 = h[0,:]
hhat = [float(model(params,jnp.array([t[0],x[i]]))[0]) for i in range(len(x))]
plt.plot(x,h1)
plt.plot(x,hhat)
plt.show()
# %%
# Create model
# import haiku as hk
# def mse(hhat, h):
#     return jnp.linalg.norm(hhat-h)**2
#
# def model(xt):
#     mlp = hk.Sequential([
#         hk.Linear(20), jax.nn.relu,
#         hk.Linear(10), jax.nn.relu,
#         hk.Linear(1)
#     ])
#     return mlp(xt)
#
# def loss(xt, h):
#     hhat = model(xt)
#     return jnp.mean(mse(hhat, h))
#
# model_fn = hk.transform(model)
# loss_fn = hk.transform(loss)
# loss_fn = hk.without_apply_rng(loss_fn) # Don't require a random number in apply function
#
# rng = jax.random.PRNGKey(42)
# params = loss_fn.init(rng, data[0][0], data[0][1])
#
# def update_rule(param, update):
#     return param - 0.01 * update
#
# for xt,h in data:
#     grads = jax.grad(loss_fn.apply)(params, xt, h)
#     params = jax.tree_map(update_rule, params, grads)
