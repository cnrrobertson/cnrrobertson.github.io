import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
# from tensorflow import keras
# from tensorflow.keras import layers

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
for i in range(10000):
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
def model_sub(x):
    return model.apply(params, x)[0]
test_pt = jnp.array([0.3,0.3])
print(model_sub(test_pt))
jax.grad(model_sub)(test_pt)

# %%
# Create noisy data
# x_data = np.linspace(-10, 10, num=1000)
# y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=1000)
x_data = x
y_data = h[10,:]
# y_data = h[0,:] + 0.005*np.random.normal(size=len(x))
print('Data created successfully')
# Create the model 
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 20, activation = 'relu'))
model.add(keras.layers.Dense(units = 20, activation = 'relu'))
model.add(keras.layers.Dense(units = 20, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")
# Training
model.fit( x_data, y_data, epochs=1000)
y_predicted = model.predict(x_data)

# Display the result
plt.scatter(x_data[::1], y_data[::1])
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
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
# %%
# # Keras style
# model = keras.Sequential(
#     [
#         keras.Input(shape=(2,)),
#         layers.Dense(30, activation="tanh"),
#         layers.Dense(1, activation="tanh")
#     ]
# )
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss="mean_squared_error",
#     metrics=["mean_squared_error"]
# )
# model.fit(
#     data, epochs=10
# )

# %%
# Jax style
# def model(params,inputs):
#     for W,b in params:
#         outputs = jnp.dot(inputs,W) + b
#         inputs = jnp.tanh(outputs)
#     return outputs
#
# def loss(params, inputs, targets):
#     preds = model(params, inputs)
#     return jnp.sum((preds - targets)**2)
# Initialize parameters
# rng = jax.random.PRNGKey(42)
# params = [
#     (jax.random.uniform(rng,(2,10)), jax.random.uniform(rng,(10,))),
#     (jax.random.uniform(rng,(10,5)), jax.random.uniform(rng,(5,))),
#     (jax.random.uniform(rng,(5,1)), jax.random.uniform(rng,(1,)))
# ]
# tx = optax.adam(1e-3)
# opt_state = tx.init(params)
# loss_grad_fn = jax.jit(jax.value_and_grad(loss))
#
# for i in range(10000):
#     loss_val, grads = loss_grad_fn(params, data[i%len(data)][0], data[i%len(data)][1])
#     updates, opt_state = tx.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     if i % 100 == 0:
#         print("Loss step {}: {}".format(i,loss_val))
