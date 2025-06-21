from engine import Value, Layer, MLP


n = MLP(8, [16, 8, 4, 1])

x_train = [
    [Value(1.0), Value(2.0), Value(3.0)],
    [Value(2.0), Value(-1.0), Value(0.5)],
    [Value(-1.5), Value(2.2), Value(0.0)],
    [Value(2.0), Value(2.0), Value(-3.0)],
    [Value(1.0), Value(-2.0), Value(3.0)],
    [Value(1.5), Value(-1.0), Value(0.5)],
    [Value(-1.5), Value(1.2), Value(0.0)],
    [Value(2.2), Value(3.8), Value(-3.0)],
]

y_train = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
loss_history=[]
for i in range(500):

    ypred = [n(x) for x in x_train]
    loss = sum((yout - yt)**2 for yt, yout in zip(y_train, ypred))
    loss_history.append(loss._data)

    for p in n.params():
        p._grad = 0.0

    loss.backward()

    for p in n.params():
        p._data += -0.01 * p._grad

    print(f"Epoch {i}: Loss = {loss._data:.4f}")

y_prd = [n(x) for x in x_train]
print("Predictions:", [y._data for y in y_prd])

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.savefig("loss graph.png")
plt.show()