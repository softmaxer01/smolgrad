import engine as e

x1 = e.Value(2.0, _label='x1')
x2 = e.Value(0.0, _label='x2')
w1 = e.Value(-3.0, _label='w1')
w2 = e.Value(1.0, _label='w2')
b = e.Value(6.8813735870, _label='b')

x1w1 = x1 * w1
x1w1._label = 'x1*w1'
x2w2 = x2 * w2
x2w2._label = 'x2*w2'

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2._label = 'x1w1+x2w2'
n = x1w1x2w2 + b
n._label = 'n'
o = n.tanh()
o._label = 'output'

o._grad = 1.0

topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)

for node in reversed(topo):
    node._backward()

print("=" * 40)
print("Output:", o)
print("Gradients:")
print(f"o: {o._grad}")
print(f"n: {n._grad}")
print(f"x1w1+x2w2: {x1w1x2w2._grad}")
print(f"b: {b._grad}")
print(f"x1w1: {x1w1._grad}")
print(f"x2w2: {x2w2._grad}")
print(f"x1: {x1._grad}")
print(f"x2: {x2._grad}")
print(f"w1: {w1._grad}")
print(f"w2: {w2._grad}")
