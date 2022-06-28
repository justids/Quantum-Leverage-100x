import ast
import matplotlib.pyplot as plt

with open('output.txt') as f:
  input = f.readlines()
  input = input[1::2]

fig, ax = plt.subplots(figsize=(10, 10))

iters = []
functionValues = []

for ix, dic in enumerate(input):
  value = (100 - ast.literal_eval(dic)['0']) * 0.01
  iters.append(6.4 * ix)
  functionValues.append(value)

ax.plot(
  iters,
  functionValues,
  label="label"
)
plt.setp(ax, xlabel='Xlabel')
plt.setp(ax, ylabel='Ylabel')

fig.suptitle("Title")

# ax.legend(loc=1)

plt.savefig("q1_plot_3000.png", dpi=300)

plt.show()

"""
maxM = 25
n = 7
jobs = []
k = 3
for m in range(1, maxM+1):
  qc = QuantumCircuit(1, 1)
  qc.x(0)
  qc.x(0)
  for _ in range(m*10):
    qc.delay(640, 0, unit="us")
  qc.x(0)
  qc.x(0)
  qc.barrier()
  qc.measure(0, 0)
  circTranspiled = transpile(qc, backend, optimization_level = 0)
  jobs.append(backend.run(circTranspiled, shots=100))

for m in range(maxM * 1):
  result = jobs[m].result()
  counts = result.get_counts()
  print(m // maxM, m % maxM)
  print(counts)
"""

"""
0 0
{'0': 98, '1': 2}
0 1
{'0': 98, '1': 2}
0 2
{'0': 100}
0 3
{'0': 98, '1': 2}
0 4
{'0': 100}
0 5
{'0': 97, '1': 3}
0 6
{'0': 100}
0 7
{'0': 98, '1': 2}
0 8
{'0': 100}
0 9
{'0': 100}
0 10
{'0': 99, '1': 1}
0 11
{'0': 99, '1': 1}
0 12
{'0': 99, '1': 1}
0 13
{'0': 99, '1': 1}
0 14
{'0': 99, '1': 1}
0 15
{'0': 99, '1': 1}
0 16
{'0': 98, '1': 2}
0 17
{'0': 99, '1': 1}
0 18
{'0': 98, '1': 2}
0 19
{'0': 100}
0 20
{'0': 100}
0 21
{'0': 100}
0 22
{'0': 99, '1': 1}
0 23
{'0': 99, '1': 1}
0 24
{'0': 99, '1': 1}
"""
