import matplotlib.pyplot as plt

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem, AmplitudeEstimation
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
import math

from sympy import are_similar

from custom_iae import CustomIterativeAmplitudeEstimation


from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeParis


#from qiskit import IBMQ
#ibm_token = 'f42d45ee0b069b6c08e97fca17c15f5bac11363126fbbf31436bdbb522fd9c53bbed7cf41eb3d40d3f5653c54096c80ccf0231a6a258795ad5e66f0bf3fafeae' # https://quantum-computing.ibm.com/account
#if 'ibmq_account' not in globals():
#  ibmq_account = IBMQ.enable_account(ibm_token)
#ibmq_provider = IBMQ.get_provider(hub='ibm-q-skku', group='snu', project='snu-students')
#print("Available IBMQ backends:")
#print(ibmq_provider.backends())

#ibmq_backend_sim = ibmq_provider.get_backend('ibmq_eaqasm_simulator')
#ibmq_backend_real = ibmq_provider.get_backend('ibm_washington')
aer_backend = Aer.get_backend("aer_simulator")
fake_backend = AerSimulator.from_backend(FakeParis())
#backend = fake_backend
#iterationShot = 100
#qi = QuantumInstance(backend, shots=iterationShot)
#print("Used backend: ", backend)

# ====================

## Written by Eliott Rosenberg in 2021. If this is useful for you, please include me in your acknowledgments.


import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler import PassManager

def random_compile(qc):
    def apply_padded_cx(qc,qubits,type='random'):
        if type == 'random':
            type = np.random.randint(16)
        if type == 0:
            qc.cx(qubits[0],qubits[1])
        elif type == 1:
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[1])
        elif type == 2:
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
            qc.y(qubits[1])
        elif type == 3:
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
            qc.z(qubits[1])
        elif type == 4:
            qc.y(qubits[0])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
            qc.x(qubits[1])
        elif type == 5:
            qc.y(qubits[0])
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
        elif type == 6:
            qc.y(qubits[0])
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
            qc.z(qubits[1])
        elif type == 7:
            qc.y(qubits[0])
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
            qc.y(qubits[1])
        elif type == 8:
            qc.x(qubits[0])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
            qc.x(qubits[1])
        elif type == 9:
            qc.x(qubits[0])
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
        elif type == 10:
            qc.x(qubits[0])
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
            qc.z(qubits[1])
        elif type == 11:
            qc.x(qubits[0])
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
            qc.y(qubits[1])
        elif type == 12:
            qc.z(qubits[0])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
        elif type == 13:
            qc.z(qubits[0])
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
            qc.x(qubits[1])
        elif type == 14:
            qc.z(qubits[0])
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[1])
        elif type == 15:
            qc.z(qubits[0])
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[1])
        return qc
    if qc.num_clbits > 0:
        qc2 = QuantumCircuit(qc.num_qubits,qc.num_clbits)
    else:
        qc2 = QuantumCircuit(qc.num_qubits)
    for gate in qc:
        if gate[0].name == 'cx':
            # pad cx gate with 1-qubit gates.
            qc2 = apply_padded_cx(qc2,gate[1])
        else:
            if qc.num_clbits > 0:
                qc2.append(gate[0],gate[1],gate[2])
            else:
                qc2.append(gate[0],gate[1])
    return simplify(qc2)



def simplify(qc):
    p = Optimize1qGatesDecomposition(basis=['rz','sx','x','cx'])
    pm = PassManager(p)
    return pm.run(qc)

#############


n = 7

qc = QuantumCircuit(n, n)

for _ in range(100):
    a = np.random.randint(n)
    b = np.random.randint(n)
    while(a == b):
        b = np.random.randint(n)

    # type ['x', 'h', 'cx']
    gates = ['x', 'rz', 'sx', 'cx']
    t = np.random.randint(len(gates))
    print(t, gates)
    gate = gates[t]

    if (gate == 'x'):
        qc.x(a)
    elif (gate == 'rz'):
        qc.rz(np.pi / 2, a)
    elif (gate == 'sx'):
        qc.sx(a)
    elif (gate == 'cx'):
        qc.cx(a, b)

qc.measure(range(n), range(n))

print(qc)

random_compiled_qc = random_compile(qc)

print(random_compiled_qc)

def getValuesBySortedKey(circuit, backend, shots):
    job = backend.run(circuit, shots = shots)
    counts = job.result().get_counts()
    valuesBySortedKey = [0 for _ in range(2**n)]

    sum = 0
    for key in counts:
        sum += counts[key]
        valuesBySortedKey[int(key, 2)] = counts[key]
    
    return valuesBySortedKey

shots = 10000

aer_qc = getValuesBySortedKey(qc, aer_backend, shots)
fake_qc = getValuesBySortedKey(qc, fake_backend, shots)
fake_random_compiled_qc = getValuesBySortedKey(random_compiled_qc, fake_backend, shots)

print(aer_qc)
print(fake_qc)
print(fake_random_compiled_qc)
print(np.corrcoef(aer_qc, fake_qc))
print(np.corrcoef(aer_qc, fake_random_compiled_qc))
print(np.corrcoef(fake_qc, fake_random_compiled_qc))
"""
for gate in qc:
    print(gate)
    print(gate[0])
    print(gate[1])
    print(gate[2])
    print("")
    if gate[0].name == 'cx':
        print("hello")
"""


        



