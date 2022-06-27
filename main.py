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

from custom_iae import CustomIterativeAmplitudeEstimation


from qiskit import IBMQ
ibm_token = 'f42d45ee0b069b6c08e97fca17c15f5bac11363126fbbf31436bdbb522fd9c53bbed7cf41eb3d40d3f5653c54096c80ccf0231a6a258795ad5e66f0bf3fafeae' # https://quantum-computing.ibm.com/account
ibmq_account = IBMQ.enable_account(ibm_token)
ibmq_provider = IBMQ.get_provider(hub='ibm-q-skku', group='snu', project='snu-students')
# print("Available IBMQ backends:")
# print(ibmq_provider.backends())

ibmq_backend_sim = ibmq_provider.get_backend('ibmq_qasm_simulator')
ibmq_backend_real = ibmq_provider.get_backend('ibmq_jakarta')
aer_backend = Aer.get_backend("aer_simulator")
backend = aer_backend
iterationShot = 100
qi = QuantumInstance(backend, shots=iterationShot)

print("Used backend: ", backend)

# params
# number of qubits to represent the uncertainty
num_uncertainty_qubits = 2

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

print("num_uncertainty_qubits : %d" % num_uncertainty_qubits)
print("initial spot price     : %.4f" % S)
print("volatility             : %.4f" % vol)
print("annual interest rate   : %.4f" % r)
print("days to maturity       : %.4f" % (T * 360))

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1.896
print("strike_price           : %.4f" % strike_price)

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)

def mkEuropeanCallObjective(c_approx):
    # setup piecewise linear objective function
    breakpoints = [low, strike_price]
    slopes = [0, 1]
    offsets = [0, 0]
    f_min = 0
    f_max = high - strike_price
    european_call_objective = LinearAmplitudeFunction(
        num_uncertainty_qubits,
        slopes,
        offsets,
        domain=(low, high),
        image=(f_min, f_max),
        breakpoints=breakpoints,
        rescaling_factor=c_approx,
    )

    return european_call_objective


def mkPayoffCircuit(european_call_objective):
    # construct A operator for QAE for the payoff function by
    # composing the uncertainty model and the objective
    num_qubits = european_call_objective.num_qubits
    european_call = QuantumCircuit(num_qubits)
    european_call.append(uncertainty_model, range(num_uncertainty_qubits))
    european_call.append(european_call_objective, range(num_qubits))

    return european_call


def runAndGetCounts(qc, shots=1024):
    """Simulates the given circuit and plots the result."""
    result = execute(qc, backend, shots=shots).result()
    counts = result.get_counts(qc)

    return counts

def plotFromCounts(counts, title='result'):
    plot_histogram(counts, title=title)



# main part

# set the approximation scaling for the payoff function
c_approx = 0.25
print("approximation scaling for the payoff function : %.4f" % c_approx)

mainCirc = QuantumCircuit(num_uncertainty_qubits * 2 + 1, 1)
europeanCallObjective = mkEuropeanCallObjective(c_approx)
payoffCircuit = mkPayoffCircuit(europeanCallObjective)
payoffCircuit.draw()
mainCirc.compose(payoffCircuit, range(num_uncertainty_qubits * 2 + 1), inplace=True)
mainCirc.measure([num_uncertainty_qubits], [0])
#mainCirc.draw()

# european_call_objective.post_processing
shotsOnce = 1024
counts = runAndGetCounts(mainCirc, shots=shotsOnce)
probability = counts['1'] / (counts['1'] + counts['0'])

# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = np.dot(uncertainty_model.probabilities, np.maximum(0, uncertainty_model.values - strike_price))

print("==========================================")
print("Simple method (Just shot %d times on PF \ket{0}^%d)" % (shotsOnce, num_uncertainty_qubits))
print("Exact value:        \t%.4f" % exact_value)
print("Estimated value:    \t%.4f" % europeanCallObjective.post_processing(probability))

# mainCirc.draw()

# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------


# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

# qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
problem = EstimationProblem(
    state_preparation=payoffCircuit,
    objective_qubits=[num_uncertainty_qubits],
    post_processing=europeanCallObjective.post_processing,
)
# construct amplitude estimation
ae = CustomIterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)

result = ae.estimate(problem)

conf_int = np.array(result.confidence_interval_processed)
print("==========================================")
print("epsilon                : %.4f" % epsilon)
print("alpha                  : %.4f" % alpha)
print("IterativeAmplitudeEstimation (Shot %d times for every iteration)" % (iterationShot))
print("Exact value:        \t%.4f" % exact_value)
print("Estimated value:    \t%.4f" % (result.estimation_processed))
print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))

#ae_circuit = ae.construct_circuit(problem, k=1)
#from qiskit import transpile
#basis_gates = ["h", "ry", "cry", "cx", "ccx", "p", "cp", "x", "s", "sdg", "y", "t", "cz"]
#transpile(ae_circuit, basis_gates=basis_gates, optimization_level=2).draw("mpl", style="iqx", filename="circuit.png")
