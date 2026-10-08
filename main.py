


import networkx as nx
import numpy as np

def generate_traffic_graph(n_nodes, edge_prob=0.5, seed=42):
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)

    # assign weights = traffic intensity
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)

    return G


# def build_qubo_matrix(G):
#     n = len(G.nodes)
#     Q = np.zeros((n, n))

#     for (i, j, data) in G.edges(data=True):
#         w = data['weight']

#         if np.random.rand() < 0.5:
#             # prefer different phases
#             Q[i][i] += w
#             Q[j][j] += w
#             Q[i][j] += -2 * w
#         else:
#             # prefer same phase
#             Q[i][j] += 2 * w

#     return Q

import numpy as np

def build_multiphase_qubo(G, num_phases=3, alpha=10, beta=2, gamma=3):
    """
    alpha = constraint strength
    beta = flow reward
    gamma = coordination strength
    """

    n = len(G.nodes)
    N = n * num_phases  # total binary variables

    Q = np.zeros((N, N))

    def idx(i, p):
        return i * num_phases + p

    # -------------------------
    # (1) One-phase constraint
    # -------------------------
    for i in range(n):
        for p in range(num_phases):
            Q[idx(i,p)][idx(i,p)] += -alpha

            for q in range(p+1, num_phases):
                Q[idx(i,p)][idx(i,q)] += 2 * alpha

    # -------------------------
    # (2) Flow + coordination
    # -------------------------
    for (i, j, data) in G.edges(data=True):
        w = data['weight']

        for p in range(num_phases):
            # reward matching phases (green wave)
            Q[idx(i,p)][idx(j,p)] += -gamma * w

        # encourage compatible phases
        for p in range(num_phases):
            for q in range(num_phases):
                if p != q:
                    Q[idx(i,p)][idx(j,q)] += beta * w

    return Q


def add_balance_penalty(Q, alpha=5):
    n = Q.shape[0]

    for i in range(n):
        Q[i][i] += alpha * (1 - n)

    for i in range(n):
        for j in range(i+1, n):
            Q[i][j] += 2 * alpha

    return Q

# def build_qubo_matrix(G):
#     n = len(G.nodes)
#     Q = np.zeros((n, n))

#     for (i, j, data) in G.edges(data=True):
#         w = data['weight']
        
#         # penalty if same phase → congestion
#         Q[i][i] += w
#         Q[j][j] += w
#         Q[i][j] += -2 * w  # encourages different phases

#     return Q

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def qubo_to_qiskit(Q):
    n = Q.shape[0]
    qp = QuadraticProgram()

    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": Q[i][i] for i in range(n)}
    quadratic = {}

    for i in range(n):
        for j in range(i+1, n):
            if Q[i][j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = Q[i][j]

    qp.minimize(linear=linear, quadratic=quadratic)

    return qp


from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def solve_qaoa(qp, p=1, optimizer_name="COBYLA"):
    if optimizer_name == "COBYLA":
        optimizer = COBYLA(maxiter=100)
    else:
        optimizer = SPSA(maxiter=100)

    sampler = Sampler()

    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)

    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)

    return result.x, result.fval


#5

def greedy_solution(Q):
    n = Q.shape[0]
    x = np.zeros(n)

    for i in range(n):
        x[i] = 1
        val1 = x @ Q @ x

        x[i] = 0
        val0 = x @ Q @ x

        x[i] = 1 if val1 < val0 else 0

    return x, x @ Q @ x

def simulated_annealing(Q, steps=1000, temp=10):
    n = Q.shape[0]
    x = np.random.randint(0, 2, size=n)

    def energy(x):
        return x @ Q @ x

    best_x = x.copy()
    best_val = energy(x)

    for t in range(steps):
        i = np.random.randint(n)
        x_new = x.copy()
        x_new[i] ^= 1  # flip bit

        delta = energy(x_new) - energy(x)

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x = x_new

        if energy(x) < best_val:
            best_x = x.copy()
            best_val = energy(x)

        temp *= 0.995

    return best_x, best_val

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

def exact_solution(qp):
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = exact.solve(qp)
    return result.x, result.fval


#6
import time

def run_experiment(n_nodes):
    G = generate_traffic_graph(n_nodes)
    Q = build_multiphase_qubo(G)
    Q = add_balance_penalty(Q, alpha=5)
    qp = qubo_to_qiskit(Q)

    results = {}

    # QAOA
    start = time.time()
    x_qaoa, val_qaoa = solve_qaoa(qp, p=2)
    results["qaoa"] = (val_qaoa, time.time() - start)

    # Greedy
    start = time.time()
    _, val_greedy = greedy_solution(Q)
    results["greedy"] = (val_greedy, time.time() - start)

    # Simulated Annealing
    start = time.time()
    _, val_sa = simulated_annealing(Q)
    results["annealing"] = (val_sa, time.time() - start)

    # Exact (only small n)
    if n_nodes <= 8:
        start = time.time()
        _, val_exact = exact_solution(qp)
        results["exact"] = (val_exact, time.time() - start)

    return results

#7

def run_all():
    sizes = [4, 6, 8, 10]

    all_results = {}

    for n in sizes:
        print(f"\nRunning for n={n}")
        results = run_experiment(n)

        for method, (val, t) in results.items():
            print(f"{method}: value={val:.2f}, time={t:.3f}s")

        all_results[n] = results

    return all_results

if __name__ == "__main__":
    results = run_all()