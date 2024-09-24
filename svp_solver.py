from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import numpy as np
import torch
import importlib
from itertools import product

import local_annealing_svp 

# Define the range of n
n_range = [39]

def valid(v, k):
    """Checks if the vector v is inside the search space of local dimension 2 ** k"""
    for i in range(len(v)):
        if v[i] < -2 ** (k - 1) or v[i] >= 2 ** (k - 1):
            return False
    return True

def process_single_shot(couplings_bin, fields_bin, identity, qubits, qudits, r, s, G):
    """Runs a single shot of the LQA algorithm and returns success metrics and config."""
    machine = local_annealing_svp.Lqa_svp(couplings_bin, fields_bin, identity, qubits, qudits, r, s)
    machine.minimise(step=0.0091, N=4000, g=1, f=0.15)
    config = np.array(machine.config)
    config_norm_sq = vector_norm_sq(config, G)
    return config, config_norm_sq

def vector_norm_sq(x, G):
    """Calculates the squared norm of a vector given a Gram matrix."""
    x = np.array(x, dtype=np.float64)
    return np.dot(x.T, np.dot(G, x))

def process_single_s_r(s, r, data, n):
    """Processes data for a single s and r pair across multiple shots."""
    for k in range(100):
        G = data[k][1]
        shortest_vector = data[k][2]
        shortest_norm_sq = data[k][3]
        qubits = 1
        
        if valid(shortest_vector, qubits) or valid(-shortest_vector, qubits):
            qudits = len(data[k][0])
            shots = 100

            couplings_bin = torch.zeros([qudits * qubits, qudits * qubits])
            fields_bin = torch.zeros([qudits * qubits])
            identity = 0

            G = np.array(G, dtype=np.float64)
            G_norm = G 
            d = qudits
            gaussian_heuristic = (d/(2 * np.pi * np.e)) ** (1/2) * (np.linalg.det(G_norm)) ** (1/(2*d))

            for i in range(qudits):
                for j in range(qudits):
                    c_ham = G_norm[i, j] / 4
                    identity += c_ham
                    for p in range(qubits):
                        ip_qubit = i * qubits + p
                        jp_qubit = j * qubits + p
                        for q in range(qubits):
                            jq_qubit = j * qubits + q
                            if ip_qubit != jq_qubit:
                                couplings_bin[ip_qubit, jq_qubit] += c_ham * (2 ** (p + q))
                            else:
                                identity += c_ham * (2 ** (p + q))
                        fields_bin[ip_qubit] += c_ham * (2 ** p)
                        fields_bin[jp_qubit] += c_ham * (2 ** p)
            
            r_ = r * gaussian_heuristic ** 2
            s_ = s 
            best_ratio = float('inf')
            best_vector = None

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_single_shot, couplings_bin, fields_bin, identity, qubits, qudits, r_, s_, G)
                    for _ in range(shots)
                ]
                
                for shot_number, future in enumerate(as_completed(futures), start=1):
                    config, config_norm_sq = future.result()
                    ratio = np.sqrt(config_norm_sq / shortest_norm_sq)
                    print(f'Shot {shot_number}: Vector {np.dot(config, data[k][0])}, Ratio {ratio:.10f}')

                    # Update best ratio and best vector if current ratio is better
                    if ratio > 0 and ratio < best_ratio:
                        best_ratio = ratio
                        best_vector = config
            
            # After all shots, print the best ratio and vector for the instance
            if best_vector is not None:
                print(f'Instance {k}: Best Ratio {best_ratio:.10f}, Best Vector {np.dot(best_vector, data[k],0)}')

def main():
    """Main function to process multiple n values."""
    # Define penalization hyperparameters
    s = 4.6e-7
    r = .72

    for n in n_range:
        # Open the corresponding .pkl file for each n
        with open(f'n={n}.pkl', 'rb') as file:
            data = pickle.load(file)
        
        process_single_s_r(s, r, data, n)

if __name__ == "__main__":
    main()