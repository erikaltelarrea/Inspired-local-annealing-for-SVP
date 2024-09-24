import pickle
from fpylll import FPLLL, GSO, LLL, IntegerMatrix, Enumeration, EnumerationError
import random
from multiprocessing import Pool
import numpy as np
import sys
import os

seed = 2021
LLL_reduce = True
cpus = 4
num_instances = 100 // cpus

if len(sys.argv) != 3 or not sys.argv[1].isnumeric() or not sys.argv[2].isnumeric():
    print("Rank limits not provided. The correct usage is 'python script_name.py rank_min rank_max'")
    sys.exit()

ranks = range(int(sys.argv[1]), int(sys.argv[2]) + 1)
q = 65537
full_dim = 180

def run_enum(M, pruned=False, pruned_dist=0):
    enum_obj = Enumeration(M)
    dist, exp = M.get_r_exp(0, 0)
    try:
        if pruned:
            result = enum_obj.enumerate(0, M.d, pruned_dist, 1)
        else:
            result = enum_obj.enumerate(0, M.d, dist, exp)
        if result:
            dist, v = round(result[0][0]), result[0][1]
            return dist, v
    except EnumerationError:
        return None, None
    return None, None

def generate_basis(d, q, _ranks):
    A = IntegerMatrix.random(d, "qary", k=d // 2, q=q)
    if LLL_reduce:
        return LLL.reduction(A)[:_ranks[-1]]
    return A[:_ranks[-1]]

def instance_calc(instance_seed):
    FPLLL.set_random_seed(instance_seed)
    basis = generate_basis(full_dim, q, ranks)
    instance_results = []

    for rank in ranks:
        A = basis[:rank]
        M = GSO.Mat(A)
        M.update_gso()

        dist, v = run_enum(M)
        v_array = np.array([int(v[i]) for i in range(A.nrows)]) if v is not None else None
        basis_matrix = np.array([[int(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)])
        instance_data = (
            basis_matrix,
            np.dot(basis_matrix, basis_matrix.T),
            v_array,
            dist
        )
        instance_results.append((rank, instance_data))

    return instance_results

def save_instance_data(rank, instance_data):
    filename = f"n={rank}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            existing_data = pickle.load(f)
    else:
        existing_data = []

    existing_data.append(instance_data)

    with open(filename, 'wb') as f:
        pickle.dump(existing_data, f)
    print(f"Data for rank {rank} saved to {filename}")

if __name__ == "__main__":
    random.seed(seed)
    pool = Pool(cpus)

    for i in range(num_instances):
        res = pool.map(instance_calc, [random.randint(0, 99999999) for _ in range(cpus)])
        for instance_result in res:
            for rank, instance_data in instance_result:
                save_instance_data(rank, instance_data)
