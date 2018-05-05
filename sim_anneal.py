from search import *
from scipy.stats import binom
import random
import math

STATE = 'NY'
N_DISTRICTS = 27

def pop_obj(labeling):
    pops_district = np.matmul(labeling, populations)
    pop_per_district = populations.sum() / N_DISTRICTS
    return ((pops_district - pop_per_district) ** 2).sum()

def gerry_obj(labeling):
    pops_district = np.matmul(labeling, populations)
    dems_distrct = np.matmul(labeling, voter_data[0])
    return binom.cdf(pops_district/2, pops_district, dems_distrct/pops_district).sum()

def objective(labeling):
    return gerry_obj(labeling) + (10e-12 * pop_obj(labeling))

adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)

populations = voter_data.sum(axis=0)
pop_per_district = populations.sum() / N_DISTRICTS

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)

assert all_districts_connected(initial_labeling, adj)

visited = set(label_hash(initial_labeling))

labeling = initial_labeling

prev_obj = objective(labeling)
T = 1.0
T_min = 0.000001
alpha = 0.9

while T > T_min:
    i = 1
    candidates = None
    while i <= 100:
        if candidates == None:
            candidates = neighbor(labeling, adj)
        new_labeling = random.choice(candidates)
        labeling[new_labeling[1], new_labeling[0]] = 0
        labeling[new_labeling[2], new_labeling[0]] = 1
        new_obj = objective(labeling)
        ap = math.e ** ((new_obj - prev_obj)/T)
        if ap > random.random():
            print("SWITCH old %f new %f" % (prev_obj, new_obj))
            prev_obj = new_obj
            candidates = None
        else:
            labeling[new_labeling[2], new_labeling[0]] = 0
            labeling[new_labeling[1], new_labeling[0]] = 1
        i += 1
    T *= alpha
    print(T)

print(gerry_obj(labeling))
pops_district = np.matmul(labeling, populations)
print(pops_district.mean())
print(prev_obj)

