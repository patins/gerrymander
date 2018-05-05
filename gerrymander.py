from search import *
from scipy.stats import binom

STATE = 'NY'
N_DISTRICTS = 27

def objective(labeling):
    pops_district = np.matmul(labeling, populations)
    dems_distrct = np.matmul(labeling, voter_data[0])
    return binom.cdf(pops_district/2, pops_district, dems_distrct/pops_district).sum()

def fair_population_objective(labeling):
    pops_district = np.matmul(labeling, populations)
    average_population = populations.sum()/N_DISTRICTS
    return -((pops_district - average_population)**2).sum()


adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)

populations = voter_data.sum(axis=0)

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)

assert all_districts_connected(initial_labeling, adj)

visited = set(label_hash(initial_labeling))

labeling = initial_labeling

best_obj = float('-inf')

for i in range(100):
    cur_obj = objective(labeling)
    if cur_obj > best_obj:
        print("improved obj old = %f, new = %f" % (best_obj, cur_obj))
        best_obj = cur_obj
    else:
        print("obj didn't increase")
        break
    labeling = search(labeling, adj, visited, objective)
    
