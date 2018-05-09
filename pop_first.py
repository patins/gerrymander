from search import *
import numpy as np

STATE = 'PA'
N_DISTRICTS = 18

def objective(labeling):
    pops_district = np.matmul(labeling, populations)
    average_population = populations.sum()/N_DISTRICTS
    return -10e-11 * ((pops_district - average_population)**2).sum()


adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)

populations = voter_data.sum(axis=0)

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)

labeling = greedy_search(initial_labeling, adj, objective)

np.save('runs/PA/greedy-population.npy', labeling)
