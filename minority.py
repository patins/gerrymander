from search import *
import numpy as np

STATE = 'NC'
N_DISTRICTS = 13

def pop_objective(labeling):
    pops_district = np.matmul(labeling, populations)
    average_population = populations.sum()/N_DISTRICTS
    return -10e-11 * ((pops_district - average_population)**2).sum()

def objective(labeling):
    pops_district = np.matmul(labeling, populations)
    pct_white_district = np.matmul(labeling, white) / pops_district
    return -((pct_white_district) ** 3).sum()

adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)
pct_white = build_race_data(fips)

populations = voter_data.sum(axis=0)

white = pct_white * populations

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)
initial_labeling = np.load('runs/NC/sa-population.npy')

TS = np.arange(0, 1.01, 0.1)

for t in TS:
    def comb_objective(labeling):
        return ((1-t) * objective(labeling)) + (t * pop_objective(labeling))

    labeling = simulated_annealing(initial_labeling, adj, comb_objective)

    np.save('runs/NC/sa-minority-L3-%s.npy' % str(t), labeling)
