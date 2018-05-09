from search import *
from scipy.stats import binom
import numpy as np

STATE = 'PA'
N_DISTRICTS = 18

def pop_objective(labeling):
    pops_district = np.matmul(labeling, populations)
    average_population = populations.sum()/N_DISTRICTS
    return -10e-12 * ((pops_district - average_population)**2).sum()

def objective(labeling):
    pops_district = np.matmul(labeling, populations)
    dems_district = np.matmul(labeling, voter_data[0])
    return -np.abs((1 - binom.cdf(pops_district/2, pops_district, dems_district/pops_district)).sum() - (N_DISTRICTS*dems))


adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)

populations = voter_data.sum(axis=0)
dems = voter_data[0].sum() / voter_data.sum()

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)
initial_labeling = np.load('runs/PA/sa-population.npy')

TS = np.arange(0, 1.01, 0.1)

for t in TS:
    def comb_objective(labeling):
        return ((1-t) * objective(labeling)) + (t * pop_objective(labeling))

    labeling = simulated_annealing(initial_labeling, adj, comb_objective)

    np.save('runs/PA/sa-pps-%s.npy' % str(t), labeling)
