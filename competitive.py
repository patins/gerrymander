from search import *
from scipy.stats import binom
import numpy as np

STATE = 'OH'
N_DISTRICTS = 16

def pop_objective(labeling):
    pops_district = np.matmul(labeling, populations)
    average_population = populations.sum()/N_DISTRICTS
    return -10e-12 * ((pops_district - average_population)**2).sum()


def objective(labeling):
    # promote competitive elections
    # push probs to 0.5
    pops_district = np.matmul(labeling, populations)
    dems_district = np.matmul(labeling, voter_data[0])
    return -((binom.cdf(pops_district/2, pops_district, dems_district/pops_district) - 0.5) ** 2).sum()

"""
def objective(labeling):
    pops_district = np.matmul(labeling, populations)
    dems_district = np.matmul(labeling, voter_data[0])

    return -1e-8 * ((dems_district - 0.5*pops_district)**2).sum()
"""

adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)

populations = voter_data.sum(axis=0)

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)
initial_labeling = np.load('runs/OH/sa-population.npy')

TS = np.arange(0, 1.01, 0.1)

for t in TS:
    def comb_objective(labeling):
        return ((1-t) * objective(labeling)) + (t * pop_objective(labeling))

    labeling = simulated_annealing(initial_labeling, adj, comb_objective)

    np.save('runs/OH/sa-competitive-L2-binom-%s.npy' % str(t), labeling)
