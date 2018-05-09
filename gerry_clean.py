from search import *
from scipy.stats import binom
import numpy as np

STATE = 'OH'
N_DISTRICTS = 16

def pop_objective(labeling):
    pops_district = np.matmul(labeling, populations)
    average_population = populations.sum()/N_DISTRICTS
    return -10e-9 * ((pops_district - average_population)**2).sum()

def objective(labeling):
    pops_district = np.matmul(labeling, populations)
    dems_distrct = np.matmul(labeling, voter_data[0])
    return (1 - binom.cdf(pops_district/2, pops_district, dems_distrct/pops_district)).sum()

adj, fips = build_state_adjacency_matrix(STATE)
voter_data = build_voter_data(fips)

populations = voter_data.sum(axis=0)

initial_labeling = build_initial_assignment(adj, populations, N_DISTRICTS)
initial_labeling = np.load('sa-population-oh.npy')

labeling = simulated_annealing(initial_labeling, adj, objective)

np.save('sa-gerry-dem-oh.npy', labeling)
