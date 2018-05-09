from parser import read_county_adjacency, read_election_data, read_race_data
import numpy as np
import math
import random

fips_to_county, fips_to_state, adjacent_counties = read_county_adjacency()
election_data = read_election_data()
race_data = read_race_data()

def fips_for_state(state):
    return [fips for fips, s in fips_to_state.items() if s == state]

def build_state_adjacency_matrix(state):
    state_fips = fips_for_state(state)
    state_fips.sort()
    n_blocks = len(state_fips)
    adj = np.zeros((n_blocks, n_blocks), dtype=np.bool_)
    for i, u in enumerate(state_fips):
        for v in adjacent_counties[u]:
            if v in state_fips:
                j = state_fips.index(v)
                adj[i, j] = 1
                adj[j, i] = 1
    adj.flags.writeable = 0
    return adj, state_fips

def build_voter_data(fips_set):
    voter_data = np.zeros((2, len(fips_set)), dtype=np.float32)
    for i, fips in enumerate(fips_set):
        voter_data[:, i] = election_data[fips]
    return voter_data

def build_race_data(fips_set):
    rd = np.zeros(len(fips_set), dtype=np.float32)
    for i, fips in enumerate(fips_set):
        rd[i] = race_data[fips]
    return rd

def validate_labeling(labeling):
    assert (labeling.sum(axis=0) == 1).all()
    assert (labeling.sum(axis=1) >= 1).all()

"""
def build_initial_assignment(adjacency, populations, n_districts):
    n_blocks = adjacency.shape[0]
    assert n_blocks >= n_districts
    labels = np.zeros((n_districts, n_blocks), dtype=np.bool_)
    avg_population = populations.sum() / n_districts
    district = 0
    while not (labels.any(axis=0) == 1).all():
        # find the largest unlabeled block
        i = ((~labels.any(axis=0)) * populations).argmin()
        current_population = populations[i]
        labels[district, i] = 1

        if n_blocks - labels.any(axis=0).sum() == n_districts - labels.any(axis=1).sum():
            district += 1
            continue
        
        while True:
            if n_blocks - labels.any(axis=0).sum() == n_districts - labels.any(axis=1).sum():
                break
            # TODO seems sus
            print(adjacency[labels[district, :].nonzero()[0], :][0])
            adj_unlab = adjacency[labels[district, :].nonzero()[0], :][0] & (~labels.any(axis=0))
            #adj_unlab = adjacency[labels[district, :], :][0] & (~labels.any(axis=0))
            possible_pops = adj_unlab * populations * (populations <= avg_population - current_population)
            if possible_pops.min() <= 0:
                break
            j = possible_pops.argmin()
            labels[district, j] = 1
            current_population += populations[j]

        district += 1
    assert district == n_districts
    validate_labeling(labels)
    return labels
"""

def build_initial_assignment(adjacency, populations, n_districts):
    n_blocks = adjacency.shape[0]
    assert n_blocks >= n_districts
    labels = np.zeros((n_districts, n_blocks), dtype=np.bool_)
    q = [0]
    visited = [0 for _ in range(n_blocks)]
    visited[0] = 1
    while len(q) > 0 and labels.any(axis=0).sum() <= n_blocks - n_districts:
        u = q.pop(0)
        labels[0, u] = 1

        for v in adjacency[u, :].nonzero()[0]:
            if not visited[v]:
                visited[v] = 1
                q.append(v)

    for i, j in enumerate(np.argwhere(labels.any(axis=0) == False)):
        labels[i+1, j] = 1

    validate_labeling(labels)

    return labels


def all_districts_connected(labeling, adjacency):
    n_districts, n_blocks = labeling.shape
    
    visited = np.zeros(n_blocks, dtype=np.bool_)
    
    for d in range(n_districts):
        # check if district d is connected
        start_block = labeling[d, :].nonzero()[0][0]
        q = [start_block]
        visited[:] = 0
        visited[start_block] = 1
        while len(q) > 0:
            u = q.pop()
            for v in (adjacency[u, :] & labeling[d, :]).nonzero()[0]:
                if not visited[v]:
                    visited[v] = 1
                    q.append(v)
        if not (visited == labeling[d, :]).all():
            return False
    return True

def search(labeling, adjacency, visited, objective):
    n_districts, n_blocks = labeling.shape
    
    max_obj_value = float('-inf')
    best_labeling = None
    
    for i in range(n_blocks):
            i_district = labeling[:, i].nonzero()[0][0]
            for j in adjacency[i, :].nonzero()[0]:
                assert i != j
                j_district = labeling[:, j].nonzero()[0][0]
                if i_district != j_district:
                    #try to grow i into j
                    labeling[j_district, j] = 0
                    labeling[i_district, j] = 1

                    if labeling[j_district, :].sum() >= 1:
                        # TODO here for debug
                        validate_labeling(labeling)
                        if all_districts_connected(labeling, adjacency):
                            obj_value = objective(labeling)
                            if obj_value > max_obj_value and label_hash(labeling) not in visited:
                                max_obj_value = obj_value
                                best_labeling = labeling.copy()

                    labeling[i_district, j] = 0
                    labeling[j_district, j] = 1
    return best_labeling

def neighbor(labeling, adjacency):
    n_districts, n_blocks = labeling.shape

    transitions = []

    for i in range(n_blocks):
            i_district = labeling[:, i].nonzero()[0][0]
            for j in adjacency[i, :].nonzero()[0]:
                assert i != j
                j_district = labeling[:, j].nonzero()[0][0]
                if i_district != j_district:
                    #try to grow i into j
                    labeling[j_district, j] = 0
                    labeling[i_district, j] = 1

                    if labeling[j_district, :].sum() >= 1:
                        # TODO here for debug
                        validate_labeling(labeling)
                        if all_districts_connected(labeling, adjacency):
                            transitions.append((j, j_district, i_district))

                    labeling[i_district, j] = 0
                    labeling[j_district, j] = 1
    return transitions

def greedy_search(initial_labeling, adjacency, objective):
    labeling = initial_labeling.copy()
    best_obj = objective(labeling)
    visited = set(label_hash(labeling))
    while True:
        labeling = search(labeling, adjacency, visited, objective)
        cur_obj = objective(labeling)
        print("objcetive = %f" % cur_obj)
        if cur_obj <= best_obj:
            return labeling
        best_obj = cur_obj

def simulated_annealing(initial_labeling, adjacency, objective):
    labeling = initial_labeling.copy()
    prev_obj = objective(labeling)

    T = 1.0
    T_min = 0.000001
    alpha = 0.9
    stop = False

    while T > T_min and not stop:
        i = 1
        candidates = None
        while i <= 100:
            if candidates == None:
                try:
                    candidates = neighbor(labeling, adjacency)
                except KeyboardInterrupt:
                    stop = True
                    break
            new_labeling = random.choice(candidates)
            labeling[new_labeling[1], new_labeling[0]] = 0
            labeling[new_labeling[2], new_labeling[0]] = 1
            new_obj = objective(labeling)
            #ap = math.e ** ((new_obj - prev_obj)/T)
            ap = 1/(1 + math.e ** ((-new_obj + prev_obj)/T))
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
    return labeling

def label_hash(labeling):
    return labeling.data.tobytes()
