from parser import read_county_adjacency, read_election_data
import numpy as np

fips_to_county, fips_to_state, adjacent_counties = read_county_adjacency()
election_data = read_election_data()

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

def validate_labeling(labeling):
    assert (labeling.sum(axis=0) == 1).all()
    assert (labeling.sum(axis=1) >= 1).all()

def build_initial_assignment(adjacency, populations, n_districts):
    n_blocks = adjacency.shape[0]
    assert n_blocks >= n_districts
    labels = np.zeros((n_districts, n_blocks), dtype=np.bool_)
    avg_population = populations.sum() / n_districts
    district = 0
    while not (labels.any(axis=0) == 1).all():
        # find the largest unlabeled block
        i = ((~labels.any(axis=0)) * populations).argmax()
        current_population = populations[i]
        labels[district, i] = 1

        if n_blocks - labels.any(axis=0).sum() == n_districts - labels.any(axis=1).sum():
            district += 1
            continue
        
        while True:
            if n_blocks - labels.any(axis=0).sum() == n_districts - labels.any(axis=1).sum():
                break
            # TODO seems sus
            adj_unlab = adjacency[labels[district, :], :][0] & (~labels.any(axis=0))
            possible_pops = adj_unlab * populations * (populations <= avg_population - current_population)
            if possible_pops.max() <= 0:
                break
            j = possible_pops.argmax()
            labels[district, j] = 1
            current_population += populations[j]

        district += 1
    assert district == n_districts
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


def label_hash(labeling):
    return labeling.data.tobytes()
