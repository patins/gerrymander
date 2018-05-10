import glob
import numpy as np
from search import *
from scipy.stats import binom
import os

states = list(set([x.split('/')[1] for x in glob.glob('runs/*/*.npy')]))
fips_for_state = {state: build_state_adjacency_matrix(state)[1] for state in states}

for state in states:
    voter_data = build_voter_data(fips_for_state[state])
    populations = voter_data.sum(axis=0)
    n_districts = np.load('runs/%s/sa-population.npy' % state).shape[0]
    dems = voter_data[0].sum()/voter_data.sum()
    
    def pop_obj(lab):
        district_pops = np.matmul(lab, populations)
        return ((district_pops - (populations.sum()/n_districts)) ** 2).sum()
    
    def gerry_obj(lab):
        pops_district = np.matmul(lab, populations)
        dems_district = np.matmul(lab, voter_data[0])
        return (1 - binom.cdf(pops_district/2, pops_district, dems_district/pops_district) ** 2).sum()

    def comp_elec_obj(lab):
        pops_district = np.matmul(lab, populations)
        dems_district = np.matmul(lab, voter_data[0])
        return ((binom.cdf(pops_district/2, pops_district, dems_district/pops_district) - 0.5) ** 2).sum()
    
    def pps_obj(lab):
        pops_district = np.matmul(labeling, populations)
        dems_district = np.matmul(labeling, voter_data[0])
        return np.abs((1 - binom.cdf(pops_district/2, pops_district, dems_district/pops_district)).sum() - (n_districts*dems))
    
    print("# STATE: %s\n" % state)
    print("## Initial Labeling Info")
    
    if os.path.exists('runs/%s/greedy-population.npy' % state):
        population_labeling = np.load('runs/%s/greedy-population.npy' % state)
        district_pops = np.matmul(population_labeling, populations)
        print("### Local Search District Populations %.5E" % pop_obj(population_labeling))
        print("[" + ', '.join(map(str, district_pops.tolist())) + "]")
        print()
    else:
        print("### No Local Search District Population\n")
    
    population_labeling = np.load('runs/%s/sa-population.npy' % state)
    district_pops = np.matmul(population_labeling, populations)
    print("### SA District Populations %.5E" % pop_obj(population_labeling))
    print("[" + ', '.join(map(str, district_pops.tolist())) + "]")
    print()
    # gerry
    gerrys = glob.glob('runs/%s/greedy-gerry-???.npy' % state)
    if len(gerrys) > 0:
        for gerry in gerrys:
            labeling = np.load(gerry)
            print("## Gerrymanding: %s - %f" % (gerry.split('/')[-1][:-4], gerry_obj(labeling)))
            
            district_dems = np.matmul(labeling, voter_data[0])
            district_pops = np.matmul(labeling, populations)
            print("% of democrats in each district\n")
            print(district_dems/district_pops)
            print()
    """# pps
    if os.path.exists('runs/%s/sa-pps.npy' % state):
        labeling = np.load('runs/%s/sa-pps.npy' % state)
        print("## SA Prop Partisan")
        district_dems = np.matmul(labeling, voter_data[0])
        district_pops = np.matmul(labeling, populations)
        print("% of democrats in each district\n")
        print(district_dems/district_pops)
        print()
    """
    print("## Tradeoff with population objective\n")
    print("t = 0 does not consider population")
    td = glob.glob('runs/%s/sa-gerry-dem-[01].*.npy' % state)
    td.sort()
    if len(td) > 0:
        print("### Gerrymandering for Democrats")
        print("t | Gerry Objective (higher is more democrats) | Population Objective (lower is better)")
        print("---|---|---")
        for f in td:
            t = f.split('-')[-1][:-4]
            labeling = np.load(f)
            print("%s | %f | %.2E" % (t, gerry_obj(labeling), pop_obj(labeling)))
        print()

    td = glob.glob('runs/%s/sa-competitive-L2-binom-[01].*.npy' % state)
    td.sort()
    if len(td) > 0:
        print("### Competitive Elections")
        print("t | CE Objective (lower is better) | Population Objective (lower is better)")
        print("---|---|---")
        for f in td:
            t = f.split('-')[-1][:-4]
            labeling = np.load(f)
            print("%s | %f | %.2E" % (t, comp_elec_obj(labeling), pop_obj(labeling)))
        print()
    td = glob.glob('runs/%s/sa-pps-[01].*.npy' % state)
    td.sort()
    if len(td) > 0:
        print("### Prop Partisanship")
        print("t | PPS Objective (lower is better) | Population Objective (lower is better)")
        print("---|---|---")
        for f in td:
            t = f.split('-')[-1][:-4]
            labeling = np.load(f)
            print("%s | %f | %.2E" % (t, pps_obj(labeling), pop_obj(labeling)))
        print()
