import codecs
import csv
from collections import defaultdict

def read_county_adjacency():
    fips_to_county = {}
    fips_to_state = {}
    adjacent_counties = defaultdict(set)

    with codecs.open('data/county_adjacency.txt', 'r', encoding='latin1') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        current_f = None
        for row in reader:
            c1, f1, c2, f2 = row
            if f1:
                f1 = int(f1)
            if f2:
                f2 = int(f2)
            if c1:
                current_f = f1
                fips_to_county[f1] = c1
                fips_to_state[f1] = c1.split(', ')[-1]    
            fips_to_county[f2] = c2
            fips_to_state[f2] = c2.split(', ')[-1]
            if current_f != f2:
                adjacent_counties[current_f].add(f2)
                adjacent_counties[f2].add(current_f)
    for k, v in adjacent_counties.items():
        s = fips_to_state[k]
        adjacent_counties[k] = frozenset(filter(lambda c: fips_to_state[c] == s, v))
    
    return (fips_to_county, fips_to_state, adjacent_counties)

def read_election_data():
    fips_to_election_data = {}
    with open('data/2016_US_County_Level_Presidential_Results.csv', 'r') as f:
        reader = csv.reader(f)
        it = iter(reader)
        next(it)
        for row in it:
            d, r, fips = row[1], row[2], row[10]
            fips_to_election_data[int(fips)] = (int(float(d)), int(float(r)))
    return fips_to_election_data