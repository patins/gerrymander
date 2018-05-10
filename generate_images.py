import shapefile
import math
from scipy.stats import binom
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.ops import cascaded_union
from search import build_state_adjacency_matrix, build_voter_data
import glob
import os

sf = shapefile.Reader("data/cb_2017_us_county_20m/cb_2017_us_county_20m.shp")

county_points = {}
for shape in sf.shapeRecords():
    assert shape.shape.shapeType == shapefile.POLYGON
    fips = int(''.join(shape.record[0:2]))
    assert fips not in county_points
    county_points[fips] = np.array(shape.shape.points)

def plot(file_name):
    state = file_name.split('/')[1]
    _, fips = build_state_adjacency_matrix(state)
    voter_data = build_voter_data(fips)
    populations = voter_data.sum(axis=0)

    labeling = np.load(file_name)

    district_pops = np.matmul(labeling, populations)
    district_dems = np.matmul(labeling, voter_data[0])
    p_republican = 1 - binom.cdf(district_pops/2, district_pops, 1 - district_dems/district_pops)
    
    all_points = np.concatenate([county_points[f] for f in fips])
    max_x, max_y = all_points.max(axis=0)
    min_x, min_y = all_points.min(axis=0)
    max_y, min_y = 90*np.sin(np.pi/180*np.array([max_y, min_y]))
    mid_x = (min_x + max_x)/2
    mid_y = (min_y + max_y)/2
    diff_x = (max_x - min_x)/2
    diff_y = (max_y - min_y)/2
    diff = max(diff_x, diff_y)
    min_x, max_x = mid_x - diff, mid_x + diff
    min_y, max_y = mid_y - diff, mid_y + diff
    
    fig, ax = plt.subplots()
    fig.set_size_inches((7,7))

    rb_cm = plt.get_cmap('bwr')
    p_cm = plt.get_cmap('Purples')

    def color(p_republican):
        x = 2*np.abs(p_republican - 0.5)
        pu = np.array(p_cm(1 - x)[:-1])
        rb = np.array(rb_cm(p_republican)[:-1])
        return x*rb + (1-x)*pu

    for district in range(labeling.shape[0]):
        for county in labeling[district, :].nonzero()[0]:
            f = fips[county]
            points = county_points[f]
            start = np.where((points == points[-1]).all(axis=1))[0][0]
            first_points = points[start:].copy()
            first_points[:, 1] = 90*np.sin(np.pi/180*first_points[:, 1])
            ax.add_patch(Polygon(
                first_points,
                True,
                color=color(p_republican[district]),
                ec=(0,0,0,0.1)
            ))
    
    for district in range(labeling.shape[0]):
        county_polygons = []
        for county in labeling[district, :].nonzero()[0]:
            f = fips[county]
            points = county_points[f]
            start = np.where((points == points[-1]).all(axis=1))[0][0]
            first_points = points[start:].copy()
            first_points[:, 1] = 90*np.sin(np.pi/180*first_points[:, 1])

            county_polygons.append(ShapelyPolygon(first_points))

        u = cascaded_union(county_polygons)

        if type(u) == ShapelyMultiPolygon:
            continue

        v = list(zip(*u.exterior.coords.xy))
        ax.add_patch(Polygon(
            v,
            True,
            fill=False,
            ec=(0,0,0,1)
        ))
    

    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)
    plt.axis('off')

def plot_districts(file):
    state = file.split('/')[1]
    _, fips = build_state_adjacency_matrix(state)
    voter_data = build_voter_data(fips)
    populations = voter_data.sum(axis=0)

    labeling = np.load(file)
    assert labeling.shape[0] <= 20
    
    all_points = np.concatenate([county_points[f] for f in fips])
    max_x, max_y = all_points.max(axis=0)
    min_x, min_y = all_points.min(axis=0)
    max_y, min_y = 90*np.sin(np.pi/180*np.array([max_y, min_y]))
    mid_x = (min_x + max_x)/2
    mid_y = (min_y + max_y)/2
    diff_x = (max_x - min_x)/2
    diff_y = (max_y - min_y)/2
    diff = max(diff_x, diff_y)
    min_x, max_x = mid_x - diff, mid_x + diff
    min_y, max_y = mid_y - diff, mid_y + diff
    
    color_map = plt.get_cmap('tab20')
    
    fig, ax = plt.subplots()
    fig.set_size_inches((7,7))

    for district in range(labeling.shape[0]):
        for county in labeling[district, :].nonzero()[0]:
            f = fips[county]
            points = county_points[f]
            start = np.where((points == points[-1]).all(axis=1))[0][0]
            first_points = points[start:].copy()
            first_points[:, 1] = 90*np.sin(np.pi/180*first_points[:, 1])
            ax.add_patch(Polygon(
                first_points,
                True,
                color=color_map(district),
                ec=(0,0,0,0.1)
            ))    

    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)
    plt.axis('off')


for fn in glob.glob('runs/*/*.npy'):
    img_name = 'imgs/' + fn[5:-4] + '.png'
    plot(fn)
    path = 'imgs/' + fn[5:].split('/')[0]
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(img_name, dpi=300)

    img_name = 'district_imgs/' + fn[5:-4] + '.png'
    plot_districts(fn)
    path = 'district_imgs/' + fn[5:].split('/')[0]
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(img_name, dpi=300)

    print("Generated images for %s" % fn)
