#!/usr/bin/env python
# Copyright 2022 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import combinations
from math import asin, cos, radians, sin, sqrt
import matplotlib
import numpy as np
import pandas as pd
import shapefile
import dimod
from neal import SimulatedAnnealingSampler
from shapely.geometry import Point, shape
from shapely.ops import unary_union

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


def distance(lat1, long1, lat2, long2):
    """Compute the distance (in miles) between two latitude/longitude points."""
    long1 = radians(long1)
    long2 = radians(long2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlong = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlong / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956 
    return c * r


def get_existing_towers(filename):
    """Loads existing tower locations from a text file."""
    with open(filename) as f:
        lines = f.readlines()
    points = []
    lats = []
    longs = []
    for line in lines:
        temp = line.split("\t")
        points.append(temp[0])
        lats.append(float(temp[1]))
        longs.append(float(temp[2][:-2]))
    towers = pd.DataFrame({'Name': points, 'Latitude': lats, 'Longitude': longs})
    return towers


def gen_new_points(num_new_points, region_map):
    """Generates a random set of new locations within the region boundaries."""
    polys = [shape(p) for p in region_map.shapes()]
    boundary = unary_union(polys)
    min_long, min_lat, max_long, max_lat = boundary.bounds
    counter = 0
    new_locs = []
    while counter < num_new_points:
        new_long = (max_long - min_long) * np.random.random() + min_long
        new_lat = (max_lat - min_lat) * np.random.random() + min_lat
        point = Point(new_long, new_lat)
        if point.intersects(boundary):
            counter += 1
            new_locs.append([new_lat, new_long])
    return new_locs


def build_bqm(num_to_build, existing_towers, new_locs, radius, lambda_constraint):
    n = len(new_locs)

    all_points = []
    for _, row in existing_towers.iterrows():
        all_points.append((row['Latitude'], row['Longitude']))
    for pt in new_locs:
        all_points.append((pt[0], pt[1]))
    max_dist = 0
    for (p, q) in combinations(all_points, 2):
        d2 = distance(p[0], p[1], q[0], q[1]) ** 2
        if d2 > max_dist:
            max_dist = d2

    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    for i in range(n):
        for j in range(i + 1, n):
            d2 = distance(new_locs[i][0], new_locs[i][1], new_locs[j][0], new_locs[j][1]) ** 2
            bias = d2 if d2 < radius ** 2 else max_dist
            coeff = -bias + 2 * lambda_constraint
            bqm.add_interaction(f"x{i}", f"x{j}", coeff)

    for i in range(n):
        lin_term = 0
        for _, row in existing_towers.iterrows():
            d2 = distance(row['Latitude'], row['Longitude'], new_locs[i][0], new_locs[i][1]) ** 2
            bias = d2 if d2 < radius ** 2 else max_dist
            lin_term += -bias
        lin_term += lambda_constraint * (1 - 2 * num_to_build)
        bqm.add_variable(f"x{i}", lin_term)

    bqm.offset += lambda_constraint * (num_to_build ** 2)

    return bqm


def visualize(region_map, existing_towers, new_locs, build_sites):
    """Визуализирует исходную ситуацию и выбранные места для строительства."""
    print("\nVisualizing scenario and solution...")
    _, (ax, ax_final) = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))
    ax.axis('off')
    ax_final.axis('off')
    polys = [shape(p) for p in region_map.shapes()]
    boundary = unary_union(polys)
    for geom in boundary.geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.5, fc='#d3d3d3', ec='none', zorder=0)
        ax_final.fill(xs, ys, alpha=0.5, fc='#d3d3d3', ec='none', zorder=0)
    ax.scatter(existing_towers.Longitude, existing_towers.Latitude, color='r', zorder=2)
    ax_final.scatter(existing_towers.Longitude, existing_towers.Latitude, color='r', zorder=2)
    radius = 30
    ax.scatter(existing_towers.Longitude, existing_towers.Latitude, color='r', alpha=0.1, s=radius ** 2, zorder=1)
    ax_final.scatter(existing_towers.Longitude, existing_towers.Latitude, color='r', alpha=0.1, s=radius ** 2, zorder=1)
    new_locations = pd.DataFrame(new_locs, columns=['Latitude', 'Longitude'])
    ax.scatter(new_locations.Longitude, new_locations.Latitude, color='y', zorder=8)
    new_builds = pd.DataFrame(build_sites, columns=['Latitude', 'Longitude'])
    ax_final.scatter(new_builds.Longitude, new_builds.Latitude, color='b', zorder=8)
    ax_final.scatter(new_builds.Longitude, new_builds.Latitude, color='b', alpha=0.1, s=radius ** 2, zorder=8)
    ax.axis('scaled')
    ax_final.axis('scaled')
    ax.set_title("Potential Sites", fontsize=24)
    ax_final.set_title("Determined Sites", fontsize=24)
    plot_filename = 'map.png'
    plt.savefig(plot_filename)
    print("\nOutput saved as", plot_filename)


if __name__ == "__main__":
    print("\nLoading map and scenario...")
    shp_file = "data/germany_states.shp"
    germany_map = shapefile.Reader(shp_file, encoding='CP1252')
    existing_towers = get_existing_towers("data/locations.txt")
    num_new = 100
    new_locs = gen_new_points(num_new, germany_map)
    num_to_build = 10

    print("\nBuilding BQM for new tower placement...")
    lambda_constraint = 10000
    bqm = build_bqm(num_to_build, existing_towers, new_locs, radius=75, lambda_constraint=lambda_constraint)

    print("\nSubmitting BQM to solver using SimulatedAnnealingSampler...")
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=1000, label='Example - TV Towers BQM')

    best_sample = sampleset.first.sample
    build_sites = []
    for key, val in best_sample.items():
        if val == 1:
            idx = int(key[1:])
            build_sites.append(new_locs[idx])

    print("\nSelected", len(build_sites), "build sites.")
    visualize(germany_map, existing_towers, new_locs, build_sites)
