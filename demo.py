from itertools import combinations
from math import asin, cos, radians, sin, sqrt
import matplotlib
import numpy as np
import pandas as pd
import shapefile
import dimod
from qdeepsdk import QDeepHybridSolver
from shapely.geometry import Point, shape
from shapely.ops import unary_union

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


def distance(lat1, long1, lat2, long2):
    """Compute the distance (in miles) between two latitude/longitude points."""
    # Convert degrees to radians
    long1 = radians(long1)
    long2 = radians(long2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlong = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlong / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956  # Earth's radius in miles
    return c * r


def get_existing_towers(filename):
    """Load existing tower locations from a text file."""
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
    """Generate a random set of new locations within the region boundaries."""
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
    """
    Build a BQM for the new tower placement problem.

    Arguments:
      num_to_build: int, number of new towers to build.
      existing_towers: DataFrame with coordinates of existing towers.
      new_locs: list of [lat, long] potential locations for new towers.
      radius: interference limiting radius.
      lambda_constraint: penalty coefficient to enforce exactly num_to_build towers.

    Returns:
      bqm: a BinaryQuadraticModel describing the problem.
    """
    n = len(new_locs)

    # Collect all points (existing and new) to compute the maximum distance squared
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

    # Initialize BQM
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    # For candidate variables, use names "x0", "x1", ..., "x{n-1}"
    # Contribution from interference between candidates:
    for i in range(n):
        for j in range(i + 1, n):
            d2 = distance(new_locs[i][0], new_locs[i][1], new_locs[j][0], new_locs[j][1]) ** 2
            bias = d2 if d2 < radius ** 2 else max_dist
            # Term: -bias * x_i * x_j, plus penalty interaction from the constraint
            coeff = -bias + 2 * lambda_constraint
            bqm.add_interaction(f"x{i}", f"x{j}", coeff)

    # Interaction between existing towers and candidates.
    # Since existing towers are fixed (their value = 1), the contribution is the sum of -bias for each pair.
    for i in range(n):
        lin_term = 0
        for _, row in existing_towers.iterrows():
            d2 = distance(row['Latitude'], row['Longitude'], new_locs[i][0], new_locs[i][1]) ** 2
            bias = d2 if d2 < radius ** 2 else max_dist
            lin_term += -bias
        # Add linear penalty from the constraint: lambda*(1 - 2*num_to_build)
        lin_term += lambda_constraint * (1 - 2 * num_to_build)
        bqm.add_variable(f"x{i}", lin_term)

    # Add constant offset from the constraint (does not affect optimization)
    bqm.offset += lambda_constraint * (num_to_build ** 2)

    return bqm


def visualize(region_map, existing_towers, new_locs, build_sites):
    """Visualize the initial scenario and the selected build sites."""
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
    # Choose a sufficiently large λ (this value can be adjusted)
    lambda_constraint = 10000
    # Here, radius is interpreted as a threshold (radius² is compared with the squared distance)
    bqm = build_bqm(num_to_build, existing_towers, new_locs, radius=75, lambda_constraint=lambda_constraint)

    print("\nSubmitting BQM to solver using QDeepHybridSolver...")
    # Convert the BQM to a QUBO dictionary and offset
    qubo, offset = bqm.to_qubo()
    # Create a mapping from variable names (e.g. "x0", "x1", ...) to indices
    n = len(new_locs)
    matrix = np.zeros((n, n))
    for (i, j), value in qubo.items():
        idx_i = int(i[1:])
        idx_j = int(j[1:])
        matrix[idx_i, idx_j] = value

    # Initialize the QDeepHybridSolver
    solver = QDeepHybridSolver()
    solver.token = "your-auth-token-here"

    # Solve the problem by passing the NumPy array
    result = solver.solve(matrix)

    # Extract the best solution: variables "x0", "x1", ... that are equal to 1
    best_sample = result['sample']
    build_sites = []
    for key, val in best_sample.items():
        if val == 1:
            # Extract the candidate index from the variable name (if key is a string) or use it directly if integer
            idx = int(key) if isinstance(key, str) else key
            build_sites.append(new_locs[idx])

    print("\nSelected", len(build_sites), "build sites.")
    visualize(germany_map, existing_towers, new_locs, build_sites)
