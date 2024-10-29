import multipers as mp
import multipers.grids as mpg
import numpy as np
import gudhi as gd
from multipers.ml.convolutions import KDE as KernelDensity
import dionysus as d
from itertools import combinations
from collections import defaultdict


# =========================================================================
# From a point cloud to a bifiltration
# =========================================================================

def get_bifiltration(cloud):
    bifiltration = gd.RipsComplex(points=cloud).create_simplex_tree()
    bifiltration = mp.SimplexTreeMulti(bifiltration, num_parameters=2)

    codensity = - KernelDensity(bandwidth=0.2).fit(cloud).score_samples(cloud) 
    bifiltration.fill_lowerstar(codensity, parameter=1)
    
    # bifiltration.collapse_edges(-2)
    # bifiltration.expansion(2)

    return bifiltration


# =========================================================================
# From a bifiltration to a grid & intervals
# =========================================================================

# The four functions below are used to construct a grid and intervals from a bifiltration.
def is_smaller(node1, node2):
    if node1[0] <= node2[0] and node1[1] <= node2[1]:
        return True
    else:
        return False


def add_source(pt, points):
    for other in points:
        if is_smaller(other, pt) and other != pt:
            return False
    return True
    

def add_sink(pt, points):
    for other in points:
        if is_smaller(pt, other) and other != pt:
            return False
    return True


def get_src_snk(points):
    src = []
    snk = []
    for pt in points:
        if add_source(pt, points):
            src.append(pt)
        if add_sink(pt, points):
            snk.append(pt)
    return (src, snk)


# Based on a given bifiltration, this function creates a 4 x 4 grid
# To reduce computational cost, we've chosen a resolution parameter of 4 for now.
def get_grid(bifiltration, number):
    grid = mpg.compute_grid(bifiltration, strategy='regular', resolution=number)

    return grid


# We will generate all (1, 1)- and (2, 1)-intervals on a given grid. 
def get_int_2_1(grid):
    grid = [(coor_x, coor_y) for coor_x in grid[0] for coor_y in grid[1]]
    list_int = []

    # (1, 1)-intervals
    for c in grid:
        list_int.append(([c], [c]))
    for c in combinations(grid, 2):
        if is_smaller(c[0], c[1]):
            list_int.append(([c[0]], [c[1]]))
        elif is_smaller(c[1], c[0]):
            list_int.append(([c[1]], [c[0]]))

    # (2, 1)-intervals
    for c in combinations(grid, 3):
        # if is_convex(c, grid) and is_connected(c):
        tmp = get_src_snk(c)
        if len(tmp[0]) == 2 and len(tmp[1]) == 1:
            list_int.append((tmp[0], tmp[1]))
    
    return list_int


# =========================================================================
# to compute GRIs
# =========================================================================

def get_bifil_on_intervals(bifiltration, intervals):
    points = []

    for intv in intervals:
        corner = (intv[0], intv[1])
        first_max = (intv[0] + intv[5], intv[1] + intv[2])
        
        points += [corner, first_max]

        if intv[3]:
            first_min, second_min = (intv[0] - intv[3], intv[1]), (intv[0], intv[1] - intv[4])

            points += [first_min, second_min]

    points = list(set(points))

    bifil_on_intervals = {pt: [] for pt in points}

    for complex, coordinate in bifiltration:
        for pt in points:
            if is_smaller(coordinate, pt):
                bifil_on_intervals[pt].append(tuple(complex))

    return bifil_on_intervals


def get_ZZpersistence(bifil_on_intervals, intv, dimension):
    corner = (intv[0], intv[1])
    first_max = (intv[0] + intv[5], intv[1] + intv[2])

    if intv[3]:
        first_min, second_min = (intv[0] - intv[3], intv[1]), (intv[0], intv[1] - intv[4])
        ZZ_coordinates = [second_min, corner, first_min, first_max]
    else:
        ZZ_coordinates = [corner, first_max]

    # return a collection of complexes
    complexes = []
    for i in ZZ_coordinates:
        complexes += bifil_on_intervals[i]

    complexes = list(set(complexes))
    
    # generate inputs for Dionysus 2
    f = d.Filtration(complexes)

    in_out = {key: [] for key in complexes}

    for complex in bifil_on_intervals[ZZ_coordinates[0]]:
        in_out[complex].append(.1)

    step = .2

    for i in range(1, len(ZZ_coordinates)):
        enter_or_leave = list(set(bifil_on_intervals[ZZ_coordinates[i]]) ^ set(bifil_on_intervals[ZZ_coordinates[i - 1]]))

        for complex in enter_or_leave:
            in_out[complex].append(step)
        
        step += .1

    times = []

    for complex in complexes:
        times.append(in_out[complex])

    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    for i, dgm in enumerate(dgms):
        if i == dimension:
            return dgm
        
    return f"We can't find any {dimension}-dimensional cycle!"


# to check how many fully supported bar exists, i.e., GRIs
def count_fullbar(dgm):
    count = 0
    fullbar = d._dionysus.DiagramPoint(0.1, float('inf'))
    for bar in dgm:
        if bar == fullbar:
            count += 1

    return count


def GRI(bifil_on_interval, interval, dimension):

    return count_fullbar(get_ZZpersistence(bifil_on_interval, interval, dimension))


# =========================================================================
# to calculate Mobius inversion
# =========================================================================

def is_contained(interval_1, interval_2):
    maximal_1 = (interval_1[0] + interval_1[5], interval_1[1] + interval_1[2])
    maximal_2 = (interval_2[0] + interval_2[5], interval_2[1] + interval_2[2])
    
    minimal_1 = tuple(set(((interval_1[0] - interval_1[3], interval_1[1]), (interval_1[0], interval_1[1] - interval_1[4]))))
    minimal_2 = tuple(set(((interval_2[0] - interval_2[3], interval_2[1]), (interval_2[0], interval_2[1] - interval_2[4]))))

    for min_1 in minimal_1:
        not_contained = True

        for min_2 in minimal_2:
            if is_smaller(min_2, min_1):
                not_contained = False
                break
            
        if not_contained:
            return False
        
        not_contained = True
    
    if not is_smaller(maximal_1, maximal_2):
        return False
    
    return True


def generate_poset(intervals):
    num_intervals = len(intervals)

    poset = [[-1 for j in range(num_intervals)] for i in range(num_intervals)]

    for i in range(num_intervals):
        poset[i][i] = 1
    
    for i in range(num_intervals):
        for j in range(num_intervals):
            if poset[i][j] == -1:
                if is_contained(intervals[i], intervals[j]):
                    poset[i][j] = 1
                    poset[j][i] = 0
                else:
                    poset[i][j] = 0
                    if is_contained(intervals[j], intervals[i]):
                        poset[j][i] = 1
                    else:
                        poset[j][i] = 0
    return poset


# poset[index] = ([children], [parents])
def generate_tree(intervals, poset):
    num_intervals = len(intervals)
    tree = [[[], []] for _ in range(num_intervals)]

    for i in range(num_intervals):
        possible_parents = [index for index in range(num_intervals) if poset[i][index] == 1 and index != i]
        
        check = np.array([0 for _ in range(num_intervals)])
        for possible_parent in possible_parents:
            check += np.array(poset[possible_parent])
        for index, value in enumerate(check):
            if value == 1:
                tree[i][1].append(index)
                tree[index][0].append(i)
            
    return tree


def mobius_inversion(int_1, int_2, poset, tree, matrix):
    if matrix[int_1][int_2] != None:
        return matrix[int_1][int_2]
    else:
        value = 0
        visit = [False for _ in range(len(matrix))]
        visit[int_2] = True

        children = [child for child in tree[int_2][0] if poset[int_1][child]]
        
        while children:
            for child in children:
                if not visit[child]:
                    visit[child] = True
                    value -= mobius_inversion(int_1, child, poset, tree, matrix)
            
            next_children = []
            for child in children:
                next_children += [_ for _ in tree[child][0] if poset[int_1][_]]
            
            children = list(set(next_children))
            
        matrix[int_1][int_2] = value

        return matrix[int_1][int_2]


def mobius_matrix(intervals):
    num_intervals = len(intervals)
    mu = [[None for j in range(num_intervals)] for i in range(num_intervals)]

    for i in range(num_intervals):
        mu[i][i] = 1
    
    poset = generate_poset(intervals)
    tree = generate_tree(intervals, poset)

    for i in range(num_intervals):
        for j in range(num_intervals):
            if not poset[i][j]:
                mu[i][j] = 0
            else:
                mobius_inversion(i, j, poset, tree, mu)

    return mu


def GPD(bifiltration, intervals, dimension=0):
    GRIs = []
    mu = mobius_matrix(intervals)
    
    bifil_on_intervals = get_bifil_on_intervals(bifiltration, intervals)

    for intv in intervals:
        GRIs.append(GRI(bifil_on_intervals, intv, dimension=dimension))

    GPDs = {tuple(intv): gpd for (intv, gpd) in zip(intervals, np.matmul(mu, GRIs))}
    return GPDs
